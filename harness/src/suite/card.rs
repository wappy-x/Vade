//! Live runs and their cards: one card per run (`run-N/card.{json,md}`) and a
//! summary card over all runs (`card.{json,md}`) with the pass count, the
//! token-ID comparison per task and the grade.
use super::{Mode, Opts, RemoveOnDrop, TaskResult, date, fnv, load_tasks, n, new_dir, obj, pretty, run_task, s, serve, suite_version, upstream, write};
use crate::json::Value;
use serve::Source;
use std::process::Command;

/// Runs that a grade needs.
pub const GRADE_RUNS: usize = 3;
/// The share of task runs that must pass for grade B, in percent.
const B_PASS_PERCENT: usize = 90;

pub fn live(o: &Opts) -> Result<u8, String> {
    let Mode::Live { endpoint, model, runs } = &o.mode else { return Err("not a live run".into()) };
    let tasks = load_tasks(o)?;
    let out_arg = o.out.as_deref().ok_or("--live needs --out DIR")?;
    let addr = upstream(endpoint)?;
    // Every run uses the same workspace path, so tool results and prompts can match.
    // It is made before --out, so that a refusal leaves no output folder.
    let fixed = std::env::temp_dir().join("vade-harness-suite-live");
    std::fs::create_dir(&fixed).map_err(|e| match e.kind() {
        std::io::ErrorKind::AlreadyExists => format!("{}: {e}; another live suite run uses it, or an old run left it (remove it)", fixed.display()),
        _ => format!("{}: {e}", fixed.display()),
    })?;
    // Removed on every return, so a failed run does not block the next one.
    let guard = RemoveOnDrop(fixed);
    let out = new_dir(out_arg)?;
    let fixed = std::fs::canonicalize(&guard.0).map_err(|e| e.to_string())?;
    let version = Command::new(&o.harness).arg("--version").output().map(|x| String::from_utf8_lossy(&x.stdout).trim().to_string()).unwrap_or_else(|e| format!("unknown ({e})"));
    let mut all: Vec<Vec<TaskResult>> = Vec::new();
    for k in 1..=*runs {
        let run_dir = out.join(format!("run-{k}"));
        std::fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;
        let started = date();
        let mut results = Vec::new();
        for (dir, t) in &tasks {
            eprintln!("vade suite {} | run {k} | {} | start", date(), t.id);
            let r = run_task(o, dir, t.clone(), &run_dir.join("tasks").join(&t.id), Source::Upstream(addr.clone()), Some(&fixed));
            eprintln!("vade suite {} | run {k} | {} | {} | {}", date(), t.id, r.status, r.note);
            results.push(r);
        }
        let card = run_card(o, endpoint, model, &started, &version, &results);
        std::fs::write(run_dir.join("card.json"), pretty(&card, 0) + "\n").map_err(|e| e.to_string())?;
        std::fs::write(run_dir.join("card.md"), run_markdown(&card, k, &results)).map_err(|e| e.to_string())?;
        all.push(results);
    }
    let summary = summary(&suite_version(o), endpoint, model, &all);
    std::fs::write(out.join("card.json"), pretty(&summary.json, 0) + "\n").map_err(|e| e.to_string())?;
    std::fs::write(out.join("card.md"), &summary.markdown).map_err(|e| e.to_string())?;
    print!("{}", summary.markdown);
    eprintln!("vade suite {} | {} runs | grade {}", date(), runs, summary.grade);
    Ok(if summary.all_pass { 0 } else { 1 })
}

pub struct Summary { pub json: Value, pub markdown: String, pub all_pass: bool, pub grade: String }

/// Grade A needs every task run at this quality or above, and this mean.
const A_MIN_QUALITY: u8 = 2;
const A_MEAN_QUALITY: f64 = 2.5;
/// Grade B needs this mean quality.
const B_MEAN_QUALITY: f64 = 2.0;

fn quality(r: &TaskResult) -> u8 { r.quality.as_ref().map_or(0, |q| q.score) }

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    match v.len() { 0 => 0.0, l if l % 2 == 1 => v[l / 2], l => (v[l / 2 - 1] + v[l / 2]) / 2.0 }
}

/// One row per category over the given task runs (run number, result):
/// tasks, task runs completed, quality mean, median wall time and tokens.
pub fn categories(runs: &[(usize, &TaskResult)]) -> (Vec<Value>, String) {
    let mut names: Vec<&str> = runs.iter().map(|(_, r)| r.task.category.as_str()).collect();
    names.sort();
    names.dedup();
    let (mut rows, mut md) = (Vec::new(), String::from("| Category | Tasks | Completed | Quality mean (0-3) | Median wall (s) | Prompt tokens | Completion tokens |\n| --- | --- | --- | --- | --- | --- | --- |\n"));
    let mut row = |name: &str, list: Vec<&TaskResult>| {
        let mut ids: Vec<&str> = list.iter().map(|r| r.task.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        let done = list.iter().filter(|r| r.status == "PASS").count();
        let mean = list.iter().map(|r| f64::from(quality(r))).sum::<f64>() / list.len().max(1) as f64;
        let wall = median(list.iter().map(|r| r.seconds).collect());
        let ex: Vec<serve::Exchange> = list.iter().flat_map(|r| r.exchanges.clone()).collect();
        let (p, c) = (sum(&ex, |x| x.prompt_tokens), sum(&ex, |x| x.completion_tokens));
        let shown = |v: &Value| if *v == Value::Null { "-".to_string() } else { write(v) };
        md.push_str(&format!("| {name} | {} | {done}/{} | {mean:.2} | {wall:.1} | {} | {} |\n", ids.len(), list.len(), shown(&p), shown(&c)));
        rows.push(obj(vec![("category", s(name)), ("tasks", n(ids.len())), ("task_runs", n(list.len())), ("completed", n(done)),
            ("quality_mean", n(format!("{mean:.2}"))), ("median_wall_seconds", n(format!("{wall:.1}"))), ("prompt_tokens", p), ("completion_tokens", c)]));
    };
    for name in &names { row(name, runs.iter().filter(|(_, r)| r.task.category == *name).map(|(_, r)| *r).collect()); }
    row("all", runs.iter().map(|(_, r)| *r).collect());
    (rows, md)
}

/// One line per task run that did not complete, with its first wrong tool call.
pub fn failures(runs: &[(usize, &TaskResult)]) -> (Vec<Value>, String) {
    let (mut rows, mut md) = (Vec::new(), String::new());
    for (k, r) in runs.iter().filter(|(_, r)| r.status != "PASS") {
        let wrong = r.first_wrong_call();
        md.push_str(&format!("- {} {} (run {k}): {}; first wrong tool call: {wrong}\n", r.status, r.task.id, r.note));
        rows.push(obj(vec![("id", s(&r.task.id)), ("run", n(k)), ("status", s(&r.status)), ("note", s(&r.note)), ("first_wrong_tool_call", s(&wrong))]));
    }
    if md.is_empty() { md.push_str("None.\n"); }
    (rows, md)
}

/// Where a run first parts from the first run: `Ok(Some(k))` when the
/// request of model call k (from 0) differs, so that tool output changed the
/// conversation and the calls from k on cannot be compared; `Ok(None)` when
/// every call is the same; `Err((k, false))` when call k has the same request
/// but other token IDs; `Err((k, true))` when the calls before k are the same
/// and one run made more calls.
fn parts(r0: &TaskResult, r: &TaskResult) -> Result<Option<usize>, (usize, bool)> {
    let (q0, q, d0, d) = (r0.call_requests(), r.call_requests(), r0.call_digests(), r.call_digests());
    for k in 0..q0.len().min(q.len()) {
        if q0[k] != q[k] { return Ok(Some(k)); }
        if d0[k] != d[k] { return Err((k, false)); }
    }
    if q0.len() == q.len() { Ok(None) } else { Err((q0.len().min(q.len()), true)) }
}

/// Compares the runs task by task. A call whose request differs from the first
/// run (a tool result that holds a time, for example) ends the comparison for
/// that run. A task is exact when all of these are true: every run streamed
/// token IDs; every model call that got the same request as in the first run
/// gave the same IDs; in each run, the first request is the same, so at least
/// one call was compared; and, when no request differs, every run made the
/// same number of calls. docs/harness-suite.md lists the row texts.
pub fn summary(version: &str, endpoint: &str, model: &str, runs: &[Vec<TaskResult>]) -> Summary {
    let first = runs.first().map(Vec::as_slice).unwrap_or(&[]);
    let (mut rows, mut lines) = (Vec::new(), String::new());
    let (mut task_runs, mut passes, mut exact_tasks, mut timeouts) = (0, 0, 0, 0);
    for (i, r0) in first.iter().enumerate() {
        let per: Vec<&TaskResult> = runs.iter().filter_map(|r| r.get(i)).collect();
        let pass = per.iter().filter(|r| r.status == "PASS").count();
        let streamed = per.iter().all(|r| r.token_count() > 0);
        let found: Vec<Result<Option<usize>, (usize, bool)>> = per.iter().skip(1).map(|r| parts(r0, r)).collect();
        let differ = found.iter().filter_map(|f| f.err()).min();
        let input = found.iter().filter_map(|f| f.ok().flatten()).min();
        // A run whose first request already differs compared no call.
        let exact = per.len() >= 2 && streamed && differ.is_none() && input != Some(0);
        let exact_text = if per.len() < 2 { "-".to_string() } else if !streamed { "no token IDs".into() } else {
            match (differ, input) {
                (Some((c, false)), _) => format!("no (model call {} differs with the same request)", c + 1),
                (Some((c, true)), _) => format!("no (one run made more calls; the first {c} are the same)"),
                (None, Some(0)) => "not compared (the first request differs)".into(),
                (None, Some(c)) => format!("yes up to call {} (its request differs: tool output)", c + 1),
                (None, None) => "yes".into(),
            }
        };
        task_runs += per.len();
        passes += pass;
        exact_tasks += usize::from(exact);
        timeouts += per.iter().flat_map(|r| &r.steps).filter(|s| s.timed_out).count();
        let digests: Vec<Value> = per.iter().map(|r| s(&format!("{:016x}", fnv(r.call_digests())))).collect();
        let scores: Vec<String> = per.iter().map(|r| quality(r).to_string()).collect();
        lines.push_str(&format!("| {} | {} | {pass}/{} | {} | {exact_text} | {} |\n", r0.task.id, r0.task.category, per.len(), scores.join(" "), per.iter().map(|r| r.status.as_str()).collect::<Vec<_>>().join(" ")));
        rows.push(obj(vec![
            ("id", s(&r0.task.id)), ("category", s(&r0.task.category)), ("passes", n(pass)), ("runs", n(per.len())),
            ("statuses", Value::Array(per.iter().map(|r| s(&r.status)).collect())), ("quality", Value::Array(per.iter().map(|r| n(quality(r))).collect())),
            ("exact", Value::Bool(exact)), ("exact_note", s(&exact_text)), ("token_digests", Value::Array(digests)),
        ]));
    }
    let tasks = first.len();
    let numbered: Vec<(usize, &TaskResult)> = runs.iter().enumerate().flat_map(|(k, r)| r.iter().map(move |x| (k + 1, x))).collect();
    let (category_rows, category_md) = categories(&numbered);
    let (failed_rows, failed_md) = failures(&numbered);
    let mean = numbered.iter().map(|(_, r)| f64::from(quality(r))).sum::<f64>() / numbered.len().max(1) as f64;
    let low = numbered.iter().all(|(_, r)| quality(r) >= A_MIN_QUALITY);
    let all_pass = passes == task_runs && task_runs > 0;
    let grade = if runs.len() < GRADE_RUNS {
        format!("none (needs {GRADE_RUNS} runs)")
    } else if all_pass && exact_tasks == tasks && timeouts == 0 && low && mean >= A_MEAN_QUALITY {
        "A".into()
    } else if passes * 100 >= task_runs * B_PASS_PERCENT && exact_tasks == tasks && mean >= B_MEAN_QUALITY {
        "B".into()
    } else {
        "F".into()
    };
    let row = format!("| {version} | {model} | {tasks} tasks | {} runs | {passes}/{task_runs} task runs pass | quality {mean:.2} | {exact_tasks}/{tasks} tasks exact | grade {grade} |", runs.len());
    let markdown = format!("# Vade agent suite summary\n\nEndpoint: {endpoint}. Model: {model}.\n\nCard row:\n\n{row}\n\n## Categories\n\n{category_md}\n## Failed task runs\n\n{failed_md}\n## Tasks\n\n| Task | Category | Passes | Quality per run | Same token IDs for the same request | Status per run |\n| --- | --- | --- | --- | --- | --- |\n{lines}\nTimed-out steps: {timeouts}. Grade A needs {GRADE_RUNS} runs, every task run passing with quality {A_MIN_QUALITY} or more, a quality mean of {A_MEAN_QUALITY} or more, every task exact and no timed-out step. Grade B needs at least {B_PASS_PERCENT} % of task runs passing, a quality mean of {B_MEAN_QUALITY} or more and every task exact. See docs/harness-suite.md.\n");
    let json = obj(vec![
        ("card", s("vade-agent-suite-summary")), ("suite_version", s(version)), ("endpoint", s(endpoint)), ("model", s(model)), ("runs", n(runs.len())),
        ("tasks", n(tasks)), ("task_runs", n(task_runs)), ("passes", n(passes)), ("quality_mean", n(format!("{mean:.2}"))), ("exact_tasks", n(exact_tasks)),
        ("timed_out_steps", n(timeouts)), ("grade", s(&grade)), ("card_row", s(&row)), ("categories", Value::Array(category_rows)),
        ("failed", Value::Array(failed_rows)), ("per_task", Value::Array(rows)),
    ]);
    Summary { json, markdown, all_pass, grade }
}

fn sum(ex: &[serve::Exchange], f: impl Fn(&serve::Exchange) -> Option<u64>) -> Value {
    let vals: Vec<u64> = ex.iter().filter_map(&f).collect();
    if vals.is_empty() { Value::Null } else { n(vals.iter().sum::<u64>()) }
}

fn native_field(line: &str, label: &str) -> Option<u64> {
    let i = line.find(label)? + label.len();
    line[i..].trim_start().chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
}

fn run_card(o: &Opts, endpoint: &str, model: &str, started: &str, version: &str, results: &[TaskResult]) -> Value {
    let git = |args: &[&str]| Command::new("git").arg("-C").arg(&o.suite).args(args).output().map(|x| String::from_utf8_lossy(&x.stdout).trim().to_string()).unwrap_or_default();
    let tasks: Vec<Value> = results.iter().map(|r| {
        let ex = &r.exchanges;
        let natives: Vec<&String> = r.steps.iter().flat_map(|s| &s.native).collect();
        obj(vec![
            ("id", s(&r.task.id)), ("title", s(&r.task.title)), ("category", s(&r.task.category)), ("mode", s(&r.task.mode)),
            ("status", s(&r.status)), ("note", s(&r.note)),
            ("quality", n(quality(r))), ("quality_misses", Value::Array(r.quality.iter().flat_map(|q| &q.misses).map(|m| s(m)).collect())),
            ("changed_lines", r.quality.as_ref().map_or(Value::Null, |q| n(q.changed_lines))),
            ("first_wrong_tool_call", if r.status == "PASS" { Value::Null } else { s(&r.first_wrong_call()) }),
            ("needs", Value::Array(r.task.needs.iter().map(|x| s(x)).collect())),
            ("checks_passed", n(r.checks.iter().filter(|c| c.pass).count())), ("checks_total", n(r.checks.len())),
            ("model_calls", n(ex.iter().filter(|x| !x.held).count())), ("held_calls", n(ex.iter().filter(|x| x.held).count())),
            ("tool_turns", n(natives.iter().filter_map(|l| native_field(l, "tool turns")).sum::<u64>())),
            ("tool_executions", n(natives.iter().filter_map(|l| native_field(l, "tool executions")).sum::<u64>())),
            ("prompt_tokens", sum(ex, |x| x.prompt_tokens)), ("completion_tokens", sum(ex, |x| x.completion_tokens)),
            ("cached_tokens", sum(ex, |x| x.cached_tokens)),
            ("peak_prompt_tokens", ex.iter().filter_map(|x| x.prompt_tokens).max().map_or(Value::Null, n)),
            ("streamed_token_ids", n(r.token_count())), ("token_digest", s(&format!("{:016x}", fnv(r.call_digests())))),
            ("wall_seconds", n(format!("{:.1}", r.seconds))),
            ("steps", Value::Array(r.steps.iter().map(|st| obj(vec![
                ("kind", s(&st.kind)), ("exit", st.exit.map_or(Value::Null, n)), ("seconds", n(format!("{:.1}", st.seconds))),
                ("timed_out", Value::Bool(st.timed_out)), ("native_run", Value::Array(st.native.iter().map(|x| s(x)).collect())),
            ])).collect())),
            ("checks", Value::Array(r.checks.iter().map(|c| obj(vec![("check", s(&c.check)), ("pass", Value::Bool(c.pass)), ("detail", s(&c.detail))])).collect())),
        ])
    }).collect();
    let count = |st: &str| results.iter().filter(|r| r.status == st).count();
    let all: Vec<serve::Exchange> = results.iter().flat_map(|r| r.exchanges.clone()).collect();
    obj(vec![
        ("card", s("vade-agent-suite")), ("suite_version", s(&suite_version(o))),
        ("started", s(started)), ("finished", s(&date())), ("source", s("live")),
        ("endpoint", s(endpoint)), ("model", s(model)),
        ("harness", obj(vec![("path", s(&o.harness.display().to_string())), ("version", s(version))])),
        ("harness_args", Value::Array(o.harness_args.iter().map(|x| s(x)).collect())),
        ("fixtures", obj(vec![("path", s(&o.suite.display().to_string())), ("git_head", s(&git(&["rev-parse", "HEAD"]))),
            ("dirty", Value::Bool(!git(&["status", "--porcelain", "--", "."]).is_empty()))])),
        ("totals", obj(vec![
            ("tasks", n(results.len())), ("pass", n(count("PASS"))), ("fail", n(count("FAIL"))),
            ("unsupported", n(count("UNSUPPORTED"))), ("error", n(count("ERROR"))),
            ("model_calls", n(all.iter().filter(|x| !x.held).count())),
            ("prompt_tokens", sum(&all, |x| x.prompt_tokens)), ("completion_tokens", sum(&all, |x| x.completion_tokens)),
            ("wall_seconds", n(format!("{:.1}", results.iter().map(|r| r.seconds).sum::<f64>()))),
        ])),
        ("tasks", Value::Array(tasks)),
    ])
}

fn run_markdown(card: &Value, run: usize, results: &[TaskResult]) -> String {
    let g = |v: Option<&Value>, k: &str| v.and_then(|v| v.get(k)).map(|x| match x { Value::String(t) => t.clone(), Value::Null => "-".into(), other => write(other) }).unwrap_or_default();
    let (totals, fx) = (card.get("totals"), card.get("fixtures"));
    let c = Some(card);
    let mut m = format!("# Vade agent suite card\n\n| Field | Value |\n| --- | --- |\n| Suite | {} |\n| Endpoint | {} |\n| Model | {} |\n| Harness | {} |\n| Fixtures | {} at {}{} |\n| Started | {} |\n| Finished | {} |\n| Result | {} pass, {} fail, {} unsupported, {} error of {} tasks |\n| Model calls | {} |\n| Prompt tokens | {} |\n| Completion tokens | {} |\n| Wall time (s) | {} |\n\n",
        g(c, "suite_version"), g(c, "endpoint"), g(c, "model"), g(card.get("harness"), "version"), g(fx, "path"), g(fx, "git_head"),
        if fx.and_then(|f| f.get("dirty")) == Some(&Value::Bool(true)) { " (uncommitted changes)" } else { "" },
        g(c, "started"), g(c, "finished"), g(totals, "pass"), g(totals, "fail"), g(totals, "unsupported"), g(totals, "error"), g(totals, "tasks"),
        g(totals, "model_calls"), g(totals, "prompt_tokens"), g(totals, "completion_tokens"), g(totals, "wall_seconds"));
    let numbered: Vec<(usize, &TaskResult)> = results.iter().map(|r| (run, r)).collect();
    m.push_str(&format!("## Categories\n\n{}\n## Failed task runs\n\n{}\n## Tasks\n\n", categories(&numbered).1, failures(&numbered).1));
    m.push_str("| Task | Category | Status | Checks | Quality | Model calls | Tool turns | Prompt tokens | Peak prompt | Completion tokens | Token digest | Wall (s) |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for t in card.get("tasks").and_then(Value::as_array).unwrap_or(&[]) {
        let t = Some(t);
        m.push_str(&format!("| {} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} |\n", g(t, "id"), g(t, "category"), g(t, "status"), g(t, "checks_passed"), g(t, "checks_total"),
            g(t, "quality"), g(t, "model_calls"), g(t, "tool_turns"), g(t, "prompt_tokens"), g(t, "peak_prompt_tokens"), g(t, "completion_tokens"), g(t, "token_digest"), g(t, "wall_seconds")));
    }
    m.push_str("\n## Notes per task\n\n");
    for r in results {
        m.push_str(&format!("- **{}** ({}): {}\n", r.task.id, r.status, r.note));
        for c in r.checks.iter().filter(|c| !c.pass) { m.push_str(&format!("  - FAIL `{}`: {}\n", c.check, c.detail)); }
        for q in r.quality.iter().flat_map(|q| &q.misses) { m.push_str(&format!("  - quality: {q}\n")); }
    }
    m.push_str("\nQuality is the rubric score (0-3) from rubric.txt next to task.txt. Token counts come from the `usage` objects in the server replies, summed over all model calls of a task. A dash means the server reported none. Tool turns come from the harness `Native run:` lines. The token digest is FNV-1a over the list of per-call digests, and each per-call digest is FNV-1a over the streamed token IDs of one answered model call.\n");
    m
}

#[cfg(test)]
mod tests {
    use super::super::{check, rubric, task};
    use super::*;

    fn result(id: &str, status: &str, ids: &[&[u64]], timed_out: bool) -> TaskResult {
        let t = task::parse(&format!("id: {id}\ncategory: bug-fix\nstep: new\nprompt: p\ncheck: last-exit-zero\n")).unwrap();
        let exchanges = ids.iter().enumerate().map(|(i, x)| serve::Exchange { call: i + 1, token_ids: x.to_vec(), ..Default::default() }).collect();
        let steps = vec![super::super::StepResult { kind: "new".into(), exit: Some(0), seconds: 1.0, native: Vec::new(), timed_out }];
        let checks = vec![check::Outcome { check: "last-exit-zero".into(), pass: status == "PASS", detail: String::new() }];
        let score = if status == "PASS" { 3 } else { 0 };
        let quality = Some(rubric::Quality { score, misses: Vec::new(), changed_lines: 0 });
        TaskResult { task: t, status: status.into(), note: String::new(), checks, steps, exchanges, seconds: 1.0, replies_left: None, quality }
    }

    #[test]
    fn grade_needs_three_exact_passing_runs() {
        let run = || vec![result("a", "PASS", &[&[1, 2], &[3]], false), result("b", "PASS", &[&[4]], false)];
        let s3 = summary("agent-suite-2", "http://h/v1/chat/completions", "m", &[run(), run(), run()]);
        assert_eq!(s3.grade, "A");
        assert!(s3.all_pass);
        assert!(s3.markdown.contains("| agent-suite-2 | m | 2 tasks | 3 runs | 6/6 task runs pass | quality 3.00 | 2/2 tasks exact | grade A |"), "{}", s3.markdown);
        assert!(s3.markdown.contains("Endpoint: http://h/v1/chat/completions."), "{}", s3.markdown);
        assert_eq!(summary("v", "e", "m", &[run(), run()]).grade, "none (needs 3 runs)");
        // One different token in call 2 of task a: not exact, so no A or B.
        let other = vec![result("a", "PASS", &[&[1, 2], &[9]], false), result("b", "PASS", &[&[4]], false)];
        let s = summary("v", "e", "m", &[run(), run(), other]);
        assert_eq!(s.grade, "F");
        assert!(s.markdown.contains("no (model call 2 differs with the same request)"), "{}", s.markdown);
        // Tool output that changes the request of call 2 ends the comparison there: still exact.
        let mut timed = run();
        timed[0].exchanges[1].request = 7;
        timed[0].exchanges[1].token_ids = vec![8, 8];
        let s = summary("v", "e", "m", &[run(), run(), timed]);
        assert_eq!(s.grade, "A");
        assert!(s.markdown.contains("yes up to call 2 (its request differs: tool output)"), "{}", s.markdown);
        // The same request with other token IDs is a model difference, also after an equal call.
        let mut same_request = run();
        same_request[0].exchanges[0].token_ids = vec![1, 5];
        same_request[0].exchanges[1].request = 7;
        assert!(summary("v", "e", "m", &[run(), run(), same_request]).markdown.contains("no (model call 1 differs"));
        // A first request that differs compares nothing: not exact.
        let mut first = run();
        first[0].exchanges[0].request = 7;
        let s = summary("v", "e", "m", &[run(), run(), first]);
        assert!(s.markdown.contains("not compared (the first request differs)") && s.grade == "F", "{}", s.markdown);
        // One run with one more call after equal calls.
        let longer = vec![result("a", "PASS", &[&[1, 2], &[3], &[4]], false), result("b", "PASS", &[&[4]], false)];
        assert!(summary("v", "e", "m", &[run(), run(), longer]).markdown.contains("no (one run made more calls; the first 2 are the same)"));
        // A failing task run with every task exact gives B when 90 % pass.
        let fail = vec![result("a", "FAIL", &[&[1, 2], &[3]], false), result("b", "PASS", &[&[4]], false)];
        let many: Vec<Vec<TaskResult>> = (0..5).map(|_| run()).chain(std::iter::once(fail)).collect();
        let s = summary("v", "e", "m", &many);
        assert_eq!((s.grade.as_str(), s.all_pass), ("B", false));
        // A timed-out step blocks A.
        let slow = vec![result("a", "PASS", &[&[1, 2], &[3]], true), result("b", "PASS", &[&[4]], false)];
        assert_eq!(summary("v", "e", "m", &[run(), run(), slow]).grade, "B");
        // A task run with quality 1 blocks A even when every check passes.
        let mut sloppy = run();
        sloppy[0].quality.as_mut().unwrap().score = 1;
        assert_eq!(summary("v", "e", "m", &[run(), run(), sloppy]).grade, "B");
        // A server that streams no token IDs cannot prove exactness.
        let none = || vec![result("a", "PASS", &[&[]], false)];
        let s = summary("v", "e", "m", &[none(), none(), none()]);
        assert_eq!(s.grade, "F");
        assert!(s.markdown.contains("no token IDs"));
    }

    #[test]
    fn categories_and_first_wrong_call() {
        let mut a = result("a", "FAIL", &[&[1], &[2]], false);
        a.task.rubric = rubric::parse("may_change: src/lib.rs\nmax_changed_lines: 4\nanswer_max_chars: 100\n").unwrap();
        a.exchanges[0].calls = vec![("workspace_read".into(), r#"{"path":"src/v1/x.rs"}"#.into())];
        a.exchanges[1].results = vec!["Read refused: This folder does not exist.".into()];
        let mut b = result("b", "PASS", &[&[3]], false);
        b.task.category = "docs".into();
        b.seconds = 4.0;
        let runs = [(1, &a), (1, &b), (2, &b)];
        let (rows, md) = categories(&runs);
        assert!(md.contains("| bug-fix | 1 | 0/1 | 0.00 | 1.0 | - | - |"), "{md}");
        assert!(md.contains("| docs | 1 | 2/2 | 3.00 | 4.0 | - | - |"), "{md}");
        assert!(md.contains("| all | 2 | 2/3 | 2.00 | 4.0 | - | - |"), "{md}");
        assert_eq!(rows.len(), 3);
        let (_, md) = failures(&runs);
        assert_eq!(md, "- FAIL a (run 1): ; first wrong tool call: call 1 workspace_read {\"path\":\"src/v1/x.rs\"} -> Read refused: This folder does not exist.\n");
        // A write outside may_change is wrong even when the harness applies it.
        a.exchanges[0].calls = vec![("apply_patch".into(), r#"{"patch":"*** Begin Patch\n*** Update File: tests/t.rs\n*** End Patch\n"}"#.into())];
        a.exchanges[1].results = vec!["Applied 1 file operations.".into()];
        assert!(a.first_wrong_call().ends_with("-> writes tests/t.rs, outside the rubric's may_change"), "{}", a.first_wrong_call());
        a.exchanges[0].calls = vec![("workspace_read".into(), r#"{"path":"src/lib.rs"}"#.into())];
        assert!(a.first_wrong_call().starts_with("none refused or outside the rubric; last tool call: call 1 workspace_read"), "{}", a.first_wrong_call());
    }

    #[test]
    fn markdown_lists_failed_checks() {
        let r = vec![result("a", "FAIL", &[&[1]], false)];
        let card = obj(vec![("tasks", Value::Array(vec![obj(vec![("id", s("a")), ("status", s("FAIL"))])]))]);
        let m = run_markdown(&card, 1, &r);
        assert!(m.contains("| a |  | FAIL |"), "{m}");
        assert!(m.contains("FAIL `last-exit-zero`"), "{m}");
        assert!(m.contains("- FAIL a (run 1): ; first wrong tool call: no tool call"), "{m}");
    }
}
