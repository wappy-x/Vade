//! `vade doctor` results: one row per check (`state  name  detail`, fixed
//! columns), the fix under each FAIL row, then one PASS or FAIL line. With
//! `--json` the same rows become one `vade.doctor.v1` object instead.
use crate::{
    evals::json::{self, Value},
    output,
    style::{paint, Stream, Token},
};
use std::{cell::Cell, time::Instant};

/// The `--json` answer this command owes: its schema, the selected setup
/// and whether it is a dry run.
#[derive(Clone, Copy)]
pub struct Answer { pub schema: &'static str, pub mode: &'static str, pub dry: bool }

thread_local! {
    static JSON: Cell<Option<Answer>> = const { Cell::new(None) };
    static ANSWERED: Cell<bool> = const { Cell::new(false) };
}

/// Select `--json` output on this thread (`main` sets it once; read-only
/// commands run on the main thread).
pub fn set_json(answer: Option<Answer>) { JSON.set(answer); ANSWERED.set(false); }
/// Whether this command prints JSON.
pub fn json() -> bool { JSON.get().is_some() }

/// Print the command's one JSON object.
pub fn print_json(value: &Value) -> Result<(), String> {
    ANSWERED.set(true);
    output::line(format_args!("{}", json::packed(value)))
}

/// The JSON object for an error that ended a `--json` command before it
/// printed its own: a doctor report with one failed row, or the schema with
/// `error` and `next`. `None` outside `--json` or once an object was printed.
pub fn unanswered(error: &str) -> Option<Value> {
    let answer = JSON.get().filter(|_| !ANSWERED.get())?;
    let (fact, hint) = crate::style::split(error);
    if answer.schema == "vade.doctor.v1" {
        let mut report = Report::new(answer.mode, answer.dry);
        report.counts[State::Fail as usize] += 1;
        report.rows.push(row(State::Fail, "doctor", &fact, hint.as_deref()));
        return Some(report.value());
    }
    Some(json::object(vec![("schema", answer.schema.into()), ("error", fact.as_str().into()), ("next", hint.map_or(Value::Null, |hint| hint.as_str().into()))]))
}

fn row(state: State, name: &str, detail: &str, next: Option<&str>) -> Value {
    json::object(vec![
        ("name", name.into()),
        ("state", state.word().to_ascii_lowercase().into()),
        ("detail", detail.into()),
        ("next", next.map_or(Value::Null, Into::into)),
    ])
}

/// Returned by [`Report::finish`] after a FAIL that was already reported in
/// full (the JSON object, which carries every failure); `main` exits 1
/// without printing it.
pub const REPORTED: &str = "\u{0}vade: failure already reported";

/// What one check found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// The check passed.
    Ok,
    /// The check failed; its row names the fix.
    Fail,
    /// The check did not apply or could not run yet; never a failure.
    Skip,
    /// A dry run planned the check without running it.
    Plan,
}
impl State {
    fn word(self) -> &'static str {
        match self { State::Ok => "ok", State::Fail => "FAIL", State::Skip => "skip", State::Plan => "plan" }
    }
    fn token(self) -> Token {
        match self { State::Ok => Token::Ok, State::Fail => Token::Error, State::Skip | State::Plan => Token::Dim }
    }
}

/// Width of the check-name column; longer names push their detail right.
const NAME: usize = 20;

/// One doctor run under construction.
pub struct Report {
    mode: &'static str,
    dry: bool,
    started: Instant,
    rows: Vec<Value>,
    counts: [usize; 4],
}

impl Report {
    /// An empty report for the `local`, `controller`, `legacy` or `none` setup.
    pub fn new(mode: &'static str, dry: bool) -> Self {
        Self { mode, dry, started: Instant::now(), rows: Vec::new(), counts: [0; 4] }
    }

    /// Record one check and, without `--json`, print its row now.
    pub fn add(&mut self, state: State, name: &str, detail: &str, next: Option<&str>) -> Result<(), String> {
        let detail = crate::text::terminal_text(detail.trim_end()).replace('\n', "; ");
        self.counts[state as usize] += 1;
        self.rows.push(row(state, name, &detail, next));
        if json() { return Ok(()); }
        let word = paint(state.token(), &format!("{:<4}", state.word()), Stream::Out);
        let line = format!("{word}  {name:<NAME$}  {detail}");
        output::line(format_args!("{}", line.trim_end()))?;
        if let Some(next) = next.filter(|_| state == State::Fail) {
            output::line(format_args!("      {} {}", paint(Token::Hint, "next:", Stream::Out), crate::text::terminal_text(next)))?;
        }
        Ok(())
    }
    /// A passed check.
    pub fn ok(&mut self, name: &str, detail: &str) -> Result<(), String> { self.add(State::Ok, name, detail, None) }
    /// A failed check and the command that fixes it. When the error already
    /// names its own next step, that step wins over the generic `next`.
    pub fn fail(&mut self, name: &str, error: &str, next: &str) -> Result<(), String> {
        output::operation_failed();
        let (fact, hint) = crate::style::split(error);
        self.add(State::Fail, name, &fact, Some(hint.as_deref().unwrap_or(next)))
    }
    /// A check that does not apply or cannot run until an earlier one passes.
    pub fn skip(&mut self, name: &str, reason: &str) -> Result<(), String> { self.add(State::Skip, name, reason, None) }
    /// A check a dry run would run.
    pub fn plan(&mut self, name: &str, detail: &str) -> Result<(), String> { self.add(State::Plan, name, detail, None) }
    /// Record `result` as ok (with its detail) or as a failure with `next`.
    pub fn check<T>(&mut self, name: &str, result: Result<T, String>, detail: impl FnOnce(&T) -> String, next: &str) -> Result<Option<T>, String> {
        match result {
            Ok(value) => { self.ok(name, &detail(&value))?; Ok(Some(value)) }
            Err(error) => { self.fail(name, &error, next)?; Ok(None) }
        }
    }

    fn result(&self) -> &'static str {
        if self.counts[State::Fail as usize] > 0 { "fail" } else if self.dry { "plan" } else { "pass" }
    }
    /// The `vade.doctor.v1` object.
    fn value(self) -> Value {
        let [ok, failed, skipped, planned] = self.counts;
        json::object(vec![
            ("schema", "vade.doctor.v1".into()),
            ("mode", self.mode.into()),
            ("dry_run", self.dry.into()),
            ("result", self.result().into()),
            ("ok", ok.into()),
            ("failed", failed.into()),
            ("skipped", skipped.into()),
            ("planned", planned.into()),
            ("checks", Value::Array(self.rows)),
            ("elapsed_ms", (self.started.elapsed().as_secs_f64() * 1000.0).into()),
        ])
    }

    /// Print the PASS or FAIL line (or the JSON object). Any failure returns an
    /// error: the next step in text mode, [`REPORTED`] in JSON mode.
    pub fn finish(self) -> Result<(), String> {
        let [ok, failed, skipped, planned] = self.counts;
        let elapsed = self.started.elapsed();
        let result = self.result();
        if json() {
            print_json(&self.value())?;
            return if failed > 0 { Err(REPORTED.into()) } else { Ok(()) };
        }
        let mut counts = format!("{ok} ok, {failed} failed, {skipped} skipped");
        if planned > 0 { counts += &format!(", {planned} planned"); }
        let time = crate::launch::elapsed(elapsed);
        match result {
            "fail" => {
                output::line(format_args!("{}  {counts} in {time}", paint(Token::Error, "FAIL", Stream::Out)))?;
                Err(format!("{failed} doctor check(s) failed. Run each next: command above, then vade doctor."))
            }
            "plan" => output::line(format_args!("{}  {counts}; dry run, nothing checked", paint(Token::Dim, "PLAN", Stream::Out))),
            _ => output::line(format_args!("{}  {counts} in {time}", paint(Token::Ok, "PASS", Stream::Out))),
        }
    }
}

/// Rows for the tools a controller needs on this computer, for a doctor that
/// has no config yet. Each runs `--version` (or `ssh -V`) and connects nowhere.
pub fn local_tools(report: &mut Report) -> Result<(), String> {
    use crate::runner::{Exec, Runner, SystemRunner};
    let mut runner = SystemRunner { dry: false };
    for (tool, flag) in [("ssh", "-V"), ("rsync", "--version"), ("curl", "--version")] {
        let found = runner.run(&Exec::new(&format!("local {tool}"), tool, vec![flag.into()])).is_ok_and(|output| output.success);
        if found { report.ok(tool, "found on this computer")?; }
        else { report.fail(tool, &format!("{tool} is not installed on this computer."), &format!("Install {tool}, then run vade doctor."))?; }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn captured(run: impl FnOnce() -> Result<(), String>) -> (Result<(), String>, String) {
        let capture = output::testing::Capture::default();
        let result = output::with_writer(capture.clone(), run);
        (result, String::from_utf8(capture.bytes.borrow().clone()).unwrap())
    }
    #[test]
    fn rows_keep_fixed_columns_and_the_last_line_says_pass_or_fail() {
        let (result, printed) = captured(|| {
            let mut report = Report::new("local", false);
            report.ok("config", "/home/a/.vade/config")?;
            report.skip("engine health", "stopped")?;
            report.finish()
        });
        result.unwrap();
        let mut lines = printed.lines();
        assert_eq!(lines.next(), Some("ok    config                /home/a/.vade/config"));
        assert_eq!(lines.next(), Some("skip  engine health         stopped"));
        assert!(lines.next().unwrap().starts_with("PASS  1 ok, 0 failed, 1 skipped in "), "{printed}");

        let (result, printed) = captured(|| {
            let mut report = Report::new("local", false);
            report.fail("model files", "missing\nunder ~/models", "Run vade launch.")?;
            report.finish()
        });
        assert_eq!(result.unwrap_err(), "1 doctor check(s) failed. Run each next: command above, then vade doctor.");
        assert!(printed.starts_with("FAIL  model files           missing under ~/models\n      next: Run vade launch.\nFAIL  0 ok, 1 failed, 0 skipped in "), "{printed}");
    }
    #[test]
    fn json_mode_prints_one_object_and_reports_failure_quietly() {
        set_json(Some(Answer { schema: "vade.doctor.v1", mode: "none", dry: true }));
        let (result, printed) = captured(|| {
            let mut report = Report::new("none", true);
            report.plan("ssh spark-a", "ssh spark-a true")?;
            report.fail("config", "missing", "vade launch")?;
            report.finish()
        });
        set_json(None);
        assert_eq!(result.unwrap_err(), REPORTED);
        assert_eq!(printed.lines().count(), 1);
        let value = json::parse(&printed).unwrap();
        assert_eq!(value["schema"].as_str(), Some("vade.doctor.v1"));
        assert_eq!((value["result"].as_str(), value["failed"].as_i64(), value["planned"].as_i64()), (Some("fail"), Some(1), Some(1)));
        assert_eq!(value["checks"][1]["next"].as_str(), Some("vade launch"));
        assert_eq!(value["checks"][0]["state"].as_str(), Some("plan"));
    }
    #[test]
    fn an_early_error_still_answers_in_the_commands_schema() {
        set_json(Some(Answer { schema: "vade.doctor.v1", mode: "legacy", dry: false }));
        let doctor = unanswered("Start with vade run qwen3-8b.").unwrap();
        assert_eq!((doctor["mode"].as_str(), doctor["result"].as_str(), doctor["failed"].as_i64()), (Some("legacy"), Some("fail"), Some(1)));
        assert_eq!(doctor["checks"][0]["name"].as_str(), Some("doctor"));
        set_json(Some(Answer { schema: "vade.detect.v1", mode: "local", dry: false }));
        let detect = unanswered("No checkpoint found under ~/models. Set VADE_MODELS to the checkpoint folder, then run vade launch.").unwrap();
        assert_eq!(json::packed(&detect), "{\"schema\":\"vade.detect.v1\",\"error\":\"No checkpoint found under ~/models.\",\"next\":\"Set VADE_MODELS to the checkpoint folder, then run vade launch.\"}");
        // Once the command printed its object, nothing more is owed.
        output::with_writer(output::testing::Capture::default(), || print_json(&detect)).unwrap();
        assert!(unanswered("late").is_none());
        set_json(None);
        assert!(unanswered("plain").is_none());
    }
}
