//! Versioned status output. Observations never imply ownership or readiness.
use crate::{
    app::App,
    cli::Command,
    config::Config,
    evals::json::{self, Value},
    run,
    runner::{Exec, Output, Runner, SystemRunner},
    style::{paint, Stream, Token},
};
use std::time::{Duration, Instant};

/// The human state line when nothing is running, and the command that starts it.
pub(crate) fn stopped(start: &str, setup: Option<&str>) -> String {
    let setup = setup.map_or_else(String::new, |setup| format!("  Last setup: {setup}\n"));
    format!("{} Vade is stopped.\n{setup}{}", paint(Token::Dim, "○", Stream::Out), next_steps(&[("Start", start)]))
}

/// What to type next, on one indented line: `  Chat: vade chat   Stop: vade down`.
pub(crate) fn next_steps(steps: &[(&str, &str)]) -> String {
    let steps: Vec<String> = steps.iter().map(|(label, command)| format!("{label}: {}", paint(Token::Command, command, Stream::Out))).collect();
    format!("  {}\n", steps.join("   "))
}

/// One `vade.status.v1` JSON report under construction: fields, checks and
/// errors. Any failed check makes the command exit 1.
pub(crate) struct Snapshot {
    value: Value,
    checks: Vec<Value>,
    errors: Vec<Value>,
    failed: bool,
}
impl Snapshot {
    /// An empty report whose phase is "unknown" until a lifecycle owner sets it.
    pub(crate) fn new(mode: &str, dry: bool) -> Self {
        Self {
            value: json::object(vec![
                ("schema", "vade.status.v1".into()),
                ("mode", mode.into()),
                ("dry_run", dry.into()),
                ("phase", "unknown".into()),
                ("ready", Value::Null),
                ("model", Value::Null),
                ("endpoint", Value::Null),
                ("build", Value::Null),
                ("generation", Value::Null),
                ("settings", Value::Null),
                ("ranks", Value::Array(Vec::new())),
                ("tokens_per_second", Value::Null),
            ]),
            checks: Vec::new(),
            errors: Vec::new(),
            failed: false,
        }
    }
    /// Set one top-level field.
    pub(crate) fn set(&mut self, key: &str, value: impl Into<Value>) {
        self.value.insert(key, value.into());
    }
    /// Record an error; the report then fails.
    pub(crate) fn error(&mut self, error: String) {
        self.failed = true;
        self.errors.push(error.into());
    }
    /// Record a check's result: `Ok(false)` fails the report, and an error is also
    /// listed under `errors`. Returns the observed state.
    pub(crate) fn observe(
        &mut self,
        name: &str,
        target: &str,
        result: Result<bool, String>,
    ) -> Option<bool> {
        let (state, error) = match result {
            Ok(state) => {
                self.failed |= !state;
                (Some(state), Value::Null)
            }
            Err(error) => {
                self.error(error.clone());
                (None, error.into())
            }
        };
        self.checks.push(json::object(vec![
            ("name", name.into()),
            ("target", target.into()),
            ("ok", state.map(Value::Bool).unwrap_or(Value::Null)),
            ("error", error),
        ]));
        state
    }
    /// Record a check that was not run (dry runs, or after an earlier failure).
    pub(crate) fn unchecked(&mut self, name: &str, target: &str) {
        self.checks.push(json::object(vec![
            ("name", name.into()),
            ("target", target.into()),
            ("ok", Value::Null),
            ("error", Value::Null),
        ]));
    }
    /// Set `ready`; not ready fails the report with a next step if no error says why.
    pub(crate) fn readiness(&mut self, ready: bool) {
        self.set("ready", ready);
        if !ready {
            self.failed = true;
            if self.errors.is_empty() {
                self.errors.push("Vade needs attention. Run vade logs or vade doctor; use vade down before another start.".into());
            }
        }
    }
    /// The JSON line, with C1 and direction controls escaped, and the exit code.
    pub(crate) fn finish(mut self, elapsed: Duration) -> (String, i32) {
        self.set("checks", Value::Array(self.checks.clone()));
        self.set("errors", Value::Array(self.errors.clone()));
        self.set("elapsed_ms", elapsed.as_secs_f64() * 1000.0);
        let encoded = json::packed(&self.value);
        // JSON escapes C0 already. Escape C1/direction controls too, preserving
        // their decoded value for a consumer while keeping terminal viewing safe.
        let mut text = String::with_capacity(encoded.len());
        for ch in encoded.chars() {
            if ch.is_control()
                || matches!(ch, '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}' | '\u{2028}' | '\u{2029}')
            {
                use std::fmt::Write;
                let _ = write!(text, "\\u{:04x}", ch as u32);
            } else {
                text.push(ch);
            }
        }
        (text, i32::from(self.failed))
    }
}

/// One deadline for all observations, including a slow or unreachable rank.
struct Bounded<R> {
    inner: R,
    deadline: Instant,
}
impl<R: Runner> Runner for Bounded<R> {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        let remaining = self.deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("Status checks reached their 15-second deadline. Retry vade status or inspect vade logs.".into());
        }
        let mut bounded = exec.clone();
        bounded.timeout = bounded.timeout.min(remaining);
        self.inner.run(&bounded)
    }
    fn interactive(&mut self, _: &Exec) -> Result<bool, String> {
        Err("Status cannot start an interactive command.".into())
    }
    fn port_free(&mut self, _: u16) -> Result<(), String> {
        Err("Status cannot reserve a port.".into())
    }
    fn pause(&mut self, _: Duration) {}
}

/// `vade status --json`: select the lifecycle owner and report within a 15 s
/// deadline, always as one JSON object.
pub(crate) fn run(dry: bool) -> (String, i32) {
    let start = Instant::now();
    let mut report = Snapshot::new("unknown", dry);
    let result = (|| {
        let runner = Bounded {
            inner: SystemRunner { dry: false },
            deadline: start + Duration::from_secs(15),
        };
        if crate::launch::handles(&Command::StatusJson) {
            report.set("mode", "local");
            return crate::launch::snapshot(crate::launch::root()?, runner, dry, &mut report);
        }
        let root = run::home()?;
        if run::handles(&Command::StatusJson, &root) {
            report.set("mode", "local");
            run::Local { root, runner, dry }.status_snapshot(&mut report)
        } else {
            report.set("mode", "configured");
            let mut config = Config::load()?;
            config.resolve_generation()?;
            App {
                config,
                runner,
                dry,
            }
            .status_snapshot(&mut report)
        }
    })();
    if let Err(error) = result {
        report.error(error);
    }
    report.finish(start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fake(Vec<Duration>);
    impl Runner for Fake {
        fn run(&mut self, exec: &Exec) -> Result<Output, String> {
            self.0.push(exec.timeout);
            Ok(Output::default())
        }
        fn interactive(&mut self, _: &Exec) -> Result<bool, String> {
            panic!("status cannot be interactive")
        }
        fn port_free(&mut self, _: u16) -> Result<(), String> {
            panic!("status cannot reserve ports")
        }
        fn pause(&mut self, _: Duration) {
            panic!("status cannot pause")
        }
    }
    #[test]
    fn one_deadline_bounds_commands_and_refuses_dispatch_after_expiry() {
        let exec = Exec::new("read-only fixture", "fixture", Vec::new());
        let mut runner = Bounded {
            inner: Fake(Vec::new()),
            deadline: Instant::now() + Duration::from_millis(500),
        };
        runner.run(&exec).unwrap();
        assert!(runner.inner.0[0] <= Duration::from_millis(500));
        assert_eq!(exec.timeout, Duration::from_secs(60));
        runner.deadline = Instant::now();
        assert!(runner.run(&exec).unwrap_err().contains("deadline"));
        assert_eq!(
            runner.inner.0.len(),
            1,
            "expired status must not dispatch a command"
        );
    }
}
