//! `vade status --dry-run` and `vade status --json` for a local launch home.
//! Dry runs read only the saved records and never probe the engine.
use super::*;
use crate::{
    evals::json::{self, Value},
    status::Snapshot,
    style::{self, Stream, Token},
};

/// `vade status --json` for the local home at `root`.
pub(crate) fn snapshot<R: Runner>(
    root: PathBuf,
    runner: R,
    dry: bool,
    report: &mut Snapshot,
) -> Result<(), String> {
    Launcher {
        home: root.clone(),
        root,
        runner,
        detector: Native,
    }
    .status_snapshot(report, dry)
}

impl<R: Runner, D: Detector> Launcher<R, D> {
    /// `vade status --dry-run`: the recorded phase or "stopped", from files only.
    pub(super) fn status_preview(&self) -> Result<(), String> {
        let record = self.record()?;
        let config = Config::load(&self.root)?;
        match (config, record) {
            (Some(config), Some(record)) => output::line(format_args!(
                "{} Vade is recorded as {} at {} (dry run; readiness not checked).\n  Local config: {}",
                style::paint(Token::Dim, "○", Stream::Out),
                required(&record, "phase")?,
                config.endpoint(),
                self.root.join("config").display()
            )),
            (None, Some(_)) => Err(
                "An owned engine has no saved config; preserve the state folder and run vade down."
                    .into(),
            ),
            (_, None) => output::text(format_args!("{}", stopped())),
        }
    }

    /// `vade status` for a ready engine: the endpoint, then one row per setting.
    pub(super) fn status_table(&self, config: &Config, record: &Fields, events: &[String]) -> Result<String, String> {
        let field = |name: &str| record.get(name).map(|value| text::terminal_text(value).into_owned());
        let drafting = match record.get("mtp").map(String::as_str) {
            Some("engine_default") | None => "engine default".to_owned(),
            Some("0") => "off (--mtp 0)".to_owned(),
            Some(depth) => format!("MTP {}", text::terminal_text(depth)),
        };
        let mut rows = vec![
            ("Model", field("endpoint_model").unwrap_or_default()),
            ("Context", format!("{} tokens, output {}", field("context_tokens").unwrap_or_default(), field("output_tokens").unwrap_or_default())),
            ("Drafting", drafting),
        ];
        if record.get("prefix_cache").is_some_and(|value| value == "0") { rows.push(("Cache", "retained prompt cache off".into())); }
        if let Some(artifacts) = install::load(&self.root)?.filter(|artifacts| artifacts.get("engine") == record.get("engine")) {
            let build = required(&artifacts, "engine_sha256")?;
            let generation: String = required(&artifacts, "generation")?.chars().take(12).collect();
            rows.push(("Build", format!("{} (generation {})", build.chars().take(12).collect::<String>(), text::terminal_text(&generation))));
        }
        rows.push(("Config", text::terminal_text(&self.root.join("config").display().to_string()).into_owned()));
        let events: String = events.iter().map(|line| format!("  {line}\n")).collect();
        Ok(format!("{} Vade is ready at {}.\n{}{events}\n{}", style::paint(Token::Ok, "●", Stream::Out), config.endpoint(), style::table(&rows, 2),
            crate::status::next_steps(&[("Chat", &text::command("chat")), ("Stop", &text::command("down"))])))
    }
}

/// `vade status` when nothing is running.
pub(super) fn stopped() -> String { crate::status::stopped(&text::command("up")) }

/// `vade status` when the recorded engine process is gone, as after a crash or a kill.
pub(super) fn stale() -> String {
    format!("{} Vade is stopped: the engine is not running (stale record found; the next start clears it).\n{}",
        style::paint(Token::Dim, "○", Stream::Out), crate::status::next_steps(&[("Start", &text::command("up"))]))
}

/// `vade status` while vade launch still loads the model in another terminal.
pub(super) fn starting() -> String {
    format!("{} Vade is starting: the engine is still loading the model. The vade launch or vade up that started it says when it is ready.\n{}",
        style::paint(Token::Dim, "◐", Stream::Out), crate::status::next_steps(&[("Load progress", &text::command("logs"))]))
}

impl<R: Runner, D: Detector> Launcher<R, D> {

    /// Fill a status report; dry runs record every live check as unchecked.
    pub(super) fn status_snapshot(
        &mut self,
        report: &mut Snapshot,
        dry: bool,
    ) -> Result<(), String> {
        // Read the owner record first: incomplete configuration must never turn
        // an existing engine into a successful 'stopped' observation.
        let record = self.record()?;
        if let Some(record) = &record {
            report.set("phase", required(record, "phase")?);
            if let Some(model) = record.get("endpoint_model") {
                report.set("model", model.clone());
            }
            report.set(
                "endpoint",
                format!("http://127.0.0.1:{}/v1", required(record, "port")?),
            );
        }
        let config = Config::load(&self.root)?;
        let Some(record) = record else {
            report.set("phase", "stopped");
            report.set("ready", false);
            return Ok(());
        };
        let config = config.ok_or(
            "An owned engine has no saved config; preserve the state folder and run vade down.",
        )?;
        let recorded = |name: &str| record.get(name).filter(|n| n.parse::<u64>().is_ok())
            .map(|n| Value::BigInt(n.clone())).unwrap_or(Value::Null);
        let mtp_source = match record.get("mtp").map(String::as_str) {
            Some("engine_default") => "engine_default",
            Some(value) if value.parse::<u32>().is_ok() => "explicit",
            _ => "unknown",
        };
        report.set("settings", json::object(vec![
            ("fp4_local", true.into()),
            ("mtp", recorded("mtp")),
            ("mtp_source", mtp_source.into()),
            ("context_limit", recorded("context_tokens")),
            ("output_limit", recorded("output_tokens")),
            ("requested_chats", Value::Null),
            ("admitted_chats", Value::Null),
            ("memory_bytes_per_rank", recorded("memory_bytes")),
            ("retained_cache_bytes", if record.get("prefix_cache").is_some_and(|v| v == "0") { 0usize.into() } else { Value::Null }),
        ]));
        if let Some(artifacts) = install::load(&self.root)? {
            // The active installation describes this engine only if its saved
            // path still matches; metadata validation does not rehash binaries.
            if artifacts.get("engine") == record.get("engine") {
                report.set("generation", required(&artifacts, "generation")?);
                report.set("build", required(&artifacts, "engine_sha256")?);
            }
        }
        let alive = if dry {
            report.unchecked("process_identity", "localhost");
            None
        } else {
            let alive = self
                .worker(&record, "inspect", Fields::new(), false)
                .and_then(|fields| match fields.get("alive").map(String::as_str) {
                    Some("true") => Ok(true),
                    Some("false") => Ok(false),
                    _ => Err("Local process identity was not reported. Run vade doctor.".into()),
                });
            report.observe("process_identity", "localhost", alive)
        };
        report.set(
            "ranks",
            Value::Array(vec![json::object(vec![
                ("name", "localhost".into()),
                ("attempted", true.into()),
                ("alive", alive.map(Value::Bool).unwrap_or(Value::Null)),
            ])]),
        );
        if dry || alive != Some(true) {
            report.unchecked("listener", "localhost");
            report.unchecked("health", "localhost");
            if !dry {
                report.readiness(false);
            }
            return Ok(());
        }
        let listener = self
            .worker(
                &record,
                "listener",
                Fields::from([("port".into(), config.port.to_string())]),
                false,
            )
            .map(|fields| fields.get("listener").is_some_and(|v| v == "owned"));
        if report.observe("listener", "localhost", listener) != Some(true) {
            report.unchecked("health", "localhost");
            report.readiness(false);
            return Ok(());
        }
        let health = self
            .health(&config)
            .and_then(|model| Ok(model.as_deref() == Some(required(&record, "endpoint_model")?)));
        let healthy = report.observe("health", "localhost", health) == Some(true);
        report.readiness(healthy && record["phase"] == "running");
        Ok(())
    }
}
