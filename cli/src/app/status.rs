//! `vade status --json` for a controller record: phase, settings and one
//! observation per rank, tunnel and health check.
use super::*;
use crate::{
    evals::json::{self, Value},
    status::Snapshot,
};

impl<R: Runner> App<R> {
    /// Fill a status report from the controller's session and live checks.
    pub(crate) fn status_snapshot(&mut self, report: &mut Snapshot) -> Result<(), String> {
        report.set("model", self.config.model.clone());
        let Some(session) = Session::load(&self.config.state_dir)? else {
            report.set("phase", "stopped");
            report.set("ready", false);
            return Ok(());
        };
        report.set("phase", session.phase.clone());
        report.set("model", session.model.clone());
        report.set("build", session.build.clone());
        report.set(
            "endpoint",
            format!("http://127.0.0.1:{}/v1", session.local_port),
        );
        report.set(
            "settings",
            json::object(vec![
                ("fp4_local", session.fp4_local.into()),
                (
                    "mtp",
                    if session.mtp_explicit {
                        (session.mtp as usize).into()
                    } else {
                        Value::Null
                    },
                ),
                (
                    "mtp_source",
                    if session.mtp_explicit {
                        "explicit".into()
                    } else {
                        "engine_default".into()
                    },
                ),
                ("context_limit", (session.context as usize).into()),
                ("output_limit", (session.output as usize).into()),
                (
                    "requested_chats",
                    session.chats.map(Value::from).unwrap_or(Value::Null),
                ),
                (
                    "admitted_chats",
                    session.admitted_chats.map(Value::from).unwrap_or(Value::Null),
                ),
                (
                    "memory_bytes_per_rank",
                    session
                        .memory_bytes
                        .map(|n| Value::BigInt(n.to_string()))
                        .unwrap_or(Value::Null),
                ),
                (
                    "retained_cache_bytes",
                    session
                        .cache_bytes
                        .map(|n| Value::BigInt(n.to_string()))
                        .unwrap_or(Value::Null),
                ),
            ]),
        );
        let mut ranks = Vec::new();
        let mut all = true;
        for rank in &session.ranks {
            let alive = if self.dry {
                report.unchecked("process_identity", &rank.machine.name);
                None
            } else if !rank.attempted {
                report.observe("process_identity", &rank.machine.name, Ok(false))
            } else {
                let observed = self
                    .worker(&session, rank, "inspect", Fields::new())
                    .and_then(|fields| match fields.get("alive").map(String::as_str) {
                        Some("true") => Ok(true),
                        Some("false") => Ok(false),
                        _ => Err(format!(
                            "{} did not report a process identity. Run vade doctor.",
                            rank.machine.name
                        )),
                    });
                report.observe("process_identity", &rank.machine.name, observed)
            };
            all &= alive == Some(true);
            ranks.push(json::object(vec![
                ("name", rank.machine.name.clone().into()),
                ("attempted", rank.attempted.into()),
                ("alive", alive.map(Value::Bool).unwrap_or(Value::Null)),
            ]));
        }
        report.set("ranks", Value::Array(ranks));
        if self.dry {
            report.unchecked("tunnel", "localhost");
            report.unchecked("health", "localhost");
            return Ok(());
        }
        let tunnel = self.connection_ok(&session);
        let tunnel = report.observe("tunnel", "localhost", tunnel);
        let mut path = None;
        let health = self.local_health_response(&session).map(|output| {
            if output.success && ready(&output.stdout, &session.model) {
                path = health::TwoSparkPath::from_health(&output.stdout);
                let speed = ["tok_s", "tokens_per_second", "tokens_per_sec"]
                    .iter()
                    .find_map(|key| crate::json::number(&output.stdout, key))
                    .filter(|value| *value >= 0.0);
                if let Some(speed) = speed {
                    report.set("tokens_per_second", speed);
                }
                true
            } else {
                false
            }
        });
        if session.two_sparks() {
            // The receipt's path and levels from the health reply; null when it names none.
            report.set("two_spark_path", path.map_or(Value::Null, |path| json::object(vec![
                ("path", path.path.into()),
                ("projection_level", path.projection_level.map_or(Value::Null, Value::from)),
                ("wire", path.wire.map_or(Value::Null, Value::from)),
                ("head", path.head.map_or(Value::Null, Value::from)),
            ])));
        }
        let health = report.observe("health", "localhost", health);
        report.readiness(
            all && tunnel == Some(true) && health == Some(true) && session.phase == "running",
        );
        Ok(())
    }
}
