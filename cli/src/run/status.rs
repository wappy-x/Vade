use super::*;
use crate::{
    evals::json::{self, Value},
    status::Snapshot,
};

impl<R: Runner> Local<R> {
    pub(crate) fn status_snapshot(&mut self, report: &mut Snapshot) -> Result<(), String> {
        report.set("model", ENDPOINT_MODEL);
        let Some(record) = self.record()? else {
            report.set("phase", "stopped");
            report.set("ready", false);
            return Ok(());
        };
        report.set("phase", record["phase"].clone());
        report.set("generation", record["generation"].clone());
        report.set("endpoint", format!("{}/v1", endpoint(&record)?));
        let alive = if self.dry {
            report.unchecked("process_identity", "localhost");
            None
        } else {
            let identity = self
                .worker(&record, "inspect", Fields::new())
                .and_then(|fields| match fields.get("alive").map(String::as_str) {
                    Some("true") => Ok(true),
                    Some("false") => Ok(false),
                    _ => Err("Local process identity was not reported. Run vade doctor.".into()),
                });
            report.observe("process_identity", "localhost", identity)
        };
        report.set(
            "ranks",
            Value::Array(vec![json::object(vec![
                ("name", "localhost".into()),
                ("attempted", true.into()),
                ("alive", alive.map(Value::Bool).unwrap_or(Value::Null)),
            ])]),
        );
        if self.dry || alive != Some(true) {
            report.unchecked("listener", "localhost");
            report.unchecked("health", "localhost");
            if !self.dry {
                report.readiness(false);
            }
            return Ok(());
        }
        let listener = self
            .worker(
                &record,
                "listener",
                Fields::from([("port".into(), record["port"].clone())]),
            )
            .map(|fields| fields.get("listener").is_some_and(|v| v == "owned"));
        let listener = report.observe("listener", "localhost", listener);
        if listener != Some(true) {
            report.unchecked("health", "localhost");
            report.readiness(false);
            return Ok(());
        }
        let mut request = curl(
            "owned health",
            vec![format!("{}/health", endpoint(&record)?)],
        );
        request.timeout = Duration::from_secs(5);
        let health = self.runner.run(&request).map(|output| {
            output.success
                && crate::json::string(&output.stdout, "status").as_deref() == Some("ready")
                && crate::json::string(&output.stdout, "model").as_deref() == Some(ENDPOINT_MODEL)
        });
        let health = report.observe("health", "localhost", health);
        report.readiness(health == Some(true) && record["phase"] == "running");
        Ok(())
    }
}
