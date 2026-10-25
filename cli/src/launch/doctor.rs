//! Read-only checks for a local installation, including incomplete first runs.
use super::*;

impl<R: Runner, D: Detector> Launcher<R, D> {
    pub(super) fn doctor(&mut self) -> Result<(), String> {
        let mut failures = Vec::new();
        let config = check(&mut failures, "Config", Config::load(&self.root)
            .and_then(|value| value.ok_or_else(|| "No saved local defaults.".into())),
            "Run vade launch to create defaults; preserve an invalid config for inspection.")?;
        let record = check(&mut failures, "Ownership", self.record(),
            "Preserve the state folder; run vade down before repairing the installation.")?;
        let installed = check(&mut failures, "Installed engine, kernels and chat", install::load(&self.root)
            .and_then(|value| value.ok_or_else(|| "Local artifacts are missing or have changed.".into())),
            "Run vade down, then vade install from your Vade checkout or with VADE_BUNDLE set.")?;
        let model = check(&mut failures, "Model", model::discover_existing(&self.home,
            config.as_ref().map(|config| config.model.as_path())).and_then(|model| {
                if !model.fp4_local {
                    return Err("Local launch requires nvidia/Qwen3.8-Flash-Next-FP4-Local.".into());
                }
                if config.as_ref().is_some_and(|config| config.model != model.path) {
                    return Err("The saved checkpoint is unavailable; another checkpoint was selected by VADE_MODELS.".into());
                }
                Ok(model)
            }), "Set VADE_MODELS to the complete FP4-Local checkpoint, then run vade launch. When replacing a valid saved checkpoint, first run vade down with the existing VADE_HOME, then choose a fresh VADE_HOME.")?;
        let models = model.as_ref().map(|model| model.path.clone())
            .or_else(|| config.as_ref().map(|config| config.model.clone()))
            .unwrap_or_else(|| self.home.join("models"));
        let identity = check(&mut failures, "Machine and NVIDIA driver", self.detector.detect(&self.root, &models),
            "Run nvidia-smi on the Spark to inspect the driver, then retry vade doctor.")?;
        if let Some(identity) = &identity {
            output::text(format_args!("{}", identity.display(&models)))?;
            let gib = 1024 * 1024 * 1024;
            let install_need = if installed.is_some() { 0 } else { INSTALL_SPACE }
                + if identity.volumes_shared { gib } else { 0 };
            check(&mut failures, "Disk", if identity.install_free >= install_need && identity.models_free >= gib {
                Ok(())
            } else {
                Err(format!("Insufficient disk working room: install needs {:.1} GiB and model volume needs 1 GiB.",
                    install_need as f64 / gib as f64))
            }, "Free space on the reported volumes, then retry vade doctor.")?;
            match (&record, &model) {
                (Some(None), Some(model)) => {
                    let result = if let Some(config) = &config {
                        config.check_resources(model, identity)
                    } else {
                        identity.check_resources(model.bytes, 0)
                    };
                    check(&mut failures, "Memory and launch limits", result,
                        "Close other GPU applications or use smaller limits with vade up --context 32768 --output 8192.")?;
                }
                (Some(Some(_)), _) => output::line(format_args!(
                    "Memory: resident engine recorded; available memory is shown above, without reserving its allocation again."))?,
                _ => output::line(format_args!("Memory admission: unavailable until the model and ownership checks pass."))?,
            }
        }
        match (config.as_ref(), record.as_ref()) {
            (Some(config), Some(Some(record))) => {
                check(&mut failures, "Owned endpoint and health", self.ready(config, record),
                    "Run vade logs, then vade down and vade launch.")?;
            }
            (_, Some(None)) => output::line(format_args!("Engine: stopped. Start: {}", text::command("launch")))?,
            _ => output::line(format_args!("Engine health: unavailable until the config and ownership checks pass; run vade down before repair."))?,
        }
        if failures.is_empty() {
            output::line(format_args!("{}", style::paint(Token::Ok, "Local checks passed.", Stream::Out)))
        } else {
            Err(format!("{} local check(s) need attention. Follow the next commands above, then retry vade doctor.", failures.len()))
        }
    }
}

fn check<T>(failures: &mut Vec<String>, label: &str, result: Result<T, String>, next: &str) -> Result<Option<T>, String> {
    match result {
        Ok(value) => {
            output::line(format_args!("{label}: {}", style::paint(Token::Ok, "OK.", Stream::Out)))?;
            Ok(Some(value))
        }
        Err(error) => {
            output::operation_failed();
            // Colour repeats what the words say: a yellow label, a cyan Next.
            output::line(format_args!("{}: {}\n{} {next}", style::paint(Token::Warn, label, Stream::Out),
                text::terminal_text(&error), style::paint(Token::Hint, "Next:", Stream::Out)))?;
            failures.push(error);
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_diagnostic_exit_probe() {
        let Ok(marker) = std::env::var("VADE_TEST_LOCAL_DOCTOR_PIPE") else { return; };
        output::with_writer(output::testing::Capture {
            fail_on: Some((if marker == "during" { "No saved local defaults." } else { "Ownership: OK." }, std::io::ErrorKind::BrokenPipe)),
            ..Default::default()
        }, || {
            let result = (|| {
                let mut failures = Vec::new();
                check::<()>(&mut failures, "Config", Err("No saved local defaults.".into()), "Run vade launch to create defaults.")?;
                check(&mut failures, "Ownership", Ok(()), "Run vade down.")?;
                Ok::<_, String>(())
            })();
            crate::fail(&result.unwrap_err());
        });
    }

    #[test]
    fn failed_diagnostic_stays_failed_when_stdout_closes_during_or_after_result() {
        let _process_guard = crate::runner::process_test_lock();
        for marker in ["during", "after"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "launch::doctor::tests::failed_diagnostic_exit_probe", "--nocapture"])
                .env("VADE_TEST_LOCAL_DOCTOR_PIPE", marker).output().unwrap();
            assert_eq!(output.status.code(), Some(1), "{marker}: {}", String::from_utf8_lossy(&output.stderr));
            assert!(String::from_utf8_lossy(&output.stderr).contains("standard output"));
        }
    }
}
