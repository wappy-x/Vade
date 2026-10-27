//! Read-only checks for a local installation, including incomplete first runs:
//! one row per check, then PASS or FAIL (see [`crate::checks`]).
use super::*;
use crate::checks::Report;

impl<R: Runner, D: Detector> Launcher<R, D> {
    pub(super) fn doctor(&mut self) -> Result<(), String> {
        let mut report = Report::new("local", false);
        let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let config = report.check("config", Config::load(&self.root)
            .and_then(|value| value.ok_or_else(|| "No saved local defaults.".into())),
            |_| self.root.join("config").display().to_string(),
            "Run vade launch to create defaults; preserve an invalid config for inspection.")?;
        report.check("VADE_HOME writable", writable(&self.root), |detail| detail.clone(),
            "Choose a VADE_HOME you own (export VADE_HOME=~/.vade), then retry vade doctor.")?;
        let record = report.check("ownership", self.record(),
            |record| if record.is_some() { "engine recorded".into() } else { "no engine recorded".into() },
            "Preserve the state folder; run vade down before repairing the installation.")?;
        let installed = report.check("installed binaries", install::load(&self.root)
            .and_then(|value| value.ok_or_else(|| "Local artifacts are missing or have changed.".into())),
            |_| "engine, kernels and chat match their saved file metadata".into(),
            "Run vade down, then vade install from your Vade checkout or with VADE_BUNDLE set.")?;
        let model = report.check("model files", model::discover_existing(&self.home,
            config.as_ref().map(|config| config.model.as_path())).and_then(|model| {
                if !model.fp4_local {
                    return Err("Local launch requires nvidia/Qwen3.8-Flash-Next-FP4-Local.".into());
                }
                if config.as_ref().is_some_and(|config| config.model != model.path) {
                    return Err("The saved checkpoint is unavailable; another checkpoint was selected by VADE_MODELS.".into());
                }
                Ok(model)
            }), |model| format!("{} ({} bytes; startup verifies the model manifest)", model.path.display(), model.bytes),
            "Set VADE_MODELS to the complete FP4-Local checkpoint, then run vade launch. When replacing a valid saved checkpoint, first run vade down with the existing VADE_HOME, then choose a fresh VADE_HOME.")?;
        let models = model.as_ref().map(|model| model.path.clone())
            .or_else(|| config.as_ref().map(|config| config.model.clone()))
            .unwrap_or_else(|| self.home.join("models"));
        let identity = report.check("nvidia-smi", self.detector.detect(&self.root, &models),
            |id| format!("{}, driver {}, CUDA {}", text::terminal_text(&id.gpu), text::terminal_text(&id.driver), text::terminal_text(&id.cuda)),
            "Run nvidia-smi on the Spark to inspect the driver, then retry vade doctor.")?;
        if let Some(identity) = &identity {
            let install_need = if installed.is_some() { 0 } else { INSTALL_SPACE }
                + if identity.volumes_shared { 1 << 30 } else { 0 };
            report.check("disk", if identity.install_free >= install_need && identity.models_free >= 1 << 30 {
                Ok(())
            } else {
                Err(format!("Insufficient disk working room: install needs {:.1} GiB and model volume needs 1 GiB.", gib(install_need)))
            }, |_| format!("{:.1} GiB free for install, {:.1} GiB for models", gib(identity.install_free), gib(identity.models_free)),
            "Free space on the reported volumes, then retry vade doctor.")?;
            let free = format!("{:.1} GiB available of {:.1} GiB (GPU and CPU share this memory)", gib(identity.memory_available), gib(identity.memory_total));
            match (&record, &model) {
                (Some(None), Some(model)) => {
                    let result = if let Some(config) = &config {
                        config.check_resources(model, identity)
                    } else {
                        identity.check_resources(model.bytes, 0)
                    };
                    report.check("free memory", result, |_| free.clone(),
                        "Close other GPU applications or use smaller limits with vade up --context 32768 --output 8192.")?;
                }
                (Some(Some(_)), _) => report.ok("free memory", &format!("{free}; the running engine holds its share"))?,
                _ => report.skip("free memory", "needs the model and ownership checks")?,
            }
        } else {
            report.skip("disk", "needs nvidia-smi and the machine checks")?;
            report.skip("free memory", "needs nvidia-smi and the machine checks")?;
        }
        match (config.as_ref(), record.as_ref()) {
            (Some(config), Some(Some(record))) => {
                report.skip(&format!("port {}", config.port), "in use by the owned engine")?;
                report.check("engine health", self.ready(config, record), |_| config.endpoint(),
                    "Run vade logs, then vade down and vade launch.")?;
            }
            (config, Some(None)) => {
                let port = config.map_or(18080, |config| config.port);
                report.check(&format!("port {port}"), self.runner.port_free(port), |_| "free on 127.0.0.1".into(),
                    &format!("Close the program listening on port {port}, then retry vade doctor."))?;
                report.skip("engine health", &format!("stopped; start: {}", text::command("launch")))?;
            }
            _ => report.skip("engine health", "needs the config and ownership checks")?,
        }
        report.finish()
    }
}

/// Whether `root` (or, before the first launch, the folder that will hold
/// it) accepts a new file. The probe file is removed at once.
fn writable(root: &Path) -> Result<String, String> {
    let (folder, note) = if root.is_dir() { (root, "") } else {
        (root.parent().filter(|parent| parent.is_dir()).ok_or_else(|| format!("{} and its parent folder do not exist.", text::quoted(&root.to_string_lossy())))?, " (created by vade launch)")
    };
    let probe = folder.join(format!(".vade-doctor-{}", std::process::id()));
    fs::OpenOptions::new().write(true).create_new(true).open(&probe)
        .map_err(|error| format!("Cannot create files in {}: {error}.", text::quoted(&folder.to_string_lossy())))?;
    let _ = fs::remove_file(&probe);
    Ok(format!("{}{note}", root.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_diagnostic_exit_probe() {
        let Ok(marker) = std::env::var("VADE_TEST_LOCAL_DOCTOR_PIPE") else { return; };
        output::with_writer(output::testing::Capture {
            fail_on: Some((if marker == "during" { "No saved local defaults." } else { "no engine recorded" }, std::io::ErrorKind::BrokenPipe)),
            ..Default::default()
        }, || {
            let result = (|| {
                let mut report = Report::new("local", false);
                report.check::<()>("config", Err("No saved local defaults.".into()), |_| String::new(), "Run vade launch to create defaults.")?;
                report.ok("ownership", "no engine recorded")?;
                report.finish()
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
