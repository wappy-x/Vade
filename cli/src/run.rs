//! Local, single-Spark bootstrap. Release generations and models publish only after verification.
//!
//! - `release`: pinned source builds and immutable release generations.
//! - `model`: the pinned Qwen3-8B weights (import, reuse or download).
//! - `status`: `vade status --json` for this bootstrap.
use crate::catalog::{MODEL_FILES, MODEL_REVISION};
use crate::cli::{Command, RunOptions};
use crate::runner::{Exec, Output, Runner, quote};
use crate::text::{self, Fields};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod model;
mod release;
mod status;

const ENDPOINT_MODEL: &str = "qwen3-8b";

/// The legacy bootstrap's home: `VADE_HOME`, else `$XDG_DATA_HOME/vade`, else
/// `~/.local/share/vade`.
pub fn home() -> Result<PathBuf, String> {
    let root = if let Some(root) = std::env::var_os("VADE_HOME") {
        PathBuf::from(root)
    } else if let Some(root) = std::env::var_os("XDG_DATA_HOME") {
        PathBuf::from(root).join("vade")
    } else {
        PathBuf::from(std::env::var_os("HOME").ok_or("Set HOME or VADE_HOME.")?)
            .join(".local/share/vade")
    };
    if !root.is_absolute()
        || root
            .components()
            .any(|p| matches!(p, std::path::Component::ParentDir))
    {
        return Err("VADE_HOME must be an absolute dedicated private directory. Set it to a private absolute path and retry.".into());
    }
    text::reject_symlinks(&root)?;
    Ok(root)
}

/// Whether the legacy bootstrap runs `command`: always for `vade run`, and for
/// the read and stop commands when its home has a setup (or, on Linux, when no
/// controller config exists).
pub fn handles(command: &Command, root: &Path) -> bool {
    matches!(command, Command::Run(_))
        || (std::env::var_os("VADE_CONFIG").is_none()
            && matches!(
                command,
                Command::Menu
                    | Command::Status
                    | Command::StatusJson
                    | Command::Down
                    | Command::Doctor
                    | Command::Logs
                    | Command::Chat
            )
            && (root.join("setup.conf").exists()
                || root.join("run.conf").exists()
                || (cfg!(target_os = "linux")
                    && !crate::config::config_path().is_ok_and(|p| p.is_file()))))
}

/// Reject an alternate local start before auth, locking or process work.
pub(crate) fn validate_start(command: &Command) -> Result<(), String> {
    if matches!(command, Command::Run(_))
        && (std::env::var_os("VADE_CONFIG").is_some() || crate::launch::owner_configured())
    {
        return Err("vade run qwen3-8b needs its own local installation; unset VADE_CONFIG and choose a fresh VADE_HOME, then keep that selection for chat, status, logs and down.".into());
    }
    if matches!(command, Command::Run(_)) && !crate::launch::legacy_configured() {
        crate::launch::check_default_controller_record()?;
    }
    Ok(())
}

/// The legacy `vade run qwen3-8b` bootstrap rooted at [`home`].
pub struct Local<R> {
    pub root: PathBuf,
    pub runner: R,
    pub dry: bool,
}

impl<R: Runner> Local<R> {
    /// Run one command for this bootstrap.
    pub fn execute(&mut self, command: Command) -> Result<(), String> {
        if self.dry {
            crate::output::line(format_args!("Dry run: no commands, downloads or file changes."))?;
        }
        match command {
            Command::Run(options) => self.run(&options),
            Command::Down => self.down(),
            Command::Status | Command::Menu => self.status(),
            Command::Logs => {
                if let Some(record) = self.record()? {
                    let reply = self.worker(&record, "logs", Fields::new())?;
                    if let Some(log) = reply.get("log") {
                        crate::output::text(format_args!("{}", text::terminal_text(log)))?;
                    }
                } else if let Ok(log) = fs::read_to_string(self.root.join("last-server.log")) {
                    crate::output::text(format_args!("{}", text::terminal_text(&log)))?;
                } else {
                    crate::output::line(format_args!("No owned server log."))?;
                }
                Ok(())
            }
            Command::Doctor | Command::Chat => {
                let record = self.record()?.ok_or("Start with vade run qwen3-8b.")?;
                self.ready(&record)?;
                let generation = self.generation(&record["generation"])?;
                if matches!(command, Command::Doctor) {
                    self.doctor(&record, &generation)
                } else {
                    let mut args = vec![
                        "direct".into(),
                        "--endpoint".into(),
                        endpoint(&record)?,
                        "--model".into(),
                        ENDPOINT_MODEL.into(),
                    ];
                    args.extend(crate::launch::chat_state_arguments()?);
                    let mut exec = Exec::new("local text chat", &generation["harness"], args);
                    clear_auth(&mut exec);
                    if !self.runner.interactive(&exec)? {
                        return Err("Chat exited with an error. Run vade doctor for the next step.".into());
                    }
                    Ok(())
                }
            }
            _ => Err("Use vade run qwen3-8b, chat, status, doctor, logs or down.".into()),
        }
    }

    fn run(&mut self, options: &RunOptions) -> Result<(), String> {
        if !self.dry {
            self.runner.recover_jobs(&self.root)?;
        }
        let prior = self.record()?;
        let owned = match &prior {
            Some(record) => {
                self.worker(record, "inspect", Fields::new())?
                    .get("alive")
                    .map(String::as_str)
                    == Some("true")
            }
            None => false,
        };
        let stored_setup = read_fields(&self.root.join("setup.conf"))?.unwrap_or_default();
        let stored_release = stored_setup.get("release").is_some_and(|s| !s.is_empty());
        let stored_weights = stored_setup
            .get("weights_source")
            .is_some_and(|s| !s.is_empty());
        let weights_source = options.weights_source.as_deref().or_else(|| {
            stored_setup
                .get("weights_source")
                .filter(|s| !s.is_empty())
                .map(Path::new)
        });
        // An owned server is reused only after identity, model and artifact verification.
        let detected = Instant::now();
        self.detect(
            owned,
            options.release.is_some() || stored_release || self.root.join("active.conf").exists(),
            weights_source.is_some_and(canonical_model_name),
        )?;
        crate::output::line(format_args!("Detection: {:.3} s", detected.elapsed().as_secs_f64()))?;
        if prior.is_none() {
            self.runner.port_free(options.port)?;
        }
        if self.dry {
            crate::output::line(format_args!(
                "# Fetch engine and harness source (or verify --release); pin source revisions before building."
            ))?;
            crate::output::line(format_args!(
                "# cargo kernels; cargo build --release --offline -p spark-serve; build vade-harness."
            ))?;
            if options.weights_source.is_some() || stored_weights {
                crate::output::line(format_args!(
                    "# Verify supplied official Qwen/Qwen3-8B weights read-only; no weight downloads."
                ))?;
            } else {
                crate::output::line(format_args!(
                    "# Resume official Qwen/Qwen3-8B revision {MODEL_REVISION} using curl -C -."
                ))?;
            }
            crate::output::line(format_args!(
                "# Hash all {} payload files; publish verified acquisitions atomically.",
                MODEL_FILES.len()
            ))?;
            crate::output::line(format_args!(
                "# Journal exact worker ownership, start on 127.0.0.1:{}, stream first answer, run doctor.",
                options.port
            ))?;
            return Ok(());
        }
        text::private_dir(&self.root)?;
        let setup_path = self.root.join("setup.conf");
        let mut setup = read_fields(&setup_path)?.unwrap_or_default();
        if setup.is_empty() {
            setup.insert("version".into(), "1".into());
            setup.insert("model_revision".into(), MODEL_REVISION.into());
            for (key, path) in [
                ("engine_source", &options.engine_source),
                ("harness_source", &options.harness_source),
                ("weights_source", &options.weights_source),
                ("release", &options.release),
            ] {
                setup.insert(
                    key.into(),
                    path.as_ref()
                        .map(|p| absolute(p))
                        .transpose()?
                        .unwrap_or_default(),
                );
            }
            text::atomic_write(&setup_path, &text::encode(&setup))?;
        } else {
            if setup.get("version").map(String::as_str) != Some("1")
                || setup.get("model_revision").map(String::as_str) != Some(MODEL_REVISION)
            {
                return Err(
                    "Setup revision is incompatible. Keep its files and select a fresh VADE_HOME."
                        .into(),
                );
            }
            for (key, path) in [
                ("engine_source", &options.engine_source),
                ("harness_source", &options.harness_source),
                ("weights_source", &options.weights_source),
                ("release", &options.release),
            ] {
                if let Some(path) = path {
                    if setup.get(key) != Some(&absolute(path)?) {
                        return Err("Setup inputs changed. Reuse the original inputs or choose a fresh VADE_HOME.".into());
                    }
                }
            }
        }
        let acquired = Instant::now();
        let generation = if let Some(record) = &prior {
            self.generation(&record["generation"])?
        } else if let Some(active) = read_fields(&self.root.join("active.conf"))? {
            self.generation(required(&active, "generation")?)?
        } else {
            self.acquire_engine(&mut setup)?
        };
        crate::output::line(format_args!(
            "Engine acquisition/build and verification: {:.3} s",
            acquired.elapsed().as_secs_f64()
        ))?;
        let model = if let Some(weights) = setup.get("weights_source").filter(|s| !s.is_empty()) {
            self.supplied_model(Path::new(weights))?
        } else {
            self.acquire_model(true)?
        };
        let mut record = if let Some(record) = prior {
            if record["port"] != options.port.to_string() {
                return Err(
                    "The owned server uses another port. Use its original --port or vade down."
                        .into(),
                );
            }
            if required(&record, "model_path")? != model.to_string_lossy() {
                return Err("Owned model path differs from the verified pinned model. Keep the record and inspect it.".into());
            }
            record
        } else {
            let token = text::random_hex(32)?;
            let record = Fields::from([
                ("version".into(), "1".into()),
                ("token".into(), token),
                ("generation".into(), generation["generation"].clone()),
                ("phase".into(), "starting".into()),
                ("port".into(), options.port.to_string()),
                ("model_path".into(), model.to_string_lossy().into_owned()),
            ]);
            // Durable intent precedes even a lost worker reply.
            text::atomic_write(&self.root.join("run.conf"), &text::encode(&record))?;
            record
        };
        let load_start = Instant::now();
        let alive = self.worker(&record, "inspect", Fields::new())?;
        let started_here = alive.get("phase").map(String::as_str) == Some("absent");
        if started_here {
            self.start(&record, &generation, &model)?;
        } else if alive.get("alive").map(String::as_str) != Some("true") {
            return Err("Owned startup is incomplete or stopped. Run vade logs and vade down; downloaded files remain reusable.".into());
        }
        let result = (|| {
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            let ready = self.readiness(&record, deadline);
            if self.runner.interrupted() {
                if started_here {
                    crate::output::stderr_line(format_args!(
                        "Interrupted during readiness. Run vade status to inspect the recorded server before retrying. Stopping the server started by this run."
                    ));
                    self.stop_record(record.clone()).map_err(|error| format!(
                        "Interrupted during readiness; cleanup could not finish: {error} The ownership record remains; run vade down."))?;
                    return Err(
                        "Interrupted during readiness. Run vade status to inspect the recorded server before retrying. The server started by this run was stopped."
                            .into(),
                    );
                }
                return Err(
                    "Interrupted during readiness. Run vade status to inspect the recorded server before retrying. The previously running server remains recorded."
                        .into(),
                );
            }
            match ready {
                Ok(true) => break,
                Ok(false) => {}
                Err(error) => return Err(error),
            }
            if Instant::now() >= deadline {
                return Err("Readiness deadline reached. The record remains: run vade logs, then vade down.".into());
            }
            self.runner.pause(
                Duration::from_millis(500).min(deadline.saturating_duration_since(Instant::now())),
            );
        }
        record.insert("phase".into(), "running".into());
        text::atomic_write(&self.root.join("run.conf"), &text::encode(&record))?;
        crate::output::line(format_args!(
            "Model load/readiness: {:.3} s",
            load_start.elapsed().as_secs_f64()
        ))?;
        self.answer(&record)?;
        self.doctor(&record, &generation)?;
        crate::output::line(format_args!("{ENDPOINT_MODEL}: ready at {}", endpoint(&record)?))?;
        crate::output::line(format_args!("Chat: vade chat\nCheck: vade status   Diagnose: vade doctor   Stop: vade down"))?;
        crate::output::line(format_args!(
            "Harness: {} direct --endpoint {} --model {ENDPOINT_MODEL}",
            text::terminal_text(&quote(&generation["harness"])),
            endpoint(&record)?
        ))?;
        crate::output::line(format_args!(
            "Claude Code: ANTHROPIC_BASE_URL={} ANTHROPIC_AUTH_TOKEN=vade-local ANTHROPIC_MODEL={ENDPOINT_MODEL} claude",
            endpoint(&record)?
        ))?;
        Ok(())
        })();
        if result.is_err() && started_here && crate::output::failure().is_some() {
            if let Err(cleanup) = self.stop_record(record) {
                crate::output::cleanup_failed();
                return Err(format!("Output failed and owned cleanup could not finish: {cleanup} Keep the ownership record and retry vade down."));
            }
        }
        result
    }

    fn detect(
        &mut self,
        owned: bool,
        artifact: bool,
        supplied_weights: bool,
    ) -> Result<(), String> {
        let script = r#"set -u
printf 'os=%s\narch=%s\n' "$(uname -s)" "$(uname -m)"
missing=''
for tool in curl sha256sum nvidia-smi sh; do command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"; done
printf 'missing=%s\n' "$missing"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader,nounits) || exit 1
  printf 'gpu=%s\n' "$gpu"
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 1
  printf 'busy=%s\n' "$(test -z "$apps" && echo no || echo yes)"
fi
awk '/^MemAvailable:/ {printf "ram_bytes=%.0f\n", $2 * 1024}' /proc/meminfo 2>/dev/null
probe=$1
while [ ! -d "$probe" ]; do probe=$(dirname "$probe"); done
df -Pk "$probe" | awk 'NR==2 {printf "disk_bytes=%.0f\n", $4 * 1024}'
"#;
        let output = self.checked(Exec::new(
            "Spark prerequisites",
            "sh",
            vec![
                "-c".into(),
                script.into(),
                "vade-detect".into(),
                self.root.to_string_lossy().into_owned(),
            ],
        ))?;
        if self.dry {
            return Ok(());
        }
        let fields = text::parse(&output.stdout)?;
        let mut missing = Vec::new();
        if fields.get("os").map(String::as_str) != Some("Linux") {
            missing.push("Linux".into());
        }
        if fields.get("arch").map(String::as_str) != Some("aarch64") {
            missing.push("aarch64".into());
        }
        if let Some(tools) = fields.get("missing").filter(|s| !s.trim().is_empty()) {
            missing.push(format!("tools:{tools}"));
        }
        let gpu = fields.get("gpu").map(String::as_str).unwrap_or("");
        let parts = gpu.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() != 3
            || !parts[0].contains("GB10")
            || !parts[1].bytes().all(|b| b.is_ascii_digit() || b == b'.')
            || parts[1].is_empty()
        {
            missing.push("one NVIDIA GB10 with a working driver".into());
        }
        let ram = fields
            .get("ram_bytes")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let gpu_free = parts
            .get(2)
            .and_then(|s| s.parse::<u64>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024));
        // GB10 uses unified RAM; nvidia-smi may return N/A for memory.free.
        if parts.get(2).is_some_and(|s| {
            gpu_free.is_none() && !matches!(*s, "N/A" | "[N/A]" | "[Not Supported]")
        }) {
            missing.push("readable GPU free memory".into());
        }
        if !owned {
            if fields.get("busy").map(String::as_str) != Some("no") {
                missing.push("idle GPU (another compute job is active)".into());
            }
            if ram < 24 * 1024 * 1024 * 1024
                || gpu_free.is_some_and(|b| b < 24 * 1024 * 1024 * 1024)
            {
                missing.push("24 GiB available unified/GPU memory".into());
            }
        }
        let disk = fields
            .get("disk_bytes")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let model_present = self.root.join("models").join(MODEL_REVISION).is_dir();
        let disk_need = if model_present || supplied_weights {
            1024 * 1024 * 1024
        } else {
            25 * 1024 * 1024 * 1024
        };
        if disk < disk_need {
            missing.push(format!("{disk_need} free disk bytes"));
        }
        if !missing.is_empty() {
            return Err(format!(
                "Missing prerequisites: {}. Nothing downloaded.",
                missing.join(", ")
            ));
        }
        if !artifact {
            let tools = self.runner.run(&Exec::new("source build prerequisites", "sh", vec!["-c".into(), r#"PATH="$HOME/.cargo/bin:$PATH"; export PATH; missing=''; for tool in git cc tar rustup cargo; do command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"; done; printf 'missing=%s\n' "$missing"; rustup run 1.99.0 rustc --version >/dev/null 2>&1 && echo stable=ok; rustup run nightly-2026-10-01 rustc --version >/dev/null 2>&1 && echo nightly=ok; rustup component list --toolchain nightly-2026-10-01 --installed 2>/dev/null"#.into()]))?;
            let tools = tools.stdout;
            let first = tools.lines().next().unwrap_or("");
            if first != "missing=" {
                missing.push(format!("build tools ({first})"));
            }
            for (needle, description) in [
                ("stable=ok", "Rust 1.99.0"),
                ("nightly=ok", "nightly-2026-10-01"),
                ("rust-src", "nightly rust-src"),
                ("llvm-tools", "nightly llvm-tools"),
                ("llvm-bitcode-linker", "nightly llvm-bitcode-linker"),
            ] {
                if !tools.contains(needle) {
                    missing.push(description.into());
                }
            }
            let target = self.runner.run(&Exec::new(
                "NVPTX prerequisite",
                &crate::app::rust_tool("rustup"),
                vec![
                    "target".into(),
                    "list".into(),
                    "--toolchain".into(),
                    "nightly-2026-10-01".into(),
                    "--installed".into(),
                ],
            ))?;
            if !target.success || !target.stdout.lines().any(|s| s == "nvptx64-nvidia-cuda") {
                missing.push("nightly nvptx64-nvidia-cuda target".into());
            }
        }
        crate::output::line(format_args!(
            "Detected: {}; available RAM {ram} bytes; GPU free {} bytes; disk {disk} bytes",
            text::terminal_text(gpu), gpu_free.unwrap_or(ram)
        ))?;
        if !missing.is_empty() {
            return Err(format!(
                "Missing prerequisites: {}. Nothing downloaded. Install prerequisites through your usual user/admin process, then retry.",
                missing.join(", ")
            ));
        }
        let executable = std::env::current_exe().map_err(|_| "Cannot locate vade. Run the installed binary from an existing directory.")?;
        let mut platform = Exec::new(
            "safe process prerequisites",
            &executable.to_string_lossy(),
            vec!["__worker".into()],
        );
        platform.stdin = b"op=platform\n".to_vec();
        let output = self.checked(platform)?;
        if text::parse(&output.stdout)?
            .get("pidfd")
            .map(String::as_str)
            != Some("ok")
        {
            return Err("Safe Linux process handles are unavailable. Nothing downloaded. Check the supported Spark prerequisites in INSTALL.md.".into());
        }
        Ok(())
    }

    fn save_setup(&self, setup: &Fields) -> Result<(), String> {
        text::atomic_write(&self.root.join("setup.conf"), &text::encode(setup))
    }
    fn activate(&self, id: &str) -> Result<(), String> {
        text::atomic_write(
            &self.root.join("active.conf"),
            &text::encode(&Fields::from([("generation".into(), id.into())])),
        )
    }
    fn start(&mut self, record: &Fields, generation: &Fields, model: &Path) -> Result<(), String> {
        let mut request = Fields::from([
            ("engine".into(), generation["engine"].clone()),
            ("cwd".into(), self.root.to_string_lossy().into_owned()),
            ("ptx_path".into(), generation["ptx"].clone()),
            ("binary_sha256".into(), generation["engine_sha256"].clone()),
            ("ptx_sha256".into(), generation["ptx_sha256"].clone()),
            ("model_path".into(), model.to_string_lossy().into_owned()),
        ]);
        let mut args = vec![
            generation["ptx"].clone(),
            model.to_string_lossy().into_owned(),
            "--bind".into(),
            format!("127.0.0.1:{}", record["port"]),
            "--draft".into(),
            "plain".into(),
            "--context-tokens".into(),
            "32768".into(),
            "--output-tokens".into(),
            "8192".into(),
            "--default-thinking".into(),
            "off".into(),
        ];
        if let Some(path) = crate::auth::file() {
            args.extend(["--api-key-file".into(), path.to_string_lossy().into_owned()]);
        }
        for (i, value) in args.into_iter().enumerate() {
            request.insert(format!("arg.{i}"), value);
        }
        for (key, value) in [
            ("SPARK_PROFILE", "0"),
            ("FLASH_BF16_EXPERTS", "0"),
            ("TP_WATCH_STDIN", "0"),
            ("SPARK_KV_BF16", "0"),
        ] {
            request.insert(format!("env.{key}"), value.into());
        }
        let reply = self.worker(record, "start", request)?;
        if reply.contains_key("error") {
            return Err(reply["error"].clone());
        }
        Ok(())
    }
    fn worker(
        &mut self,
        record: &Fields,
        operation: &str,
        request: Fields,
    ) -> Result<Fields, String> {
        self.worker_until(
            record,
            operation,
            request,
            Instant::now() + Duration::from_secs(90),
        )
    }
    fn worker_until(
        &mut self,
        record: &Fields,
        operation: &str,
        request: Fields,
        deadline: Instant,
    ) -> Result<Fields, String> {
        self.worker_request(record, operation, request, deadline, false)
    }
    fn worker_request(
        &mut self,
        record: &Fields,
        operation: &str,
        mut request: Fields,
        deadline: Instant,
        cleanup: bool,
    ) -> Result<Fields, String> {
        request.insert("token".into(), required(record, "token")?.into());
        request.insert("op".into(), operation.into());
        request.insert(
            "state_dir".into(),
            self.root.join("worker").to_string_lossy().into_owned(),
        );
        let executable = std::env::current_exe().map_err(|_| "Cannot find vade helper. Run the installed binary from an existing directory.")?;
        let mut exec = Exec::new(
            &format!("local {operation}"),
            &executable.to_string_lossy(),
            vec!["__worker".into()],
        );
        exec.stdin = text::encode(&request).into_bytes();
        exec.timeout = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_secs(90));
        let output = if cleanup {
            self.runner.cleanup(&exec)?
        } else {
            self.runner.run(&exec)?
        };
        if self.dry {
            return Ok(Fields::new());
        }
        let fields = text::parse(&output.stdout)?;
        if !output.success || fields.contains_key("error") {
            return Err(fields.get("error").cloned().unwrap_or_else(|| {
                "Local worker did not reply. Keep ownership records and retry vade down.".into()
            }));
        }
        Ok(fields)
    }
    fn record(&self) -> Result<Option<Fields>, String> {
        let record = read_fields(&self.root.join("run.conf"))?;
        if let Some(record) = &record {
            if required(record, "version")? != "1"
                || !text::hex(required(record, "token")?, 64)
                || !text::hex(required(record, "generation")?, 24)
                || !matches!(
                    required(record, "phase")?,
                    "starting" | "running" | "stopping"
                )
            {
                return Err("Invalid local ownership record. Keep it for inspection.".into());
            }
            endpoint(record)?;
        }
        Ok(record)
    }
    fn ready(&mut self, record: &Fields) -> Result<(), String> {
        self.ready_until(record, Instant::now() + Duration::from_secs(15))
    }
    fn ready_until(&mut self, record: &Fields, deadline: Instant) -> Result<(), String> {
        if self.readiness(record, deadline)? {
            Ok(())
        } else {
            Err("Owned server health/model is not ready. Run vade logs and vade doctor before retrying.".into())
        }
    }
    fn readiness(&mut self, record: &Fields, deadline: Instant) -> Result<bool, String> {
        if self.runner.interrupted() {
            return Err("Interrupted during readiness. Run vade status to inspect the recorded server before retrying.".into());
        }
        let identity = self.worker_until(record, "inspect", Fields::new(), deadline)?;
        if !self.dry && identity.get("alive").map(String::as_str) != Some("true") {
            return Err("The recorded server is stopped or its identity is uncertain. Keep ownership records; run vade logs, then vade down.".into());
        }
        if let Err(error) = self.worker_until(
            record,
            "listener",
            Fields::from([("port".into(), record["port"].clone())]),
            deadline,
        ) {
            if error == "The recorded server does not own the loopback listener." {
                return Ok(false);
            }
            return Err(error);
        }
        let mut health_request = curl(
            "owned health",
            vec![format!("{}/health", endpoint(record)?)],
        );
        health_request.timeout = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_secs(5));
        let health = self.runner.run(&health_request)?;
        if !health.success {
            return Ok(false);
        }
        if !self.dry
            && (crate::json::string(&health.stdout, "status").as_deref() != Some("ready")
                || crate::json::string(&health.stdout, "model").as_deref() != Some(ENDPOINT_MODEL))
        {
            return Ok(false);
        }
        if let Some(bytes) = crate::json::number(&health.stdout, "device_bytes") {
            crate::output::line(format_args!("Health tracked GPU bytes: {bytes:.0} (driver/untracked storage excluded)"))?;
        } else {
            crate::output::line(format_args!("Health tracked GPU bytes: unknown (telemetry unavailable)"))?;
        }
        Ok(true)
    }
    fn answer(&mut self, record: &Fields) -> Result<(), String> {
        let mut request = curl(
            "first streaming answer",
            vec![
                "--no-buffer".into(),
                "--header".into(),
                "Content-Type: application/json".into(),
                "--data-binary".into(),
                "@-".into(),
                format!("{}/v1/chat/completions", endpoint(record)?),
            ],
        );
        // Preserve the authentication error body only for this streamed answer.
        for arg in &mut request.args {
            if arg == "--fail" { *arg = "--fail-with-body".into(); }
        }
        request.timeout = Duration::from_secs(120);
        request.stdin = br#"{"model":"qwen3-8b","messages":[{"role":"user","content":"Explain why ice floats in one sentence."}],"max_tokens":128,"temperature":0,"top_p":1,"top_k":0,"enable_thinking":false,"stream":true}"#.to_vec();
        let _auth_header = crate::auth::curl_headers(&mut request, self.dry)?;
        let timer = Instant::now();
        let mut first = None;
        let mut content = String::new();
        let mut finish = None;
        let mut done = false;
        let mut invalid = false;
        let mut unauthorized = false;
        let output = self.runner.stream(&request, &mut |line| {
            if crate::json::path_string(line, &["error", "code"]).as_deref() == Some("invalid_api_key") {
                unauthorized = true;
            }
            if let Some(event) = line.strip_prefix("data: ") {
                if event == "[DONE]" {
                    done = true;
                    return Ok(());
                }
                if crate::json::string(event, "model").as_deref() != Some(ENDPOINT_MODEL) {
                    invalid = true;
                    // A closed pipe must not hide the model failure already observed.
                    crate::output::operation_failed();
                }
                if let Some(value) =
                    crate::json::path_string(event, &["choices", "0", "delta", "content"])
                        .filter(|v| !v.is_empty())
                {
                    first.get_or_insert_with(|| timer.elapsed());
                    crate::output::text(format_args!("{}", text::terminal_text(&value)))?;
                    content.push_str(&value);
                }
                if let Some(value) =
                    crate::json::path_string(event, &["choices", "0", "finish_reason"])
                {
                    finish = Some(value);
                }
            }
            Ok(())
        })?;
        if unauthorized { return Err("HTTP 401 Unauthorized: authentication failed. Check --api-key-file or VADE_API_KEY_FILE against the running engine; no request was replayed.".into()); }
        if !output.success
            || invalid
            || content.is_empty()
            || !done
            || !finish
                .as_ref()
                .is_some_and(|s| matches!(s.as_str(), "stop" | "length"))
        {
            return Err("First answer was incomplete or had the wrong model. Run vade logs and vade doctor.".into());
        }
        crate::output::line(format_args!(""))?;
        crate::output::line(format_args!(
            "First content token: {:.3} s; complete answer: {:.3} s",
            first.unwrap().as_secs_f64(),
            timer.elapsed().as_secs_f64()
        ))?;
        Ok(())
    }
    fn doctor(&mut self, record: &Fields, generation: &Fields) -> Result<(), String> {
        let started = Instant::now();
        let mut exec = Exec::new(
            "local doctor",
            &generation["harness"],
            vec![
                "doctor".into(),
                endpoint(record)?,
                "--model".into(),
                ENDPOINT_MODEL.into(),
                "--max-seconds".into(),
                "120".into(),
            ],
        );
        exec.timeout = Duration::from_secs(150);
        clear_auth(&mut exec);
        let output = self.checked(exec)?;
        crate::output::text(format_args!("{}", text::terminal_text(&output.stdout)))?;
        crate::output::line(format_args!(
            "Doctor: passed in {:.3} s.",
            started.elapsed().as_secs_f64()
        ))?;
        Ok(())
    }
    fn status(&mut self) -> Result<(), String> {
        let Some(record) = self.record()? else {
            crate::output::line(format_args!("Vade: stopped. Start: vade run qwen3-8b"))?;
            return Ok(());
        };
        self.ready(&record)?;
        crate::output::line(format_args!(
            "{ENDPOINT_MODEL}: {} at {}; generation {}",
            record["phase"],
            endpoint(&record)?,
            record["generation"]
        ))?;
        Ok(())
    }
    fn down(&mut self) -> Result<(), String> {
        if !self.dry {
            self.runner.recover_jobs(&self.root)?;
        }
        let Some(record) = self.record()? else {
            crate::output::line(format_args!("Vade: stopped."))?;
            return Ok(());
        };
        self.stop_record(record)?;
        crate::output::line(format_args!("Vade: stopped. Verified model and engine files remain for the next run."))
    }
    fn stop_record(&mut self, mut record: Fields) -> Result<(), String> {
        self.cleanup_worker(&record, "inspect")?;
        record.insert("phase".into(), "stopping".into());
        if !self.dry {
            text::atomic_write(&self.root.join("run.conf"), &text::encode(&record))?;
        }
        self.cleanup_worker(&record, "stop")?;
        self.cleanup_worker(&record, "gpu")?;
        let reply = self.cleanup_worker(&record, "clean")?;
        if !self.dry {
            if let Some(log) = reply.get("log") {
                text::atomic_write(&self.root.join("last-server.log"), log)?;
            }
            fs::remove_file(self.root.join("run.conf"))
                .map_err(|_| "Cannot remove stopped ownership record.")?;
        }
        Ok(())
    }
    fn cleanup_worker(&mut self, record: &Fields, operation: &str) -> Result<Fields, String> {
        self.worker_request(
            record,
            operation,
            Fields::new(),
            Instant::now() + Duration::from_secs(90),
            true,
        )
    }
    fn hash(&mut self, path: &Path) -> Result<String, String> {
        text::reject_symlinks(path)?;
        if !self.dry && !fs::symlink_metadata(path).is_ok_and(|m| m.is_file()) {
            return Err(format!(
                "Artifact is missing or not a regular file: {}",
                path.display()
            ));
        }
        let output = self.checked(Exec::new(
            "payload SHA256",
            "sha256sum",
            vec![path.to_string_lossy().into_owned()],
        ))?;
        let hash = output
            .stdout
            .split_whitespace()
            .next()
            .ok_or("SHA256 command returned no digest. Check sha256sum and the retained command log before retrying.")?;
        if !text::hex(hash, 64) {
            return Err("SHA256 command returned an invalid digest. Check sha256sum and the retained command log before retrying.".into());
        }
        Ok(hash.into())
    }
    fn verify_hash(&mut self, path: &Path, expected: &str) -> Result<(), String> {
        if !text::hex(expected, 64) {
            return Err("Manifest digest must be a full SHA256. Restore the manifest from the verified release or rebuild it.".into());
        }
        if self.hash(path)? != expected {
            return Err(format!(
                "Payload digest mismatch: {}. No completion marker can override this failure.",
                path.display()
            ));
        }
        Ok(())
    }
    fn checked(&mut self, exec: Exec) -> Result<Output, String> {
        let mut exec = exec;
        if !self.dry && matches!(exec.label.as_str(), "engine build" | "harness build") {
            crate::build_job::prepare(&self.root, &mut exec)?;
        }
        if exec.label.starts_with("fetch ") {
            exec.timeout = Duration::from_secs(600);
        }
        let output = self.runner.run(&exec)?;
        if !self.dry && !output.success {
            return Err(format!(
                "{} failed. Its private command logs are retained; retry the same command to resume.",
                exec.label
            ));
        }
        Ok(output)
    }
}

fn curl(label: &str, args: Vec<String>) -> Exec {
    let mut all = vec![
        "-q".into(),
        "--fail".into(),
        "--location".into(),
        "--silent".into(),
        "--show-error".into(),
        "--proto".into(),
        "=https,http".into(),
        "--proto-redir".into(),
        "=https".into(),
        "--connect-timeout".into(),
        "15".into(),
        "--noproxy".into(),
        "127.0.0.1,localhost".into(),
    ];
    all.extend(args);
    Exec::new(label, "curl", all)
}
fn required<'a>(fields: &'a Fields, key: &str) -> Result<&'a str, String> {
    fields
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("Record is missing {key}. Keep it for inspection."))
}
fn read_fields(path: &Path) -> Result<Option<Fields>, String> {
    text::reject_symlinks(path)?;
    match fs::read_to_string(path) {
        Ok(data) => text::parse(&data).map(Some),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(text::file_error("Cannot read private state record", path, &error)),
    }
}
fn endpoint(record: &Fields) -> Result<String, String> {
    let port = required(record, "port")?
        .parse::<u16>()
        .ok()
        .filter(|p| *p >= 1024)
        .ok_or("Invalid owned endpoint port. Keep the ownership record for inspection; run vade doctor before changing it.")?;
    Ok(format!("http://127.0.0.1:{port}"))
}
fn canonical_model_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("Qwen3-8B"))
}
fn absolute(path: &Path) -> Result<String, String> {
    text::reject_symlinks(path)?;
    fs::canonicalize(path)
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(|error| text::file_error("Cannot resolve input path", path, &error))
}
fn copy(source: &Path, destination: &Path) -> Result<(), String> {
    text::reject_symlinks(source)?;
    text::reject_symlinks(destination)?;
    let metadata = fs::symlink_metadata(source).map_err(|error| text::file_error("Cannot inspect build/release artifact", source, &error))?;
    if !metadata.is_file() {
        return Err(format!("Build/release artifact {} is not a regular file. Restore the verified release or rebuild it.", text::quoted(&source.to_string_lossy())));
    }
    fs::copy(source, destination).map_err(|error| text::file_error("Cannot stage build artifact", destination, &error))?;
    fs::OpenOptions::new()
        .write(true)
        .open(destination)
        .and_then(|f| f.sync_all())
        .map_err(|error| text::file_error("Cannot sync build artifact", destination, &error))?;
    Ok(())
}
fn sync_dir(directory: &Path) -> Result<(), String> {
    fs::File::open(directory)
        .and_then(|f| f.sync_all())
        .map_err(|error| text::file_error("Cannot sync publication directory", directory, &error))
}
fn header_size(headers: &str) -> Option<u64> {
    let mut size = None;
    for line in headers.lines() {
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("x-linked-size") {
                return value.trim().parse().ok();
            }
            if name.eq_ignore_ascii_case("content-length") {
                size = value.trim().parse::<u64>().ok().filter(|n| *n > 0);
            }
        }
    }
    size
}
fn clean_build_environment(exec: &mut Exec) {
    for (name, _) in std::env::vars_os() {
        if ["FLASH_", "SPARK_", "TP_", "SEQPAR_"]
            .iter()
            .any(|prefix| name.to_string_lossy().starts_with(prefix))
        {
            exec.unset.push(name.to_string_lossy().into_owned());
        }
    }
    exec.unset.push("KERNEL_BIN".into());
    exec.env.insert("RUSTUP_TOOLCHAIN".into(), "1.99.0".into());
    if let Ok(home) = std::env::var("HOME") {
        exec.env.insert(
            "PATH".into(),
            format!(
                "{home}/.cargo/bin:{}",
                std::env::var("PATH").unwrap_or_default()
            ),
        );
    }
    exec.unset.extend(
        [
            "CARGO_TARGET_DIR",
            "RUSTC_WRAPPER",
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_BUILD_TARGET",
            "RUSTC",
            "RUSTDOC",
        ]
        .map(str::to_owned),
    );
}
fn clear_auth(exec: &mut Exec) {
    crate::auth::forward(exec);
    exec.unset.extend(
        [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "OPENAI_BASE_URL",
            "ANTHROPIC_BASE_URL",
        ]
        .map(str::to_owned),
    );
}

#[cfg(test)]
#[rustfmt::skip]
mod tests;
