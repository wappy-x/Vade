//! The owner flow on one Spark. Configuration is generated once; process
//! ownership and cleanup use the same Linux worker as the remote launcher.
//!
//! - `config`: the saved local defaults; `model`: find and check an existing
//!   checkpoint; `install`: publish the engine, kernels and chat together.
//! - `doctor`, `status` and `chat`: the read-mostly commands.
//! - `build_id`: the build identity of a local source tree.
use crate::{
    cli::{Command, UpOptions},
    detect, output,
    runner::{Exec, Runner},
    text::{self, Fields},
};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

mod build_id;
mod config;
mod install;
mod model;
mod status;
mod doctor;
mod chat;
use config::Config;
pub(crate) use status::snapshot;
pub(crate) use chat::state_arguments as chat_state_arguments;

const INSTALL_SPACE: u64 = 8 * 1024 * 1024 * 1024;

/// The local Vade home: `VADE_HOME`, else `~/.vade`. It must be absolute and
/// private.
pub fn root() -> Result<PathBuf, String> {
    let path = std::env::var_os("VADE_HOME")
        .map(PathBuf::from)
        .unwrap_or(owner_home()?.join(".vade"));
    if !path.is_absolute()
        || path
            .components()
            .any(|p| matches!(p, std::path::Component::ParentDir))
    {
        return Err("Vade needs an absolute private install folder (VADE_HOME).".into());
    }
    text::reject_symlinks(&path)?;
    Ok(path)
}

fn owner_home() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| "Cannot find your home folder.".into())
}

/// Whether the local home has saved defaults or an owned engine.
pub(crate) fn owner_configured() -> bool {
    root().is_ok_and(|p| p.join("config").exists() || p.join("state/run.conf").exists())
}

/// Whether the legacy `vade run qwen3-8b` home has a setup or a running server.
pub(crate) fn legacy_configured() -> bool {
    crate::run::home().is_ok_and(|p| p.join("setup.conf").exists() || p.join("run.conf").exists())
}

/// Creating local defaults changes later lifecycle selection. Preserve any
/// default-controller ownership evidence before allowing that transition.
pub(crate) fn check_default_controller_record() -> Result<(), String> {
    let path = crate::config::config_path()?;
    let file = match open_regular(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound
            && fs::symlink_metadata(&path).is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) => return Ok(()),
        Err(_) => return Err("Cannot inspect the default controller config; keep it for inspection before starting a new local installation.".into()),
    };
    let mut source = String::new();
    file.take(1_048_577).read_to_string(&mut source)
        .map_err(|_| "Cannot read the default controller config before starting a new local installation.")?;
    if source.len() > 1_048_576 {
        return Err("The default controller config is too large; keep it for inspection before starting a new local installation.".into());
    }
    let config = crate::config::Config::parse(&source, &path)?;
    let record = config.state_dir.join("session.conf");
    match open_regular(&record) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound
            && fs::symlink_metadata(&record).is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) => Ok(()),
        Ok(_) => Err("The default controller has an ownership record; run vade down with its config before creating local defaults, or set VADE_CONFIG to keep using that controller.".into()),
        Err(_) => Err("Cannot inspect the default controller ownership record; keep it for inspection and use its VADE_CONFIG before starting a new local installation.".into()),
    }
}

/// Whether the local owner flow runs `command`. An explicit `VADE_CONFIG` always
/// selects the controller instead.
pub fn handles(command: &Command) -> bool {
    if matches!(command, Command::Detect | Command::Launch(_)) {
        return true;
    }
    // One explicit selector applies to the entire controller lifecycle, even
    // when either local installation already has generated configuration.
    if std::env::var_os("VADE_CONFIG").is_some() {
        return false;
    }
    let configured = owner_configured();
    let legacy = legacy_configured();
    let fresh = !crate::config::config_path().is_ok_and(|p| p.is_file()) && !legacy;
    match command {
        Command::Install | Command::Update => configured || fresh || legacy,
        // Route conflicting local starts here so run() can refuse them before
        // a flag silently changes the installation selected by later commands.
        Command::Up(_) => configured || fresh || legacy,
        Command::Chat
        | Command::Down
        | Command::Status
        | Command::StatusJson
        | Command::Logs
        | Command::Doctor
        | Command::Menu => configured || (cfg!(target_os = "linux") && fresh),
        _ => false,
    }
}

/// Run a local lifecycle command after refusing conflicting selections. Without
/// `--dry-run` it needs a Linux aarch64 Spark.
pub fn run(command: Command, dry: bool) -> Result<(), String> {
    if matches!(command, Command::Launch(_)) && std::env::var_os("VADE_CONFIG").is_some() {
        return Err("VADE_CONFIG selects the controller installation; use vade up, or unset VADE_CONFIG before local vade launch and its lifecycle commands.".into());
    }
    if matches!(command, Command::Install | Command::Update | Command::Launch(_) | Command::Up(_))
        && !owner_configured() && legacy_configured()
    {
        return Err("This installation uses vade run qwen3-8b; reuse that command, choose a fresh VADE_HOME for vade launch, or set VADE_CONFIG for controller up and its lifecycle commands.".into());
    }
    if let Command::Up(options) | Command::Launch(options) = &command {
        Config::validate_options(options)?;
    }
    if matches!(command, Command::Install | Command::Launch(_) | Command::Up(_))
        && !owner_configured()
    {
        check_default_controller_record()?;
    }
    if !dry && !cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        return Err(
            "Run the local launcher on a Linux aarch64 Spark; --dry-run previews it here.".into(),
        );
    }
    let mut launcher = Launcher {
        root: root()?,
        home: owner_home()?,
        runner: crate::runner::SystemRunner { dry: false },
        detector: Native,
    };
    launcher.execute(command, dry)
}

trait Detector {
    fn detect(&mut self, install: &Path, models: &Path) -> Result<detect::Identity, String>;
}
struct Native;
impl Detector for Native {
    fn detect(&mut self, install: &Path, models: &Path) -> Result<detect::Identity, String> {
        detect::detect(install, models)
    }
}

struct Launcher<R, D> {
    root: PathBuf,
    home: PathBuf,
    runner: R,
    detector: D,
}

impl<R: Runner, D: Detector> Launcher<R, D> {
    fn execute(&mut self, command: Command, dry: bool) -> Result<(), String> {
        if let Command::Up(options) | Command::Launch(options) = &command {
            Config::validate_options(options)?;
        }
        if dry && matches!(command, Command::Status | Command::Menu) {
            return self.status_preview();
        }
        if dry {
            output::line(format_args!(
                "Dry run: no files, processes or connections changed."
            ))?;
            let action = match command {
                Command::Down => "Stop the owned local engine",
                Command::Chat => "Attach chat to the owned local engine",
                Command::Logs => "Read the owned local engine log",
                Command::Install => "Detect this Spark and model; save defaults; install locally",
                Command::Up(_) => "Start or attach to the local FP4-Local engine; use saved defaults or detect a first setup",
                Command::Update => "Refresh local engine and chat from VADE_BUNDLE or the current source checkout; stop first with vade down",
                Command::Doctor => "Check this Spark, driver, memory, disk, model, installed artifacts and owned endpoint",
                Command::Detect => "Detect this Spark and existing model",
                _ => {
                    "Detect this Spark and model; save defaults; install locally; start on loopback; open chat"
                }
            };
            output::line(format_args!(
                "{action}. Local config: {}",
                self.root.join("config").display()
            ))?;
            if let Command::Up(options) | Command::Launch(options) = &command {
                output::line(format_args!("Local defaults: context {}, output {} (lowered only on engine refusal); explicit context: {:?}, output: {:?}.",
                    crate::cli::FP4_CONTEXT, crate::cli::fp4_output(crate::cli::FP4_CONTEXT),
                    options.supplied_options.contains("--context").then_some(options.context),
                    options.output_explicit.then_some(options.output)))?;
            }
            return Ok(());
        }
        let options = match &command {
            Command::Up(options) | Command::Launch(options) => options.clone(),
            _ => UpOptions::default(),
        };
        let began = Instant::now();
        match command {
            Command::Detect => {
                self.detection(Config::load(&self.root)?.as_ref(), INSTALL_SPACE)?;
                Ok(())
            }
            Command::Install | Command::Launch(_) | Command::Up(_) => {
                // Release the operation lock before interactive chat. Another terminal
                // can attach or stop the resident engine while this chat is open.
                let (config, artifacts, record) = {
                    let _lock = crate::state::Lock::acquire(&self.root)?;
                    self.recover_builds()?;
                    let configured = Config::load(&self.root)?;
                    let installed = install::load(&self.root)?;
                    let prior = self.record()?;
                    if prior.is_some() && (configured.is_none() || installed.is_none()) {
                        return Err("An owned engine has incomplete install records; run vade down before reinstalling.".into());
                    }
                    let mut config = if let Some(record) = &prior {
                        let config = configured.ok_or("Run vade down before repairing the local config.")?;
                        config.check_running_options(&options, record)?
                    } else {
                        let (model, identity) = self.detection(configured.as_ref(),
                            if installed.is_some() { 0 } else { INSTALL_SPACE })?;
                        let configured_phase = phase("Config", "choose and save local defaults")?;
                        let previous = configured.as_ref().map(Config::display);
                        let changed_model = configured.as_ref().is_some_and(|c| c.model != model.path);
                        let mut config = match configured {
                            Some(config) if !changed_model => config,
                            prior => {
                                let mut config = Config::new(&self.root, &model)?;
                                if let Some(prior) = prior { config.port = prior.port; }
                                config
                            }
                        };
                        config.apply(&options, &model, &identity)?;
                        if changed_model {
                            for name in ["model-manifest.sha256", "model-manifest.sha256.sizes"] {
                                let path = self.root.join("state").join(name);
                                text::reject_symlinks(&path)?;
                                match fs::remove_file(&path) {
                                    Ok(()) => (),
                                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
                                    Err(_) => return Err("Cannot clear the previous model's private receipt; keep the state folder and retry vade install.".into()),
                                }
                            }
                        }
                        if previous.as_deref() != Some(&config.display()) {
                            if previous.is_some() {
                                text::atomic_write(&self.root.join("config"), &config.display())?;
                            } else { config.save(&self.root)?; }
                            output::line(format_args!("Saved defaults in {}:\n{}",
                                self.root.join("config").display(), config.display()))?;
                        }
                        done("Config", configured_phase)?;
                        config
                    };
                    let start = phase(
                        "Install",
                        if installed.is_some() {
                            "reuse installed files"
                        } else {
                            "prepare local engine and chat"
                        },
                    )?;
                    let artifacts = match installed {
                        Some(files) => files,
                        None => install::ensure(&self.root, &mut self.runner)?,
                    };
                    done("Install", start)?;
                    let record = if command == Command::Install {
                        None
                    } else {
                        let start = phase(
                            "Up",
                            if prior.is_some() {
                                "attach to owned engine"
                            } else {
                                "load model on this Spark"
                            },
                        )?;
                        let retry = if matches!(command, Command::Up(_)) { "vade up" } else { "vade launch" };
                        let record = self.up_admitted(&mut config, &options, &artifacts, prior, retry)?;
                        done("Up", start)?;
                        Some(record)
                    };
                    (config, artifacts, record)
                };
                if matches!(command, Command::Launch(_)) {
                    output::line(format_args!(
                        "Launch ready in {:.3} s.",
                        began.elapsed().as_secs_f64()
                    ))?;
                    let record = record.as_ref().ok_or("Launch did not produce an engine record.")?;
                    if self.runner.terminal() {
                        self.chat(&config, &artifacts, record)?;
                    } else {
                        // Unattended (script, pipe, /dev/null): finish cleanly and leave the engine running.
                        output::line(format_args!("Standard input is not a terminal, so chat was not opened. Chat: {}", text::command("chat")))?;
                    }
                }
                Ok(())
            }
            Command::Chat => {
                let config = Config::load(&self.root)?
                    .ok_or("Run vade launch to install and start Vade.")?;
                let artifacts = install::load(&self.root)?.ok_or("Run vade install first.")?;
                let record = self.record()?.ok_or("Vade is stopped; run vade launch.")?;
                self.ready(&config, &record)?;
                self.chat(&config, &artifacts, &record)
            }
            Command::Down => {
                let _lock = crate::state::Lock::acquire(&self.root)?;
                self.recover_builds()?;
                if let Some(record) = self.record()? {
                    let start = phase("Down", "stop this launch's owned engine")?;
                    self.stop(&record)?;
                    done("Down", start)
                } else {
                    output::line(format_args!("Vade is already stopped."))
                }
            }
            Command::Logs => {
                let log = if let Some(record) = self.record()? {
                    self.worker(&record, "logs", Fields::new(), false)?
                        .get("log")
                        .cloned()
                        .unwrap_or_default()
                } else {
                    let file = match open_regular(&self.root.join("state/last-server.log")) {
                        Ok(file) => Some(file),
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                        Err(_) => return Err("Cannot read the saved engine log; it must be a regular readable file.".into()),
                    };
                    if let Some(file) = file {
                        let mut log = String::new();
                        file.take(1024 * 1024 + 1).read_to_string(&mut log)
                            .map_err(|_| "Cannot read the saved engine log.")?;
                        if log.len() > 1024 * 1024 {
                            return Err("Saved engine log exceeds 1 MiB.".into());
                        }
                        log
                    } else {
                        "No engine log yet.\n".into()
                    }
                };
                output::text(format_args!("{}", text::terminal_text(&log)))
            }
            Command::Doctor => self.doctor(),
            Command::Update => {
                let _lock = crate::state::Lock::acquire(&self.root)?;
                self.recover_builds()?;
                if self.record()?.is_some() {
                    return Err("The local engine is running; run vade down, then vade update and vade up.".into());
                }
                let config = Config::load(&self.root)?.ok_or("No local setup yet; run vade launch first.")?;
                let (model, identity) = self.detection(Some(&config), INSTALL_SPACE)?;
                config.check_resources(&model, &identity)?;
                install::refresh(&self.root, &mut self.runner)?;
                output::line(format_args!("Local engine and chat updated. Start: {}", text::command("up")))
            }
            Command::Status | Command::Menu => {
                match (Config::load(&self.root)?, self.record()?) {
                    (Some(config), Some(record)) => {
                        self.ready(&config, &record)?;
                        output::line(format_args!(
                            "Vade is ready at {}. Chat: {}",
                            config.endpoint(),
                            text::command("chat")
                        ))
                    }
                    (None, Some(_)) => Err("An owned engine has no saved config; preserve the state folder and run vade down.".into()),
                    (_, None) => output::line(format_args!("Vade is stopped. Start: {}", text::command("launch"))),
                }
            }
            _ => Err("Use vade install, vade launch or vade chat.".into()),
        }
    }

    fn detection(
        &mut self,
        config: Option<&Config>,
        install_space: u64,
    ) -> Result<(model::Model, detect::Identity), String> {
        let start = phase("Detect", "read this Spark and existing model")?;
        let model = model::discover(&self.home, config.map(|c| c.model.as_path()))?;
        let identity = self.detector.detect(&self.root, &model.path)?;
        if !model.fp4_local {
            return Err("Local launch needs nvidia/Qwen3.8-Flash-Next-FP4-Local. Set VADE_MODELS to that checkpoint and choose a fresh VADE_HOME before vade launch if defaults were already saved. For FP8, set VADE_CONFIG and use vade up --fp8-legacy.".into());
        }
        identity.check_resources(model.bytes, install_space)?;
        output::text(format_args!("{}", identity.display(&model.path)))?;
        done("Detect", start)?;
        Ok((model, identity))
    }

    fn record(&self) -> Result<Option<Fields>, String> {
        let record = read_fields(&self.root.join("state/run.conf"))?;
        if let Some(record) = &record {
            if required(record, "version")? != "1"
                || !text::hex(required(record, "token")?, 64)
                || !matches!(
                    required(record, "phase")?,
                    "starting" | "running" | "stopping"
                )
            {
                return Err(
                    "Invalid engine ownership record; preserve it and run vade doctor.".into(),
                );
            }
        }
        Ok(record)
    }

    fn recover_builds(&mut self) -> Result<(), String> {
        self.runner.recover_jobs(&self.root).inspect_err(|_| output::cleanup_failed())
    }

    /// Start at the requested or default context; the engine's admission is the
    /// authority. A context refusal lowers a default or saved context through
    /// 16384, 8192 and 4096, saved once a start succeeds; a refused explicit
    /// --context is an error.
    fn up_admitted(&mut self, config: &mut Config, options: &UpOptions, artifacts: &Fields,
        prior: Option<Fields>, retry: &str) -> Result<Fields, String> {
        let explicit = options.supplied_options.contains("--context");
        let (first, first_output) = (config.context, config.output);
        let mut refused = Vec::new();
        loop {
            let mut refusal = None;
            let error = match self.up(config, artifacts, prior.clone(), retry, &mut refusal) {
                Ok(record) => {
                    if !refused.is_empty() {
                        // Saved only after the lowered start succeeded; a total failure keeps the original.
                        text::atomic_write(&self.root.join("config"), &config.display())?;
                        let lowered = if config.output < first_output { format!("; output {first_output} lowered to {}", config.output) } else { String::new() };
                        output::line(format_args!("Context {} selected (the engine refused {} for memory){lowered}; saved; run vade launch --context {first} to try {} again.",
                            config.context, refused.iter().map(u32::to_string).collect::<Vec<_>>().join(" and "),
                            if first == crate::cli::FP4_CONTEXT { "the default" } else { "it" }))?;
                    }
                    return Ok(record);
                }
                Err(error) => error,
            };
            let Some(refusal) = refusal else { return Err(error) };
            let context = config.context;
            if explicit {
                return Err(format!("The engine refused --context {context} for memory:\n{}\nChoose a smaller --context, or omit it so the engine's admission selects one; run vade doctor to inspect memory.",
                    text::terminal_text(&refusal)));
            }
            if self.runner.interrupted() || config.step_down(options.output_explicit).is_none() { return Err(error); }
            refused.push(context);
        }
    }

    fn up(
        &mut self,
        config: &Config,
        artifacts: &Fields,
        prior: Option<Fields>,
        retry: &str,
        refusal: &mut Option<String>,
    ) -> Result<Fields, String> {
        if let Some(record) = prior {
            config.check_running_limits(&record)?;
            config.check_running_cache(&record)?;
            if required(&record, "model_path")? != config.model.to_string_lossy()
                || required(&record, "port")? != config.port.to_string()
                || required(&record, "engine")? != required(artifacts, "engine")?
            {
                return Err("The owned engine uses different saved files or launch limits; run vade down before starting it again.".into());
            }
            self.ready(config, &record)?;
            return Ok(record);
        }
        let prefix_cache = config.prefix_cache()?;
        self.runner.port_free(config.port).map_err(|_| {
            format!(
                "Port {} is in use; close the application using it and retry {retry}.",
                config.port
            )
        })?;
        let mut record = Fields::from([
            ("version".into(), "1".into()),
            ("token".into(), text::random_hex(32)?),
            ("phase".into(), "starting".into()),
            ("port".into(), config.port.to_string()),
            (
                "model_path".into(),
                config.model.to_string_lossy().into_owned(),
            ),
            ("engine".into(), required(artifacts, "engine")?.into()),
        ]);
        record.extend(config.launch_limits());
        if let Some(value) = &prefix_cache { record.insert("prefix_cache".into(), value.clone()); }
        self.worker(&record, "probe", Fields::new(), false)?;
        text::atomic_write(&self.root.join("state/run.conf"), &text::encode(&record))?;
        let result: Result<Fields, String> = (|| {
            let mut request = Fields::from([
                ("engine".into(), required(artifacts, "engine")?.into()),
                ("cwd".into(), self.root.to_string_lossy().into_owned()),
                ("ptx_path".into(), required(artifacts, "ptx")?.into()),
                (
                    "binary_sha256".into(),
                    required(artifacts, "engine_sha256")?.into(),
                ),
                (
                    "ptx_sha256".into(),
                    required(artifacts, "ptx_sha256")?.into(),
                ),
                (
                    "model_path".into(),
                    config.model.to_string_lossy().into_owned(),
                ),
                (
                    "env.FLASH_SERVE_OUTPUT_TOKENS".into(),
                    config.output.to_string(),
                ),
                (
                    "env.TP_RUN_ID".into(),
                    format!("vade-local-{}", text::random_hex(12)?),
                ),
                ("env.SEQPAR_RDMA".into(), "0".into()),
                ("env.TP_RDMA".into(), "0".into()),
                (
                    "env.XDG_CACHE_HOME".into(),
                    self.root.join("state/cache").to_string_lossy().into_owned(),
                ),
            ]);
            if let Some(value) = prefix_cache {
                request.insert("env.VADE_FLASHNEXT_PREFIX_CACHE".into(), value);
            }
            if let Some(manifest) = self.stage_manifest(&config.model)? {
                request.insert(
                    "env.FLASH_NVFP4_MANIFEST".into(),
                    manifest.to_string_lossy().into_owned(),
                );
            }
            config.engine_env(&mut request);
            let args = vec![
                "0".into(),
                "1".into(),
                "127.0.0.1:0".into(),
                "127.0.0.1:0".into(),
                required(artifacts, "ptx")?.into(),
                config.model.to_string_lossy().into_owned(),
                config.context.to_string(),
                format!("127.0.0.1:{}", config.port),
                "0".into(),
                "--fp4-local".into(),
            ];
            for (index, value) in args.into_iter().enumerate() {
                request.insert(format!("arg.{index}"), value);
            }
            self.worker(&record, "start", request, false)?;
            let deadline = Instant::now() + Duration::from_secs(600);
            let mut next_progress = Instant::now() + Duration::from_secs(15);
            loop {
                if self.runner.interrupted() {
                    return Err("Launch interrupted; the owned engine was stopped.".into());
                }
                let alive = self.worker(&record, "inspect", Fields::new(), false)?;
                if alive.get("alive").map(String::as_str) != Some("true") {
                    return Err("Engine stopped while loading. Run vade logs, then vade doctor before retrying vade up.".into());
                }
                if let Some(name) = self.health(config)? {
                    self.worker(
                        &record,
                        "listener",
                        Fields::from([("port".into(), config.port.to_string())]),
                        false,
                    )?;
                    record.insert("endpoint_model".into(), name);
                    record.insert("phase".into(), "running".into());
                    text::atomic_write(&self.root.join("state/run.conf"), &text::encode(&record))?;
                    return Ok(record.clone());
                }
                if Instant::now() >= deadline {
                    return Err(
                        "Model loading timed out after 600 s. Run vade logs, then vade doctor before retrying vade up."
                            .into(),
                    );
                }
                if Instant::now() >= next_progress {
                    output::line(format_args!(
                        "Up: still loading ({} s elapsed).",
                        600 - deadline.saturating_duration_since(Instant::now()).as_secs()
                    ))?;
                    next_progress = Instant::now() + Duration::from_secs(15);
                }
                self.runner.pause(Duration::from_millis(100));
            }
        })();
        match result {
            Ok(record) => Ok(record),
            Err(error) => {
                let log = self.worker(&record, "logs", Fields::new(), false).ok()
                    .and_then(|fields| fields.get("log").cloned()).unwrap_or_default();
                // A memory refusal may be retried at a smaller context by the caller.
                *refusal = config::memory_refusal(&log);
                if refusal.is_none() && output::failure().is_none() { output::operation_failed(); }
                let tail = log.lines().rev().take(20).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
                let diagnostic = format!("{error}\n{}\nFull log: vade logs", text::terminal_text(&tail));
                // Only our token is stopped; worker birth identity and pidfds protect other processes.
                match self.stop(&record) {
                    Ok(()) => Err(diagnostic),
                    Err(cleanup) => { *refusal = None; Err(format!("{diagnostic}\nCleanup incomplete: {cleanup}")) }
                }
            }
        }
    }

    fn ready(&mut self, config: &Config, record: &Fields) -> Result<(), String> {
        if required(record, "phase")? != "running" {
            return Err("Startup is incomplete; run vade down, then vade launch.".into());
        }
        self.worker(
            record,
            "listener",
            Fields::from([("port".into(), config.port.to_string())]),
            false,
        )?;
        if self.health(config)?.as_deref() != Some(required(record, "endpoint_model")?) {
            return Err(
                "The owned engine is not ready; run vade logs, then vade down and vade launch."
                    .into(),
            );
        }
        Ok(())
    }

    fn stage_manifest(&self, model: &Path) -> Result<Option<PathBuf>, String> {
        // The engine may persist sizes beside an NVFP4 manifest. Give it a
        // private metadata copy; all weight paths still resolve in the original
        // checkpoint, and the engine verifies their hashes in full as usual.
        // If none was supplied, NVFP4 generates its receipt here instead.
        // FP8 ignores this override and maintains checkpoint-local receipts.
        // Keep any receipt generated by an earlier launch intact.
        let destination = self.root.join("state/model-manifest.sha256");
        let file = match open_regular(&model.join("manifest.sha256")) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Some(destination)),
            Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => return Err("The model checksum manifest must be a regular file.".into()),
            Err(_) => return Err("Cannot read the model checksum manifest.".into()),
        };
        let mut manifest = String::new();
        file.take(1024 * 1024 + 1)
            .read_to_string(&mut manifest)
            .map_err(|_| "Cannot read the model checksum manifest.")?;
        if manifest.len() > 1024 * 1024 {
            return Err("Model checksum manifest exceeds 1 MiB.".into());
        }
        text::atomic_write(&destination, &manifest)?;
        Ok(Some(destination))
    }

    fn health(&mut self, config: &Config) -> Result<Option<String>, String> {
        let mut exec = Exec::new(
            "local health",
            "curl",
            vec![
                // This must be first: ignore default curl configuration so
                // write-out/output options cannot change the health JSON.
                "-q".into(),
                "--silent".into(),
                "--fail".into(),
                "--noproxy".into(),
                "*".into(),
                "--connect-timeout".into(),
                "1".into(),
                "--max-time".into(),
                "2".into(),
                format!("{}/health", config.endpoint()),
            ],
        );
        exec.timeout = Duration::from_secs(3);
        exec.transient_probe = true;
        clear_credentials(&mut exec);
        let reply = self.runner.run(&exec)?;
        if !reply.success
            || crate::json::string(&reply.stdout, "status").as_deref() != Some("ready")
        {
            return Ok(None);
        }
        Ok(crate::json::string(&reply.stdout, "model")
            .filter(|s| !s.is_empty() && !s.chars().any(char::is_control)))
    }

    fn worker(
        &mut self,
        record: &Fields,
        op: &str,
        mut request: Fields,
        cleanup: bool,
    ) -> Result<Fields, String> {
        request.insert("op".into(), op.into());
        request.insert("token".into(), required(record, "token")?.into());
        request.insert(
            "state_dir".into(),
            self.root
                .join("state/worker")
                .to_string_lossy()
                .into_owned(),
        );
        let exe = std::env::current_exe().map_err(|_| "Cannot locate the Vade process helper.")?;
        let mut exec = Exec::new(
            &format!("local {op}"),
            &exe.to_string_lossy(),
            vec!["__worker".into()],
        );
        exec.stdin = text::encode(&request).into_bytes();
        exec.timeout = Duration::from_secs(if matches!(op, "start" | "stop" | "clean") {
            120
        } else if op == "probe" {
            6
        } else {
            3
        });
        clear_credentials(&mut exec);
        let reply = if cleanup {
            self.runner.cleanup(&exec)?
        } else {
            self.runner.run(&exec)?
        };
        let fields = text::parse(&reply.stdout)?;
        if !reply.success || fields.contains_key("error") {
            return Err(fields.get("error").cloned().unwrap_or_else(|| {
                format!("Local {op} failed; preserve the state folder and run vade down.")
            }));
        }
        Ok(fields)
    }

    fn stop(&mut self, record: &Fields) -> Result<(), String> {
        self.stop_owned(record).inspect_err(|_| output::cleanup_failed())
    }

    fn stop_owned(&mut self, record: &Fields) -> Result<(), String> {
        self.worker(record, "stop", Fields::new(), true)?;
        // Worker cleanup returns its bounded, redacted log before removing the
        // journal; preserve that diagnostic without keeping stale ownership.
        let cleaned = self.worker(record, "clean", Fields::new(), true)?;
        if let Some(log) = cleaned.get("log") {
            text::atomic_write(&self.root.join("state/last-server.log"), log)?;
        }
        fs::remove_file(self.root.join("state/run.conf")).map_err(|_| {
            "Engine stopped, but its launch record could not be removed; retry vade down.".into()
        })
    }

    fn chat(&mut self, config: &Config, artifacts: &Fields, record: &Fields) -> Result<(), String> {
        let context = record.get("context_tokens").and_then(|value| value.parse::<u32>().ok())
            .filter(|value| (4096..=262144).contains(value))
            .ok_or("Invalid or missing recorded context; run vade down, then vade launch.")?;
        if context < chat::AGENT_MIN_CONTEXT {
            output::line(format_args!("Context {context} is below the agent minimum of {}; opening direct text chat.", chat::AGENT_MIN_CONTEXT))?;
        } else if cfg!(target_os = "linux") {
            output::line(format_args!("Automatic workspace checks are disabled on Linux; checks require the macOS sandbox."))?;
        }
        let start = phase("Chat", "open local chat; /exit leaves the engine running")?;
        let workspace = std::env::current_dir().map_err(|_| "Open an existing workspace and run vade chat.")?;
        let mut exec = Exec::new("local chat", required(artifacts, "harness")?,
            chat::arguments(&config.endpoint(), required(record, "endpoint_model")?, &workspace, context, chat::state_arguments()?));
        clear_credentials(&mut exec);
        let success = self.runner.interactive(&exec)?;
        if !success {
            output::operation_failed();
        }
        done("Chat", start)?;
        if success {
            Ok(())
        } else {
            Err("Chat exited with an error; the engine remains available through vade chat.".into())
        }
    }
}

fn clear_credentials(exec: &mut Exec) {
    // The existing worker stores inherited secret values for log redaction.
    // Give local children only ordinary runtime environment names so no cloud
    // credentials or model-tuning switches enter those private files at all.
    const KEEP: &[&str] = &[
        "HOME",
        "PATH",
        "LD_LIBRARY_PATH",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "COLORTERM",
        "COLUMNS",
        "LINES",
    ];
    exec.unset
        .extend(std::env::vars_os().filter_map(|(name, _)| {
            let name = name.to_string_lossy().into_owned();
            (!KEEP.contains(&name.as_str())).then_some(name)
        }));
}

struct Phase {
    name: String,
    started: Instant,
    finished: bool,
}
impl Drop for Phase {
    fn drop(&mut self) {
        if !self.finished {
            if output::failure().is_none() {
                output::operation_failed();
            }
            // Failure reporting must not overwrite the operation's exit
            // classification if stdout happens to close while unwinding.
            output::stderr_line(format_args!(
                "{}: stopped after {:.3} s.",
                self.name,
                self.started.elapsed().as_secs_f64()
            ));
        }
    }
}
fn phase(name: &str, action: &str) -> Result<Phase, String> {
    output::line(format_args!("{name}: {action}."))?;
    Ok(Phase {
        name: name.into(),
        started: Instant::now(),
        finished: false,
    })
}
fn done(name: &str, mut phase: Phase) -> Result<(), String> {
    phase.finished = true;
    output::line(format_args!(
        "{name}: done in {:.3} s.",
        phase.started.elapsed().as_secs_f64()
    ))
}
fn required<'a>(fields: &'a Fields, key: &str) -> Result<&'a str, String> {
    fields
        .get(key)
        .filter(|v| !v.is_empty())
        .map(String::as_str)
        .ok_or_else(|| format!("Saved launch data is missing {key}; preserve the state folder."))
}
fn read_fields(path: &Path) -> Result<Option<Fields>, String> {
    text::reject_symlinks(path)?;
    let file = match open_regular(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("Cannot read saved launch data; check folder permissions.".into()),
    };
    let mut data = String::new();
    file.take(65537)
        .read_to_string(&mut data)
        .map_err(|_| "Cannot read saved launch data.")?;
    if data.len() > 65536 {
        return Err("Saved launch data exceeds 64 KiB.".into());
    }
    text::parse(&data).map(Some)
}

/// Open before validating the descriptor so a substituted FIFO cannot block
/// the launcher. Nonblocking mode does not change regular-file reads.
fn open_regular(path: &Path) -> std::io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(crate::unix_flags::O_NONBLOCK | crate::unix_flags::O_CLOEXEC)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Expected a regular file."));
    }
    Ok(file)
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod health_tests;
