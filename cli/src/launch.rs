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
    style::{self, Stream, Token},
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
#[cfg(test)]
pub(crate) use model::{test_checkpoint, test_config, test_mixed_config};
pub(crate) use model::serving_format;
pub(crate) use model::read_prompt_line;
mod status;
mod doctor;
mod chat;
use config::Config;
pub(crate) use status::snapshot;
pub(crate) use chat::chat_state_arguments;
pub(crate) use chat::setup as chat_setup;
pub(crate) use chat::{session_arguments as chat_session_arguments, sessions as sessions_exec, SESSIONS_FAILED};

const INSTALL_SPACE: u64 = 8 * 1024 * 1024 * 1024;
/// vade chat while another terminal's vade launch or vade up still loads the model.
const STILL_LOADING: &str = "The engine is still loading the model, so chat cannot open yet. Wait for vade launch to say it is ready, then run vade chat again. Run vade logs to follow the load.";
/// A start whose command ended early (it is gone) and left its record behind.
const UNFINISHED: &str = "The last engine start did not finish, and the engine is not ready. Run vade down to clear it, then vade launch.";

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

/// The checkpoint `vade launch` serves before any VADE_MODELS: the one saved in
/// `root`, while it still passes the launch checks.
pub(crate) fn saved_model(root: &Path) -> Option<PathBuf> {
    let model = Config::load(root).ok().flatten()?.model;
    serving_format(&model).is_ok().then_some(model)
}

/// The Vade checkout a source install builds from: VADE_SOURCE, else the
/// checkout that holds this vade program or the current folder.
pub(crate) fn checkout() -> Result<PathBuf, String> { install::source_path() }

/// Whether the local home has saved defaults or an owned engine.
pub(crate) fn owner_configured() -> bool {
    root().is_ok_and(|p| p.join("config").exists() || p.join("state/run.conf").exists())
}

/// Whether the local setup has saved defaults and an install of this
/// checkout or bundle, so vade install would only reuse it. Any doubt is `false`:
/// vade install then runs and names the problem.
pub(crate) fn owner_install_current() -> bool {
    let current = || -> Result<bool, String> {
        let root = root()?;
        if Config::load(&root)?.is_none() { return Ok(false); }
        let Some(files) = install::load(&root)? else { return Ok(false) };
        Ok(install::changed_source(&files, &mut crate::runner::SystemRunner { dry: false })?.is_none())
    };
    current().unwrap_or(false)
}

/// The installed harness of the local owner flow, if the install is complete.
pub(crate) fn installed_harness() -> Result<Option<PathBuf>, String> {
    Ok(install::load(&root()?)?.and_then(|artifacts| artifacts.get("harness").map(PathBuf::from)))
}

/// `vade config` rows for the saved local defaults: (label, text, JSON key,
/// JSON value), or `None` before the first launch.
pub(crate) fn saved_settings() -> Result<Option<(PathBuf, Vec<(&'static str, String, &'static str, crate::evals::json::Value)>)>, String> {
    use crate::evals::json::Value;
    let root = root()?;
    let Some(config) = Config::load(&root)? else { return Ok(None) };
    let drafting = match config.mtp() { None => "engine default".into(), Some(0) => "off (--mtp 0)".into(), Some(depth) => format!("depth {depth}") };
    Ok(Some((root.join("config"), vec![
        ("Model", config.model.display().to_string(), "model_path", config.model.to_string_lossy().as_ref().into()),
        ("Size", format!("{} bytes", config.model_bytes), "model_bytes", Value::Int(config.model_bytes as i64)),
        ("Context", format!("{} tokens", config.context), "context_tokens", Value::Int(config.context.into())),
        ("Output", format!("{} tokens", config.output), "output_tokens", Value::Int(config.output.into())),
        ("Port", format!("{} on 127.0.0.1", config.port), "port", Value::Int(config.port.into())),
        ("Drafting", drafting, "mtp", config.mtp().map_or(Value::Null, |depth| Value::Int(depth.into()))),
        ("Sparks", "1 (this Spark)".into(), "sparks", Value::Int(1)),
    ])))
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
    // The controller's own record and the record of every group guard alike,
    // also a group no longer in the config.
    let groups = config.group_record_names().map_err(|_| "Cannot inspect the default controller group ownership records; keep them for inspection before starting a new local installation.")?;
    let records = std::iter::once((None, config.state_dir.join("session.conf")))
        .chain(groups.iter().map(|name| (Some(name.as_str()), config.group_state_dir(name).join("session.conf"))));
    for (group, record) in records {
        let stop = group.map_or_else(|| "run vade down with its config".to_string(), |name| config.group_down_step(name));
        match open_regular(&record) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound
                && fs::symlink_metadata(&record).is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) => {}
            Ok(_) => return Err(format!("The default controller has an ownership record; {stop} before creating local defaults, or set VADE_CONFIG to keep using that controller.")),
            Err(_) => return Err(format!("Cannot inspect the default controller ownership record; keep it for inspection and {stop} before starting a new local installation.")),
        }
    }
    Ok(())
}

/// Whether the local owner flow runs `command`. An explicit `VADE_CONFIG` always
/// selects the controller instead.
pub fn handles(command: &Command) -> bool {
    if matches!(command, Command::Detect | Command::Launch(_)) {
        return true;
    }
    // One explicit selector applies to the entire controller lifecycle, even
    // when either local installation already has generated configuration.
    if crate::config::controller_selected() {
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
        Command::Chat(_)
        | Command::Sessions(_)
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
    if matches!(command, Command::Launch(_)) && crate::config::controller_selected() {
        return Err("VADE_CONFIG or VADE_GROUP selects the controller installation; use vade up, or unset VADE_CONFIG and VADE_GROUP before local vade launch and its lifecycle commands.".into());
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
    // A first install on another computer means a controller, which setup writes.
    if command == Command::Install && !cfg!(all(target_os = "linux", target_arch = "aarch64")) && !owner_configured() && !legacy_configured() {
        return Err("This computer is not a Spark, and it has no controller config yet. Run vade setup to drive Sparks from here, or run vade install on the Spark itself.".into());
    }
    if !dry && !cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        return Err(format!(
            "This command runs only on a DGX Spark (Linux aarch64), and this computer is {} {}. Run it on the Spark, or add --dry-run to preview it here. To drive Sparks from this computer, set VADE_CONFIG (see INSTALL.md).",
            std::env::consts::OS, std::env::consts::ARCH,
        ));
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
        if dry && crate::checks::json() && matches!(command, Command::Doctor | Command::Detect) {
            return self.dry_json(&command);
        }
        if dry {
            output::line(format_args!(
                "Dry run: no files, processes or connections changed."
            ))?;
            let action = match command {
                Command::Down => "Stop the local engine that vade started",
                Command::Chat(_) => "Open chat on the running local engine",
                Command::Sessions(_) => "List, export or delete saved chat sessions",
                Command::Logs => "Show the local engine log",
                Command::Install => "Detect this Spark and model; save defaults; install locally; prepare the private GPU startup cache",
                Command::Up(_) => "Start or attach to the local FP4-Local engine; use saved defaults or detect a first setup",
                Command::Update => "Refresh local engine and chat from VADE_BUNDLE or the current source checkout; stop first with vade down",
                Command::Doctor => "Check this Spark, driver, memory, disk, model, installed files and the running engine",
                Command::Detect => "Detect this Spark and existing model",
                _ => {
                    "Detect this Spark and model; save defaults; install locally; start on loopback; open chat"
                }
            };
            if command == Command::Install {
                output::text(format_args!("{}", self.install_plan()?))?;
            } else {
                output::line(format_args!("{action}."))?;
            }
            output::line(format_args!("Local config: {}", self.root.join("config").display()))?;
            if let Command::Up(options) | Command::Launch(options) = &command {
                // A start keeps the saved limits unless a flag overrides them.
                let saved = Config::load(&self.root)?;
                let planned = saved.as_ref().map(|config| config.planned(options));
                let context = if options.supplied_options.contains("--context") { format!("{} (explicit)", options.context) }
                    else if let Some(config) = &planned { format!("{} (saved; lowered only on engine refusal)", config.context) }
                    else { format!("{} (default; lowered only on engine refusal)", options.context) };
                let output = if options.output_explicit { format!("{} (explicit)", options.output) }
                    else if let (Some(config), Some(saved)) = (&planned, &saved) {
                        if config.output < saved.output { format!("{} (saved {}, capped at context)", config.output, saved.output) }
                        else { format!("{} (saved)", config.output) }
                    }
                    else { format!("{} (default)", options.output) };
                output::line(format_args!("Limits: context {context}, output {output}."))?;
            }
            return Ok(());
        }
        let options = match &command {
            Command::Up(options) | Command::Launch(options) => options.clone(),
            _ => UpOptions::default(),
        };
        let began = Instant::now();
        match command {
            Command::Detect if crate::checks::json() => self.detect_json(),
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
                    let prior = match self.record()? {
                        // A crash or a kill leaves the record of an engine that no longer runs.
                        Some(record) if command != Command::Install && self.engine_gone(&record) => {
                            self.stop(&record)?;
                            output::line(format_args!("Cleared the record of an engine that is no longer running (it stopped or was killed). Its log: {}", text::command("logs")))?;
                            None
                        }
                        prior => prior,
                    };
                    if prior.is_some() && (configured.is_none() || installed.is_none()) {
                        return Err("An owned engine has incomplete install records; run vade down before reinstalling.".into());
                    }
                    // vade install builds before the model is on disk, so
                    // scripts/install.sh can run first and vade pull later.
                    if command == Command::Install && prior.is_none() && configured.is_none()
                        && std::env::var_os("VADE_MODELS").is_none_or(|path| path.is_empty())
                        && let Err(reason) = model::find(&self.home)
                    {
                        return self.install_without_model(installed, &reason);
                    }
                    // A new or changed model choice gets its checkpoint check
                    // in Prepare before the first start.
                    let (mut config, chosen) = if let Some(record) = &prior {
                        let config = configured.ok_or("Run vade down before repairing the local config.")?;
                        (config.check_running_options(&options, record)?, false)
                    } else {
                        let (model, identity) = self.detection(configured.as_ref(),
                            if installed.is_some() { 0 } else { INSTALL_SPACE })?;
                        let configured_phase = phase("Config", "choose and save local defaults")?;
                        let previous = configured.as_ref().map(Config::display);
                        let changed_model = configured.as_ref().is_some_and(|c| c.model != model.path);
                        let chosen = configured.is_none() || changed_model;
                        let mut config = match configured {
                            Some(config) if !changed_model => config,
                            prior => {
                                let mut config = Config::new(&self.root, &model)?;
                                match prior {
                                    Some(prior) => config.port = prior.port,
                                    None if config.port != config::DEFAULT_PORT => output::line(format_args!(
                                        "Port {} on 127.0.0.1 is in use, so the API uses port {} instead. Clients and vade status use the saved port.",
                                        config::DEFAULT_PORT, config.port))?,
                                    None => (),
                                }
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
                        (config, chosen)
                    };
                    let (artifacts, fresh) = self.install_files(installed, command == Command::Install && prior.is_none())?;
                    let record = if command == Command::Install {
                        if prior.is_none() { self.prepare(Some(&config.model), &artifacts, "install")?; }
                        None
                    } else {
                        // A first launch, or the first one after vade install
                        // without a model, fills the same caches as install, so
                        // startup neither compiles nor hashes on the clock.
                        if prior.is_none() && (fresh || chosen) {
                            self.prepare(Some(&config.model), &artifacts, if matches!(command, Command::Up(_)) { "up" } else { "launch" })?;
                        }
                        let loading = format!("load {} on this Spark, context {} tokens",
                            text::terminal_text(&config.model.file_name().unwrap_or_default().to_string_lossy()), config.context);
                        let start = phase("Up", if prior.is_some() { "attach to the running engine" } else { &loading })?;
                        let retry = if matches!(command, Command::Up(_)) { "vade up" } else { "vade launch" };
                        let record = self.up_admitted(&mut config, &options, &artifacts, prior, retry)?;
                        done("Up", start)?;
                        Some(record)
                    };
                    (config, artifacts, record)
                };
                if command != Command::Install {
                    // One line with what a client needs: where, how much context, how many Sparks.
                    output::line(format_args!("{} ready in {}: {}, context {} tokens, output {}, 1 Spark.",
                        if matches!(command, Command::Launch(_)) { "Launch" } else { "Up" }, elapsed(began.elapsed()), config.endpoint(), config.context, config.output))?;
                }
                if matches!(command, Command::Up(_)) {
                    output::text(format_args!("{}", crate::status::next_steps(&[("Chat", &text::command("chat")), ("Stop", &text::command("down"))])))?;
                }
                if matches!(command, Command::Launch(_)) {
                    let record = record.as_ref().ok_or("Launch did not produce an engine record.")?;
                    if self.runner.terminal() {
                        self.chat(&config, &artifacts, record, &crate::cli::ChatOptions::default())?;
                    } else {
                        // Unattended (script, pipe, /dev/null): finish cleanly and leave the engine running.
                        output::line(format_args!("Standard input is not a terminal, so chat was not opened. Chat: {}", text::command("chat")))?;
                    }
                }
                Ok(())
            }
            Command::Chat(options) => {
                let config = Config::load(&self.root)?
                    .ok_or("Vade is not set up on this Spark yet. Run vade launch to install it, start it and open chat.")?;
                let artifacts = install::load(&self.root)?.ok_or("The engine and chat are not installed yet. Run vade launch to install them and open chat.")?;
                let record = self.record()?.ok_or("The engine is not running. Run vade launch to start it and open chat.")?;
                if self.loading(&record) { return Err(STILL_LOADING.into()); }
                self.ready(&config, &record)?;
                self.chat(&config, &artifacts, &record, &options)
            }
            Command::Sessions(action) => {
                let artifacts = install::load(&self.root)?.ok_or("The engine and chat are not installed yet. Run vade launch to install them.")?;
                let exec = chat::sessions(required(&artifacts, "harness")?, &action)?;
                if self.runner.interactive(&exec)? { Ok(()) } else { Err(chat::SESSIONS_FAILED.into()) }
            }
            Command::Down => {
                let _lock = crate::state::Lock::acquire(&self.root)?;
                self.recover_builds()?;
                if let Some(record) = self.record()? {
                    let start = phase("Down", "stop the engine that vade started")?;
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
                    (Some(_), Some(record)) if record["phase"] == "running" && self.engine_gone(&record) =>
                        output::text(format_args!("{}", status::stale())),
                    (Some(_), Some(record)) if self.loading(&record) =>
                        output::text(format_args!("{}", status::starting())),
                    (Some(config), Some(record)) => {
                        let health = self.ready(&config, &record)?;
                        let mut read = crate::app::events::read(config.endpoint(), "local events");
                        read.timeout = Duration::from_secs(6);
                        clear_credentials(&mut read);
                        let events = self.runner.run(&read)?;
                        let events = crate::app::events::status_lines(&events, &health, crate::app::events::now_ms());
                        output::text(format_args!("{}", self.status_table(&config, &record, &events)?))
                    }
                    (None, Some(_)) => Err("An owned engine has no saved config; preserve the state folder and run vade down.".into()),
                    (_, None) => output::text(format_args!("{}", status::stopped())),
                }
            }
            _ => Err("Use vade install, vade launch or vade chat.".into()),
        }
    }

    /// The Install phase: reuse the installed files, or build and stage them.
    /// With `check_source`, files from another checkout or bundle are replaced.
    /// Returns the files and whether they are new.
    fn install_files(&mut self, installed: Option<Fields>, check_source: bool) -> Result<(Fields, bool), String> {
        // vade install from a new checkout or bundle must not keep an engine from another one.
        let outdated = check_source
            && installed.as_ref().map_or(Ok(false), |files| install::outdated(files, &mut self.runner))?;
        let start = phase("Install", if installed.is_some() && !outdated { "reuse installed files" } else { "prepare local engine and chat" })?;
        let fresh = installed.is_none() || outdated;
        let artifacts = match installed {
            Some(_) if outdated => install::rebuild(&self.root, &mut self.runner)?,
            Some(files) => files,
            None => install::ensure(&self.root, &mut self.runner)?,
        };
        done("Install", start)?;
        Ok((artifacts, fresh))
    }

    /// `vade install` with no model on disk yet: check this Spark, build and
    /// stage the engine, kernels and chat, and compile the GPU code. It saves
    /// no defaults. The first vade launch or vade up finds the model, saves
    /// them and checks the checkpoint files.
    fn install_without_model(&mut self, installed: Option<Fields>, reason: &str) -> Result<(), String> {
        let start = phase("Detect", "read this Spark; the model is not on disk yet")?;
        let models = self.home.join("models");
        let models = if models.is_dir() { models } else { self.home.clone() };
        let identity = self.detector.detect(&self.root, &models)?;
        identity.check_resources(0, if installed.is_some() { 0 } else { INSTALL_SPACE })?;
        output::text(format_args!("{}", identity.display(&models)))?;
        output::line(format_args!("Model: {reason} The install goes on without it; vade launch needs the model."))?;
        done("Detect", start)?;
        let (artifacts, _) = self.install_files(installed, true)?;
        self.prepare(None, &artifacts, "install")?;
        output::line(format_args!("Installed without a model, so no local defaults were saved yet.\n{}\nThen start the engine and open chat: {}",
            model::MODEL_HELP, text::command("launch")))
    }

    fn detection(
        &mut self,
        config: Option<&Config>,
        install_space: u64,
    ) -> Result<(model::Model, detect::Identity), String> {
        let start = phase("Detect", "read this Spark and the model on disk")?;
        let model = model::discover(&self.home, config.map(|c| c.model.as_path()))?;
        let identity = self.identify(&model, install_space)?;
        output::text(format_args!("{}", identity.display(&model.path)))?;
        done("Detect", start)?;
        Ok((model, identity))
    }

    /// The checks `vade detect` makes with and without `--json`: this Spark,
    /// an FP4-Local checkpoint, and room for it.
    fn identify(&mut self, model: &model::Model, install_space: u64) -> Result<detect::Identity, String> {
        let identity = self.detector.detect(&self.root, &model.path)?;
        if !model.fp4_local {
            return Err("Local launch needs nvidia/Qwen3.8-Flash-Next-FP4-Local. Set VADE_MODELS to that checkpoint and choose a fresh VADE_HOME before vade launch if defaults were already saved. FP8 needs 2, 4, or 6 machines: set VADE_CONFIG and use vade up --fp8.".into());
        }
        identity.check_resources(model.bytes, install_space)?;
        Ok(identity)
    }

    /// What `vade install` will do on this Spark, for `--dry-run`.
    fn install_plan(&self) -> Result<String, String> {
        let shown = |path: &Path| text::terminal_text(&path.to_string_lossy()).into_owned();
        let model = match (saved_model(&self.root), std::env::var_os("VADE_MODELS").filter(|path| !path.is_empty())) {
            (Some(model), _) => format!("{} (saved)", shown(&model)),
            (None, Some(path)) => format!("{} (VADE_MODELS)", shown(Path::new(&path))),
            (None, None) => format!("the FP4-Local checkpoint under {}, found at install (optional: without it, install saves no defaults)", shown(&self.home.join("models"))),
        };
        let mut rows = vec![("Where", "this Spark".to_owned())];
        rows.extend(install::plan(&self.root)?);
        rows.extend([
            ("Model", model),
            ("State", format!("{}: config, stage and state", shown(&self.root))),
            ("Steps", "detect this Spark and model, save defaults when the model is there, stage the engine, kernels and chat, warm the GPU startup cache".into()),
            ("Next", text::command("install")),
        ]);
        Ok(format!("Plan for vade install:\n{}", style::table(&rows, 2)))
    }

    /// `vade detect --json`: the same checks and exit code as `vade detect`,
    /// then one `vade.detect.v1` object. It never prompts for a checkpoint;
    /// a failed check ends in the schema's `error` object (see `main`).
    fn detect_json(&mut self) -> Result<(), String> {
        use crate::evals::json::{self, Value};
        let saved = Config::load(&self.root)?;
        let model = model::discover_existing(&self.home, saved.as_ref().map(|c| c.model.as_path()))?;
        let id = self.identify(&model, INSTALL_SPACE)?;
        let value = json::object(vec![
            ("schema", "vade.detect.v1".into()),
            ("dry_run", false.into()),
            ("vendor", id.vendor.as_str().into()),
            ("product", id.product.as_str().into()),
            ("os", id.os.as_str().into()),
            ("architecture", id.architecture.as_str().into()),
            ("gpu", id.gpu.as_str().into()),
            ("driver", id.driver.as_str().into()),
            ("cuda", id.cuda.as_str().into()),
            ("memory_total_bytes", Value::Int(id.memory_total as i64)),
            ("memory_available_bytes", Value::Int(id.memory_available as i64)),
            ("install_free_bytes", Value::Int(id.install_free as i64)),
            ("models_free_bytes", Value::Int(id.models_free as i64)),
            ("volumes_shared", id.volumes_shared.into()),
            ("model", json::object(vec![
                ("path", model.path.to_string_lossy().as_ref().into()),
                ("bytes", Value::Int(model.bytes as i64)),
                ("fp4_local", model.fp4_local.into()),
            ])),
        ]);
        crate::checks::print_json(&value)
    }

    /// `--json --dry-run` for doctor and detect: the plan as their one object.
    fn dry_json(&mut self, command: &Command) -> Result<(), String> {
        use crate::evals::json;
        if *command == Command::Detect {
            return crate::checks::print_json(&json::object(vec![("schema", "vade.detect.v1".into()), ("dry_run", true.into()),
                ("plan", "Detect this Spark and existing model".into())]));
        }
        let mut report = crate::checks::Report::new("local", true);
        let state = self.root.join("state");
        for (name, detail) in [
            ("config", self.root.join("config").display().to_string()),
            ("VADE_HOME writable", self.root.display().to_string()),
            ("ownership", state.join("run.conf").display().to_string()),
            ("installed binaries", "engine, kernels and chat against their install hashes".into()),
            ("model files", "the saved checkpoint, or VADE_MODELS".into()),
            ("nvidia-smi", "GPU, driver and CUDA".into()),
            ("disk", "free space for install and models".into()),
            ("free memory", "available memory against the saved limits".into()),
            ("port", "the saved port on 127.0.0.1".into()),
            ("engine health", "the owned engine's /health".into()),
        ] { report.plan(name, &detail)?; }
        report.finish()
    }

    fn record(&self) -> Result<Option<Fields>, String> {
        let record = read_fields(&self.root.join("state/run.conf"))?;
        if let Some(record) = &record {
            if required(record, "version")? != "1"
                || !text::is_lower_hex(required(record, "token")?, 64)
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
                    return Err("Startup was interrupted, so Vade stopped the engine it was starting.".into());
                }
                let alive = self.worker(&record, "inspect", Fields::new(), false)?;
                if alive.get("alive").map(String::as_str) != Some("true") {
                    return Err(format!("The engine stopped while it loaded the model. Run vade logs to see why, then vade doctor, before you retry {retry}."));
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
                        format!("The model did not finish loading in 600 s. Run vade logs to see why, then vade doctor, before you retry {retry}."),
                    );
                }
                if Instant::now() >= next_progress {
                    progress("Up", Token::Dim, format_args!(
                        "still loading ({} s elapsed)",
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

    /// The ready engine's `/health` body. A record that is not running is a
    /// start that did not finish (a stop never writes a phase); callers
    /// without the lock ask [`Self::loading`] first.
    fn ready(&mut self, config: &Config, record: &Fields) -> Result<String, String> {
        if required(record, "phase")? != "running" {
            return Err(UNFINISHED.into());
        }
        if let Err(error) = self.worker(
            record,
            "listener",
            Fields::from([("port".into(), config.port.to_string())]),
            false,
        ) {
            return Err(if self.engine_gone(record) { format!("The engine is not running; it stopped or was killed. Run {} to clear its record and start it again.", text::command("up")) } else { error });
        }
        match self.health_reply(config)? {
            Some((model, body)) if model == required(record, "endpoint_model")? => Ok(body),
            _ => Err(
                "The engine is not answering. Run vade logs to see why, then vade down and vade launch."
                    .into(),
            ),
        }
    }

    /// Whether a vade launch or vade up still loads this engine: the record
    /// is `starting` and a command holds the lock. Only callers that do not
    /// hold the lock may ask. Two short windows: a vade down or vade install
    /// working on a `starting` record also holds the lock, and the probe takes
    /// a free lock for an instant, so a lifecycle command started at that
    /// same instant is refused as busy.
    fn loading(&self, record: &Fields) -> bool {
        record.get("phase").is_some_and(|phase| phase == "starting") && crate::state::Lock::held(&self.root)
    }

    /// Whether the worker confirms that the recorded engine process is gone,
    /// as after a crash or a kill. A worker error answers false, so the
    /// caller keeps its usual checks.
    fn engine_gone(&mut self, record: &Fields) -> bool {
        self.worker(record, "inspect", Fields::new(), false)
            .is_ok_and(|reply| reply.get("alive").is_some_and(|alive| alive == "false"))
    }

    fn stage_manifest(&self, model: &Path) -> Result<Option<PathBuf>, String> {
        // The engine may persist sizes beside an NVFP4 manifest. Give it a
        // private metadata copy; all weight paths still resolve in the original
        // checkpoint, and the engine verifies their hashes in full as usual.
        // If none was supplied, NVFP4 generates its receipt here instead.
        // FP8 ignores this override and maintains checkpoint-local receipts.
        // Keep any receipt generated by an earlier launch intact.
        let destination = self.root.join("state/model-manifest.sha256");
        let Some(manifest) = supplied_manifest(model)? else { return Ok(Some(destination)) };
        text::atomic_write(&destination, &manifest)?;
        Ok(Some(destination))
    }

    fn health(&mut self, config: &Config) -> Result<Option<String>, String> {
        Ok(self.health_reply(config)?.map(|(model, _)| model))
    }
    /// The served model name and the whole body, from a ready `/health` reply.
    fn health_reply(&mut self, config: &Config) -> Result<Option<(String, String)>, String> {
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
            .filter(|s| !s.is_empty() && !s.chars().any(char::is_control)).map(|model| (model, reply.stdout)))
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
        } else if op == "warm-ptx" {
            // Hashing a large checkpoint on a slow disk may outlast compilation.
            if request.contains_key("verify_model") { 900 } else { 330 }
        } else if op == "probe" {
            6
        } else {
            3
        });
        clear_credentials(&mut exec);
        if request.contains_key("verify_model") {
            // Receipts and their stamps go where startup reads them, never into the checkpoint.
            exec.env.insert("FLASH_NVFP4_MANIFEST".into(), self.root.join("state/model-manifest.sha256").to_string_lossy().into_owned());
            exec.env.insert("XDG_CACHE_HOME".into(), self.root.join("state/cache").to_string_lossy().into_owned());
            // Removals apply after overrides; an exported name must not erase them.
            exec.unset.retain(|name| !exec.env.contains_key(name));
        }
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
        if matches!(op, "probe" | "start" | "warm-ptx") {
            if let Some(note) = fields.get("gpu_note") {
                output::line(format_args!("{}", text::terminal_text(note)))?;
            }
        }
        Ok(fields)
    }

    /// Fill the private startup caches: compile the GPU code and, with a
    /// `model`, check every checkpoint file.
    fn prepare(&mut self, model: Option<&Path>, artifacts: &Fields, retry: &str) -> Result<(), String> {
        let started = phase("Prepare", if model.is_some() { "compile GPU code and check every checkpoint file (private caches only)" }
            else { "compile GPU code (private cache only)" })?;
        let record = Fields::from([("token".into(), text::random_hex(32)?)]);
        let mut request = Fields::from([
            ("engine".into(), required(artifacts, "engine")?.into()),
            ("ptx_path".into(), required(artifacts, "ptx")?.into()),
            ("binary_sha256".into(), required(artifacts, "engine_sha256")?.into()),
            ("ptx_sha256".into(), required(artifacts, "ptx_sha256")?.into()),
        ]);
        // The engine warm-up gets no MODEL: that mode writes receipts beside the
        // checkpoint. With a manifest, startup would check every file before
        // loading, so the helper does it now, during compilation, into the same
        // private receipt and stamp cache. Without one, startup serves at once
        // and the engine hashes in the background, so nothing is hashed here.
        let result = match model {
            None => self.worker(&record, "warm-ptx", request, false),
            Some(model) => self.stage_manifest(model).and_then(|manifest| {
                if supplied_manifest(model)?.is_none() {
                    output::line(format_args!("Prepare: no checksum list for this checkpoint ({}.sha256 beside it, or manifest.sha256 inside it). Vade cannot check its files against the published hashes. To check them, see Check the files in INSTALL.md.",
                        model.file_name().map_or_else(|| "<folder>".into(), |name| name.to_string_lossy())))?;
                }
                if manifest.is_some_and(|path| path.exists()) {
                    request.insert("verify_model".into(), model.to_string_lossy().into_owned());
                }
                self.worker(&record, "warm-ptx", request, false)
            }),
        };
        let fields = result.map_err(|error| format!(
            "{error}\nLocal cache preparation failed; installed files were kept. Run {}, fix the reported cause, then retry {}.",
            text::command("doctor"), text::command(retry)))?;
        if let Some(bytes) = fields.get("model_hashed_bytes") {
            let reused = fields.get("model_reused_files").map_or("0", String::as_str);
            output::line(format_args!("Prepare: checkpoint checked ({bytes} bytes hashed, {reused} {} reused).", if reused == "1" { "file" } else { "files" }))?;
        }
        done("Prepare", started)
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

    fn chat(&mut self, config: &Config, artifacts: &Fields, record: &Fields, options: &crate::cli::ChatOptions) -> Result<(), String> {
        let context = record.get("context_tokens").and_then(|value| value.parse::<u32>().ok())
            .filter(|value| (4096..=262144).contains(value))
            .ok_or("Invalid or missing recorded context; run vade down, then vade launch.")?;
        if context < chat::AGENT_MIN_CONTEXT && *options != crate::cli::ChatOptions::default() {
            return Err(format!("Context {context} is below the agent minimum of {}, so vade chat opens direct text chat, which saves no transcript. --resume and --name need the agent.", chat::AGENT_MIN_CONTEXT));
        }
        if context < chat::AGENT_MIN_CONTEXT {
            output::line(format_args!("{}", chat::low_context_notice(context)))?;
        } else if cfg!(target_os = "linux") {
            output::line(format_args!("{}", chat::LINUX_NOTICE))?;
        }
        let start = phase("Chat", "open local chat; /exit leaves the engine running")?;
        let workspace = std::env::current_dir().map_err(|_| "Open an existing workspace and run vade chat.")?;
        let mut exec = Exec::new("local chat", required(artifacts, "harness")?,
            chat::arguments(&config.endpoint(), required(record, "endpoint_model")?, &workspace, context, chat::chat_state_arguments()?));
        exec.args.extend(chat::session_arguments(options));
        clear_credentials(&mut exec);
        exec.env.insert("VADE_CHAT_SETUP".into(), chat::setup(1, config.drafting()));
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
        // A switch, not a secret: VADE_MCP=off turns MCP servers off in chat.
        "VADE_MCP",
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
                "{} stopped after {}",
                style::paint(Token::Warn, &format!("{:>8}", self.name), Stream::Err),
                elapsed(self.started.elapsed())
            ));
        }
    }
}
fn phase(name: &str, action: &str) -> Result<Phase, String> {
    progress(name, Token::Title, format_args!("{action}"))?;
    Ok(Phase {
        name: name.into(),
        started: Instant::now(),
        finished: false,
    })
}
fn done(name: &str, mut phase: Phase) -> Result<(), String> {
    phase.finished = true;
    progress(name, Token::Ok, format_args!("done in {}", elapsed(phase.started.elapsed())))
}
/// The step name right-aligned in one column, then what it is doing.
pub(crate) fn progress(name: &str, token: Token, text: std::fmt::Arguments<'_>) -> Result<(), String> {
    output::line(format_args!("{} {text}", style::paint(token, &format!("{name:>8}"), Stream::Out)))
}
/// A step duration: milliseconds below a second, else tenths of a second.
pub(crate) fn elapsed(duration: Duration) -> String {
    if duration < Duration::from_secs(1) { format!("{} ms", duration.as_millis()) } else { format!("{:.1} s", duration.as_secs_f64()) }
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
/// The publisher's checksum list for a checkpoint, in the engine's row form:
/// `manifest.sha256` inside it, else the list ops/fetch-model.sh publishes
/// beside the folder (`<folder>.sha256`), written by sha256sum with `./` names.
pub(crate) fn supplied_manifest(model: &Path) -> Result<Option<String>, String> {
    let sibling = model.file_name().zip(model.parent()).map(|(name, parent)| {
        let mut name = name.to_owned();
        name.push(".sha256");
        parent.join(name)
    });
    let mut found = None;
    for path in std::iter::once(model.join("manifest.sha256")).chain(sibling) {
        match open_regular(&path) {
            Ok(file) => { found = Some(file); break; }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => return Err("The model checksum manifest must be a regular file.".into()),
            Err(_) => return Err("Cannot read the model checksum manifest.".into()),
        }
    }
    let Some(file) = found else { return Ok(None) };
    let mut manifest = String::new();
    file.take(1024 * 1024 + 1)
        .read_to_string(&mut manifest)
        .map_err(|_| "Cannot read the model checksum manifest.")?;
    if manifest.len() > 1024 * 1024 {
        return Err("Model checksum manifest exceeds 1 MiB.".into());
    }
    Ok(Some(manifest.lines().map(|row| match (row.get(..64), row.get(64..68)) {
        (Some(hash), Some(mark @ ("  ./" | " *./"))) => format!("{hash}{}{}\n", &mark[..2], &row[68..]),
        _ => format!("{row}\n"),
    }).collect()))
}

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
