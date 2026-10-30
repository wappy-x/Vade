//! `vade setup`: a few questions, then `vade install`, `vade up` and
//! `vade doctor`. Each step runs as its own `vade` command, so setup shares their
//! locks, checks and messages and has no install code of its own. Setup writes
//! only a new controller config; it never replaces one.
use crate::{cli::SetupOptions, output, text};
use std::io::{BufRead, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Where setup runs the steps: this Spark, or Sparks over SSH from a
/// controller config.
#[derive(Debug, PartialEq)]
pub(super) enum Target {
    /// This Spark; `model` becomes VADE_MODELS for the steps. `installed`
    /// skips `vade install` when the saved setup has current installed files.
    Local { model: Option<String>, installed: bool },
    /// A controller config at `path`; `text` is the new file, or `None` to use
    /// the one that is there. `up` holds the `vade up` options. `installed`
    /// skips `vade install`: a second install makes a new generation, and a
    /// running engine from the first one would then not attach.
    Controller { path: PathBuf, text: Option<String>, up: Vec<String>, installed: bool },
}

/// What setup knows before it asks: whether this computer is a Spark, whether
/// VADE_CONFIG names the config, and the checkout the steps build from.
pub(super) struct Facts {
    pub spark: bool,
    pub explicit: bool,
    pub path: PathBuf,
    pub checkout: Option<PathBuf>,
    pub home: PathBuf,
    /// VADE_MODELS, which the steps inherit and vade install reads first.
    pub models: Option<String>,
}

/// The pinned model of one checkpoint, read from the checkout's example config.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Profile { pub fp8: bool, pub model: String, pub repo: String, pub revision: String }

pub fn run(options: &SetupOptions, dry: bool) -> Result<(), String> {
    let path = crate::config::config_path()?;
    let explicit = std::env::var_os("VADE_CONFIG").is_some();
    // The same order as main's dispatch: explicit VADE_CONFIG, saved local
    // defaults, the legacy setup, then the default controller config.
    let target = if !explicit && crate::launch::owner_configured() {
        refuse_answers(options, "This Spark already has local setup.")?;
        output::line(format_args!("Using the local setup in {}. Setup does not change it.", crate::launch::root()?.join("config").display()))?;
        // scripts/install.sh already ran vade install; a second one only repeats Prepare.
        let installed = crate::launch::owner_install_current();
        if installed {
            output::line(format_args!("The installed files are current, so setup skips vade install."))?;
        }
        Target::Local { model: None, installed }
    } else if !explicit && crate::launch::legacy_configured() {
        return Err("This installation uses vade run qwen3-8b. Use that command, or choose a fresh VADE_HOME for vade setup.".into());
    } else if path.is_file() {
        refuse_answers(options, &format!("A controller config already exists at {}.", text::quoted(&path.to_string_lossy())))?;
        output::line(format_args!("Using the controller config at {}. Setup does not change it.", text::quoted(&path.to_string_lossy())))?;
        let config = crate::config::Config::load()?;
        let file = std::fs::read_to_string(&path).map_err(|_| format!("Cannot read {}. Check its permissions.", text::quoted(&path.to_string_lossy())))?;
        let running = crate::state::Session::load(&config.state_dir)?.is_some();
        let installed = running || config.activation.is_file();
        if installed {
            output::line(format_args!("This config already has an install, so setup skips vade install. After you pull new code, run vade update."))?;
        }
        if running {
            output::line(format_args!("An engine from this config is running, so vade up attaches to it with its saved settings."))?;
        }
        Target::Controller { up: reuse_start(running, &file, &config)?, path, text: None, installed }
    } else {
        let facts = Facts {
            spark: cfg!(all(target_os = "linux", target_arch = "aarch64")),
            explicit,
            path,
            checkout: crate::launch::checkout().ok(),
            home: crate::launch::root()?,
            models: std::env::var("VADE_MODELS").ok().filter(|models| !models.is_empty()),
        };
        let terminal = std::io::stdin().is_terminal();
        if !options.yes && !terminal {
            return Err("Standard input is not a terminal, so setup cannot ask its questions. Add --yes to take the defaults, with --hosts and --link where they are needed. Run vade setup --help.".into());
        }
        let mut input = std::io::stdin().lock();
        let mut ask = |question: &str| -> Result<String, String> {
            output::text(format_args!("{question} "))?;
            std::io::Write::flush(&mut std::io::stdout()).map_err(|_| "Cannot show the setup question.")?;
            read_answer(&mut input)
        };
        answer(options, &facts, &mut ask, profile)?
    };
    let steps = steps(&target);
    if dry {
        output::line(format_args!("Dry run: setup writes nothing and starts nothing."))?;
        match &target {
            Target::Local { model, .. } => output::line(format_args!("Where: this Spark\nModel: {}", local_model(model.as_deref(),
                crate::launch::owner_configured(), std::env::var("VADE_MODELS").ok().filter(|models| !models.is_empty()).as_deref())))?,
            Target::Controller { path, text: Some(config), .. } => output::text(format_args!("Config to write at {}:\n{config}", text::quoted(&path.to_string_lossy())))?,
            Target::Controller { .. } => (),
        }
        output::line(format_args!("Steps:"))?;
        for (args, _) in &steps { output::line(format_args!("  vade {}", args.join(" ")))?; }
        return Ok(());
    }
    if let Target::Controller { path, text: Some(config), .. } = &target {
        // Create only: another setup may have written this file while this one asked.
        if !text::create_private(path, config)? {
            return Err(format!("{} appeared while setup asked its questions. Setup did not change it. Run vade setup again to use it.", text::quoted(&path.to_string_lossy())));
        }
        output::line(format_args!("Wrote the controller config at {}.", text::quoted(&path.to_string_lossy())))?;
    }
    let program = std::env::current_exe().map_err(|_| "Cannot find the vade program that runs setup.")?;
    let _interruptions = crate::runner::Interruptions::install();
    let total = steps.len();
    for (number, (args, env)) in steps.iter().enumerate() {
        let shown = format!("vade {}", args.join(" "));
        // A stop between steps starts nothing new.
        if crate::runner::interrupted() {
            return Err(format!("Setup was interrupted before {shown}. Run vade setup again to continue."));
        }
        output::line(format_args!("Setup step {} of {total}: {shown}", number + 1))?;
        let (passed, stopped) = run_step(step_command(&program, args, env, options.yes), &crate::runner::interrupted)
            .map_err(|_| format!("Cannot run {shown}. Check that the vade program still exists, then run vade setup again."))?;
        if stopped || crate::runner::interrupted() {
            return Err(format!("Setup was interrupted during {shown}, and that step was told to stop. Run vade setup again to continue."));
        }
        if !passed {
            return Err(format!("{shown} failed, so setup stopped. Setup repeats only the steps that need it. Fix the cause it printed above, then run vade setup again."));
        }
    }
    output::line(format_args!("Setup done."))?;
    output::text(format_args!("{}", crate::status::next_steps(&[("Chat", &text::command("chat")), ("Status", &text::command("status")), ("Stop", &text::command("down"))])))
}

/// The model that vade install will use on this Spark, in the order it looks:
/// the answer (passed as VADE_MODELS), the saved local setup, VADE_MODELS from
/// the environment, then a search of ~/models.
pub(super) fn local_model(model: Option<&str>, saved: bool, environment: Option<&str>) -> String {
    match (model, saved, environment) {
        (Some(model), _, _) => format!("{} (as VADE_MODELS)", text::terminal_text(model)),
        (None, true, _) => "the model saved in the local setup".into(),
        (None, false, Some(models)) => format!("{} (VADE_MODELS)", text::terminal_text(models)),
        (None, false, None) => "found under ~/models by vade install".into(),
    }
}

/// Answer flags conflict with a setup that already exists.
fn refuse_answers(options: &SetupOptions, existing: &str) -> Result<(), String> {
    if options.hosts.is_empty() && options.fp8.is_none() && options.model.is_none() && options.links.is_empty() { return Ok(()); }
    Err(format!("{existing} Setup uses it as it is, so --hosts, --fp8, --fp4-local, --model and --link do not apply. Run vade setup without them, or move the setup away to start over."))
}

/// One answer line, without its line ending. End of input is an error, so a
/// closed input never takes a default by accident.
fn read_answer(input: &mut impl BufRead) -> Result<String, String> {
    let mut line = String::new();
    // At most one byte over the limit, so a line with no end cannot fill memory.
    match std::io::Read::take(&mut *input, 4097).read_line(&mut line) {
        Ok(0) => Err("Setup got no answer (end of input); nothing was written. Run vade setup again, or add --yes.".into()),
        Ok(_) if line.len() > 4096 => Err("That answer is too long; nothing was written. Run vade setup again.".into()),
        Ok(_) => Ok(line.trim().to_owned()),
        Err(_) => Err("Cannot read the answer; nothing was written. Run vade setup again.".into()),
    }
}

/// The pinned model of the FP4-Local or FP8 profile, from the example config
/// in `checkout`, so setup and the docs name the same revision.
fn profile(checkout: &Path, fp8: bool) -> Result<Profile, String> {
    let path = checkout.join("cli/examples").join(if fp8 { "vade.conf" } else { "fp4-local.conf" });
    let source = std::fs::read_to_string(&path).map_err(|_| format!("Cannot read {}. Run vade setup from a complete Vade checkout.", path.display()))?;
    let config = crate::config::Config::parse(&source, &path)?;
    config.validate_model_revision()?;
    Ok(Profile { fp8, model: config.model, repo: config.model_repo, revision: config.model_revision })
}

/// Ask the questions that flags did not answer, and choose the target.
/// `--yes` takes each default and fails on a question without one.
pub(super) fn answer(options: &SetupOptions, facts: &Facts, ask: &mut dyn FnMut(&str) -> Result<String, String>,
    profile: fn(&Path, bool) -> Result<Profile, String>) -> Result<Target, String> {
    let local_default = facts.spark && !facts.explicit;
    let hosts: Vec<String> = if !options.hosts.is_empty() { options.hosts.clone() } else if local_default && options.yes { Vec::new() } else if options.yes {
        return Err("Setup needs the Sparks to drive from this computer. Add --hosts with what you type after ssh, such as --hosts spark-a, in cable order.".into());
    } else {
        let reply = ask(if local_default { "Which Sparks? Type what you type after ssh, in cable order, separated by commas [this Spark]:" }
            else { "Which Sparks? Type what you type after ssh, in cable order, separated by commas:" })?;
        if reply.is_empty() && local_default { Vec::new() } else {
            let hosts: Vec<String> = reply.split(',').map(|host| host.trim().to_owned()).collect();
            let mut unique = std::collections::BTreeSet::new();
            if reply.is_empty() || hosts.iter().any(|host| !crate::config::ssh_destination(host) || !unique.insert(host.clone())) {
                return Err("Give one or more distinct SSH destinations, such as spark-a,spark-b; nothing was written. Run vade setup again.".into());
            }
            hosts
        }
    };
    if hosts.is_empty() {
        if options.fp8 == Some(true) { return Err("FP8 runs on 2, 4 or 6 Sparks, not on this Spark alone. Add --hosts with those Sparks, or omit --fp8.".into()); }
        if !options.links.is_empty() { return Err("One Spark has no cables to describe. Omit --link.".into()); }
        let model = match (&options.model, options.yes) {
            (Some(model), _) => Some(model.clone()),
            (None, true) => None,
            (None, false) => Some(ask(&format!("Where is the FP4-Local model on this Spark? [{}]:",
                facts.models.as_deref().map_or("found under ~/models".into(), |models| format!("{} from VADE_MODELS", text::terminal_text(models)))))?).filter(|reply| !reply.is_empty()),
        };
        // vade install reads VADE_MODELS with the same ~/ expansion.
        if model.as_deref().is_some_and(|model| !(model.starts_with('/') || model.starts_with("~/")) || model.chars().any(char::is_control)) {
            return Err("Give the model folder as an absolute path or a path under ~/, such as ~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local.".into());
        }
        return Ok(Target::Local { model, installed: false });
    }
    let fp8 = match options.fp8 {
        Some(fp8) => fp8,
        None if options.yes || !matches!(hosts.len(), 2 | 4 | 6) => false,
        None => match ask("Which checkpoint? FP4-Local or FP8 [FP4-Local]:")?.to_ascii_lowercase().as_str() {
            "" | "fp4" | "fp4-local" => false,
            "fp8" => true,
            _ => return Err("Answer FP4-Local or FP8; nothing was written. Run vade setup again.".into()),
        },
    };
    if fp8 && !matches!(hosts.len(), 2 | 4 | 6) {
        return Err(format!("FP8 runs on 2, 4 or 6 Sparks, and this setup has {}. Use 2, 4 or 6 Sparks in --hosts, or omit --fp8.", hosts.len()));
    }
    let checkout = facts.checkout.as_deref().ok_or("Setup needs the Vade checkout that holds this vade program. Run it from your checkout, or set VADE_SOURCE to the checkout.")?;
    let profile = profile(checkout, fp8)?;
    let default_model = format!("~/models/{}", profile.repo);
    let model = match (&options.model, options.yes) {
        (Some(model), _) => model.clone(),
        (None, true) => default_model.clone(),
        (None, false) => Some(ask(&format!("Where is the model on each Spark? [{default_model}]:"))?).filter(|reply| !reply.is_empty()).unwrap_or(default_model.clone()),
    };
    // Check it here, before the cable questions, with the parser's own rule for model_path.
    if crate::config::remote_path(model.clone(), "", "model_path").is_err() {
        return Err(format!("Give the model folder on each Spark as an absolute path or a path under ~/, with only letters, digits and _ - . /, such as {default_model}; nothing was written. Run vade setup again."));
    }
    let names = machine_names(&hosts);
    let mut links = Vec::new();
    for pair in hosts.windows(2) {
        for (from, to) in [(&pair[0], &pair[1]), (&pair[1], &pair[0])] {
            let given = options.links.iter().find(|(f, t, _)| f == from && t == to).map(|(_, _, address)| address.clone());
            let address = match given {
                Some(address) => address,
                None if options.yes => return Err(format!("Setup needs the address of {from} on its cable to {to}. Add --link {from}:{to}=ADDRESS for each cable end.")),
                None => ask(&format!("Address of {from} on its cable to {to}:"))?,
            };
            if !crate::config::fabric_address(&address) {
                return Err(format!("Give the address of {from} on its cable to {to} as a host name or IP address, without a port; nothing was written. Run vade setup again."));
            }
            links.push((from.clone(), to.clone(), address));
        }
    }
    if let Some((from, to, _)) = options.links.iter().find(|(from, to, _)| !links.iter().any(|(f, t, _)| f == from && t == to)) {
        return Err(format!("--link {from}:{to} does not name two neighbors in --hosts. Give one --link per cable end between neighbors in cable order."));
    }
    let text = render(checkout, &facts.home, &profile, &hosts, &names, &links, &model);
    // The steps read this file, so it must pass the same parser and the
    // same Spark and cable checks as vade up before setup writes it.
    let up = up_options(fp8, &names);
    check_start(&crate::config::Config::parse(&text, &facts.path)?, &up)
        .map_err(|error| format!("Nothing was written: vade up would refuse this setup. {error}"))?;
    Ok(Target::Controller { path: facts.path.clone(), text: Some(text), up, installed: false })
}

/// A config section name for each SSH destination: its host part with every
/// other character as `-`, made unique.
pub(super) fn machine_names(hosts: &[String]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for host in hosts {
        let host = host.rsplit('@').next().unwrap_or(host);
        let mut name: String = host.chars().take(56).map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') { c } else { '-' }).collect();
        if !name.starts_with(|c: char| c.is_ascii_alphanumeric()) { name.insert_str(0, "spark"); }
        let base = name.clone();
        let mut index = 2;
        while names.contains(&name) { name = format!("{base}-{index}"); index += 1; }
        names.push(name);
    }
    names
}

/// The controller config: this checkout as the source, Vade's private folders
/// under the Vade home, and one section per Spark with its cable addresses.
/// Every other key keeps the parser's default.
pub(super) fn render(checkout: &Path, home: &Path, profile: &Profile, hosts: &[String], names: &[String],
    links: &[(String, String, String)], model: &str) -> String {
    let start = format!("vade {}", std::iter::once("up".to_owned()).chain(up_options(profile.fp8, names)).collect::<Vec<_>>().join(" "));
    let mut out = format!("# Written by vade setup. Start: {start}\n# Edit this file to change the setup; vade setup never replaces it.\n[global]\n");
    for (key, folder) in [("engine_source", "engine"), ("harness_source", "harness"), ("cli_source", "cli")] {
        out += &format!("{key} = {}\n", checkout.join(folder).display());
    }
    out += &format!("stage_dir = {}\nstate_dir = {}\n", home.join("controller/stage").display(), home.join("controller/state").display());
    out += &format!("model = {}\nmodel_repo = {}\nmodel_revision = {}\n", profile.model, profile.repo, profile.revision);
    for (host, name) in hosts.iter().zip(names) {
        out += &format!("\n[machine {name}]\nssh = {host}\nmodel_path = {model}\n");
        for (_, to, address) in links.iter().filter(|(from, _, _)| from == host) {
            let peer = &names[hosts.iter().position(|host| host == to).unwrap_or_default()];
            out += &format!("fabric.{peer} = {address}\n");
        }
    }
    out
}

/// The `vade up` options for a config that is already there. The checkpoint
/// comes from the config's model, so an FP8 config starts with `--fp8` even
/// without setup's first line. The Sparks come from that line when it names
/// machines of this config. A config without it (written by hand) starts
/// every machine it lists, in its order, except `vade verify` engines.
pub(super) fn start_options(file: &str, config: &crate::config::Config) -> Vec<String> {
    let names: Vec<String> = match file.lines().next().and_then(|line| line.strip_prefix("# Written by vade setup. Start: vade up")) {
        Some(rest) => match rest.split_whitespace().collect::<Vec<_>>().as_slice() {
            ["--hosts", names] | ["--fp8", "--hosts", names] => names.split(',').map(str::to_owned).collect(),
            _ => Vec::new(),
        },
        None => config.machines.iter().filter(|machine| machine.verify_url.is_none()).map(|machine| machine.name.clone()).collect(),
    };
    let mut options = Vec::new();
    if config.model.ends_with("-fp8") { options.push("--fp8".to_owned()); }
    // Even one name: a plain vade up takes the first machine, which may be a verify engine.
    if !names.is_empty() && names.iter().all(|name| config.machines.iter().any(|machine| &machine.name == name)) {
        options.extend(["--hosts".to_owned(), names.join(",")]);
    }
    options
}

/// The `vade up` options for a config that is already there. A running engine
/// keeps its saved launch settings (such as a smaller --context) only under a
/// bare `vade up`: any option makes up compare a fresh launch and refuse it.
pub(super) fn reuse_start(running: bool, file: &str, config: &crate::config::Config) -> Result<Vec<String>, String> {
    if running { return Ok(Vec::new()); }
    let up = start_options(file, config);
    check_start(config, &up)?;
    Ok(up)
}

/// The checks that `vade up` makes on its Sparks and cables for these options,
/// such as both ends of each cable and different addresses on it, run before
/// setup writes a config or starts an install.
pub(super) fn check_start(config: &crate::config::Config, up: &[String]) -> Result<(), String> {
    let args: Vec<String> = std::iter::once("up".to_owned()).chain(up.iter().cloned()).collect();
    match crate::cli::parse(&args)?.command {
        crate::cli::Command::Up(options) => config.selected(&options).map(|_| ()),
        _ => Err("Setup built an invalid vade up command. Run vade up yourself.".into()),
    }
}

/// The `vade up` options for a profile on these machines, in cable order.
fn up_options(fp8: bool, names: &[String]) -> Vec<String> {
    let mut options = Vec::new();
    if fp8 { options.push("--fp8".to_owned()); }
    if names.len() > 1 { options.extend(["--hosts".to_owned(), names.join(",")]); }
    options
}

/// install, up and doctor, with their arguments and environment. Install
/// checks every Spark (SSH, GPU, disk, model and network) before it builds.
/// Doctor runs last, on the running engine, so it checks exactly the Sparks
/// and links that up chose; before up it would check only its default
/// selection. A setup that already has a current install starts at up.
fn steps(target: &Target) -> Vec<(Vec<String>, Vec<(String, String)>)> {
    let (up, env, installed) = match target {
        Target::Local { model, installed } => (Vec::new(), model.iter().map(|model| ("VADE_MODELS".to_owned(), model.clone())).collect(), *installed),
        Target::Controller { up, installed, .. } => (up.clone(), Vec::new(), *installed),
    };
    [vec!["install".to_owned()], std::iter::once("up".to_owned()).chain(up).collect(), vec!["doctor".to_owned()]]
        .into_iter().skip(usize::from(installed)).map(|args| (args, env.clone())).collect()
}

/// The command for one step. With `yes` nobody answers, so a step that would
/// ask gets end of input and stops with its own message instead.
pub(super) fn step_command(program: &Path, args: &[String], env: &[(String, String)], yes: bool) -> Command {
    let mut command = Command::new(program);
    command.args(args).envs(env.iter().cloned());
    if yes { command.stdin(Stdio::null()); }
    command
}

/// Run one step on this terminal. Once `stop` is true, pass SIGTERM on to the
/// step so it unwinds its own work, then wait for it. A Ctrl-C at the terminal
/// reaches the step directly; this covers a signal sent to setup alone.
/// Returns whether the step passed and whether it was told to stop.
pub(super) fn run_step(mut command: Command, stop: &dyn Fn() -> bool) -> std::io::Result<(bool, bool)> {
    let mut child = command.spawn()?;
    let mut stopped = false;
    loop {
        // Signal only before the step is reaped, so its process ID is still its own.
        match child.try_wait() {
            Ok(Some(status)) => return Ok((status.success(), stopped)),
            Ok(None) if !stopped && stop() => {
                stopped = true;
                let sent = Command::new("kill").args(["-TERM", &child.id().to_string()])
                    .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
                if !sent.is_ok_and(|status| status.success()) { let _ = child.kill(); }
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(50)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        }
    }
}

#[cfg(test)]
#[path = "walk_tests.rs"]
mod tests;
