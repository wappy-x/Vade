//! `vade setup`: a few questions, then `vade install`, `vade up` and
//! `vade doctor`. Each step runs as its own `vade` command, so setup shares their
//! locks, checks and messages and has no install code of its own. A list of
//! Sparks is a new answer: what exists moves aside ([`super::replace`]), and
//! nothing is deleted. With no list, setup uses what exists as it is.
use super::replace::{Held, WRITTEN};
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
    /// Whether this vade runs from a launcher bundle, which can stand in for a checkout.
    pub bundle: bool,
    pub home: PathBuf,
    /// VADE_MODELS, which the steps inherit and vade install reads first.
    pub models: Option<String>,
}

/// The pinned model of one checkpoint, read from the checkout's example config.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Profile { pub fp8: bool, pub model: String, pub repo: String, pub revision: String }

/// Run setup. `None` when it is done. `Some` for `vade up --sparks` when the
/// Sparks are already set up and installed: the `vade up` options to go on with.
pub fn run(options: &SetupOptions, dry: bool) -> Result<Option<Vec<String>>, String> {
    crate::config::PublicNames::hosts(&options.hosts).run(|| {
        let saved = crate::config::config_path().ok().filter(|path| path.is_file()).and_then(|_| crate::config::Config::load().ok());
        let names = saved.as_ref().map(crate::config::PublicNames::controller).unwrap_or_default();
        names.run(|| run_public(options, dry))
    })
}

fn run_public(options: &SetupOptions, dry: bool) -> Result<Option<Vec<String>>, String> {
    let path = crate::config::config_path()?;
    let explicit = std::env::var_os("VADE_CONFIG").is_some();
    let local = !explicit && crate::launch::owner_configured();
    let chosen = !options.hosts.is_empty();
    // This Spark alone, and it is set up: that is a plain vade up.
    if options.from_up && local && matches!(options.hosts.as_slice(), [host] if crate::runner::this_computer(host)) {
        return Ok(Some(options.up.clone()));
    }
    // At a terminal, a question gives the same list as the flag. With a setup in
    // place, Enter keeps it. With none, answer() asks below.
    let asked = if !chosen && !options.yes && !answers(options) && (local || path.is_file()) && std::io::stdin().is_terminal() {
        let reply = prompt(&mut std::io::stdin().lock(), "Which Sparks? Type what you type after ssh, in any order, separated by commas [keep this setup]:")?;
        if reply.is_empty() { Vec::new() } else { sparks_in(&reply)? }
    } else { Vec::new() };
    let answered = SetupOptions { hosts: if chosen { options.hosts.clone() } else { asked }, ..options.clone() };
    let (options, chosen) = (&answered, !answered.hosts.is_empty());
    let mut held = if chosen { Some(Held::read(&path, local, &crate::launch::root()?)?) } else { None };
    if options.from_up && held.as_ref().is_some_and(Held::grouped) { return Ok(Some(hosts_start(options))); }
    // The same Sparks keep their model when no flag names one.
    let kept = SetupOptions { fp8: options.fp8.or(held.as_ref().and_then(|held| held.checkpoint(&options.hosts))), ..options.clone() };
    let options = &kept;
    let same = options.links.is_empty() && held.as_ref().is_some_and(|held| held.same(&options.hosts, options.fp8));
    if chosen && !dry {
        let mut runner = crate::runner::SystemRunner { dry: false };
        crate::runner::check_ssh(&mut runner, &options.hosts.iter().map(|host| (host.as_str(), host.as_str())).collect::<Vec<_>>())?;
    }
    // The same order as main's dispatch: explicit VADE_CONFIG, saved local
    // defaults, the legacy setup, then the default controller config.
    let target = if local && !chosen {
        refuse_answers(options, &format!("This Spark already has local setup in {}.", text::quoted(&crate::launch::root()?.join("config").to_string_lossy())))?;
        output::line(format_args!("Using the local setup in {}. Setup does not change it.", crate::launch::root()?.join("config").display()))?;
        // scripts/install.sh already ran vade install; a second one only repeats Prepare.
        let installed = crate::launch::owner_install_current();
        if installed {
            output::line(format_args!("The installed files are current, so setup skips vade install."))?;
        }
        Target::Local { model: None, installed }
    } else if !explicit && !local && crate::launch::legacy_configured() {
        return Err("This installation uses vade run qwen3-8b. Use that command, or choose a fresh VADE_HOME for vade setup.".into());
    } else if path.is_file() && (!chosen || same) {
        if !chosen { refuse_answers(options, &format!("A controller config already exists at {}.", text::quoted(&path.to_string_lossy())))?; }
        if chosen && options.hosts.len() > 1 { output::line(format_args!("Cable order: {}.", options.hosts.join(" - ")))?; }
        reuse(options, &path, chosen)?
    } else {
        let bundle = crate::launch::bundle_folder().is_ok_and(|bundle| bundle.is_some());
        let facts = Facts {
            spark: cfg!(all(target_os = "linux", target_arch = "aarch64")),
            explicit,
            path: path.clone(),
            checkout: crate::launch::setup_checkout(bundle),
            bundle,
            home: crate::launch::root()?,
            models: std::env::var("VADE_MODELS").ok().filter(|models| !models.is_empty()),
        };
        let terminal = std::io::stdin().is_terminal();
        if !options.yes && !terminal {
            return Err("Standard input is not a terminal, so setup cannot ask its questions. Add --yes to take the defaults, with --hosts and --link where they are needed. Run vade setup --help.".into());
        }
        let mut input = std::io::stdin().lock();
        let retry = options.retry();
        let mut runner = crate::runner::SystemRunner { dry: false };
        let target = answer(options, &facts, &mut |question| prompt(&mut input, question), profile,
            &mut |host| if dry { Ok(None) } else { super::links::probe(&mut runner, host, &retry).map(Some) })?;
        // The cables can give the order that this config already has.
        let chain = hosts_of(&target).unwrap_or_else(|| options.hosts.clone());
        let unchanged = match (&target, &held) {
            (Target::Controller { text: Some(text), .. }, Some(old)) => old.same_text(text) || (options.links.is_empty() && old.same(&chain, options.fp8)),
            _ => false,
        };
        if unchanged { reuse(&SetupOptions { hosts: chain, ..options.clone() }, &path, true)? } else { target }
    };
    // What stays in place does not move.
    if let (Some(old), Target::Controller { text: None, .. }) = (&mut held, &target) { old.controller = None; }
    let line = held.as_ref().and_then(|old| old.line(hosts_of(&target).map_or(options.hosts.len(), |hosts| hosts.len()), dry));
    if line.is_none() && options.from_up && let Target::Controller { text: None, up, installed: true, .. } = &target { return Ok(Some(up.clone())); }
    let steps = steps(&target);
    let inventory = if let Target::Controller { path, text, .. } = &target {
        let config = match text {
            Some(text) => crate::config::Config::parse(text, path)?,
            None => crate::config::Config::load()?,
        };
        crate::sets::inventory_text(&config)
    } else { String::new() };
    if dry {
        output::line(format_args!("Dry run: setup writes nothing and starts nothing."))?;
        output::text(format_args!("{inventory}"))?;
        if let Some(line) = &line { output::line(format_args!("{line}"))?; }
        match &target {
            Target::Local { model, .. } => output::line(format_args!("Where: this Spark\nModel: {}", local_model(model.as_deref(),
                crate::launch::owner_configured(), std::env::var("VADE_MODELS").ok().filter(|models| !models.is_empty()).as_deref())))?,
            Target::Controller { path, text: Some(config), .. } => {
                output::line(format_args!("Public output hides private addresses. The config file keeps the real values."))?;
                output::text(format_args!("Config to write at {}:\n{config}", text::quoted(&path.to_string_lossy())))?;
                if options.links.is_empty() && options.hosts.len() > 1 { output::line(format_args!("Setup reads the cable addresses from each Spark when it runs."))?; }
            }
            Target::Controller { .. } => (),
        }
        output::line(format_args!("Steps:"))?;
        for (args, _) in &steps { output::line(format_args!("  vade {}\n    {}", args.join(" "), step_note(&args[0], &target)))?; }
        return Ok(None);
    }
    output::text(format_args!("{inventory}"))?;
    let program = std::env::current_exe().map_err(|_| "Cannot find the vade program that runs setup.")?;
    let _interruptions = crate::runner::Interruptions::install();
    if let Some(old) = &held {
        old.stop(&program, options.yes)?;
        old.move_aside(&crate::launch::root()?)?;
    }
    if let Some(line) = &line { output::line(format_args!("{line}"))?; }
    if let Target::Controller { path, text: Some(config), .. } = &target {
        output::line(format_args!("Writing the controller config to {}.", text::quoted(&path.to_string_lossy())))?;
        // Create only: another setup may have written this file while this one asked.
        let created = text::create_private(path, config);
        // No controller config, so the old files go back to their places.
        if !matches!(created, Ok(true)) && let Some(old) = &held { old.restore(); }
        if !created? {
            return Err(format!("{} appeared while setup asked its questions. Setup did not change it. Run vade setup again to use it.", text::quoted(&path.to_string_lossy())));
        }
    }
    let total = steps.len();
    let again = options.retry();
    for (number, (args, env)) in steps.iter().enumerate() {
        let shown = format!("vade {}", args.join(" "));
        // A stop between steps starts nothing new.
        if crate::runner::interrupted() {
            return Err(format!("Setup was interrupted before {shown}. Run {again} again to continue."));
        }
        output::line(format_args!("Setup step {} of {total}: {shown}\n{}", number + 1, step_note(&args[0], &target)))?;
        let (passed, stopped) = run_step(step_command(&program, args, env, options.yes), &crate::runner::interrupted)
            .map_err(|_| format!("Cannot run {shown}. Check that the vade program still exists, then run {again} again."))?;
        if stopped || crate::runner::interrupted() {
            return Err(format!("Setup was interrupted during {shown}, and that step was told to stop. Run {again} again to continue."));
        }
        if !passed {
            return Err(format!("{shown} failed, so setup stopped. Setup repeats only the steps that need it. Fix the cause it printed above, then run {again} again."));
        }
    }
    output::line(format_args!("Setup done."))?;
    output::text(format_args!("{}", crate::status::next_steps(&[("Chat", &text::command("chat")), ("Status", &text::command("status")), ("Stop", &text::command("down"))])))?;
    Ok(None)
}

/// The config in place at `path`, for its own Sparks or for the `chosen` list.
/// `chosen` counts as installed only when the active install holds every Spark,
/// so a failed first install runs again.
fn reuse(options: &SetupOptions, path: &Path, chosen: bool) -> Result<Target, String> {
    let quiet = options.from_up;
    if !quiet { output::line(format_args!("Using the controller config at {}. Setup does not change it.", text::quoted(&path.to_string_lossy())))?; }
    let config = crate::config::Config::load()?;
    let file = std::fs::read_to_string(path).map_err(|_| format!("Cannot read {}. Check its permissions.", text::quoted(&path.to_string_lossy())))?;
    let running = crate::state::Session::load(&config.state_dir)?.is_some();
    let installed = running || if chosen { covered(&config) } else { config.activation.is_file() };
    if installed && !quiet {
        output::line(format_args!("This config already has an install, so setup skips vade install. After you pull new code, run vade update."))?;
    }
    if running && !quiet {
        output::line(format_args!("An engine from this config is running, so vade up attaches to it with its saved settings."))?;
    }
    let up = match (chosen, running) {
        (true, false) => { let up = up_options(config.model.ends_with("-fp8"), &options.hosts); check_start(&config, &up)?; up }
        _ => reuse_start(running, &file, &config)?,
    };
    Ok(Target::Controller { up: up.into_iter().chain(options.up.iter().cloned()).collect(), path: path.to_owned(), text: None, installed })
}

/// Whether the active install holds each machine of `config`.
fn covered(config: &crate::config::Config) -> bool {
    let mut config = config.clone();
    config.resolve_generation().is_ok() && crate::state::Manifest::load(&config.build_manifest)
        .is_ok_and(|manifest| config.machines.iter().all(|machine| manifest.artifacts.contains_key(&machine.name)))
}

/// The Sparks in the `vade up` options of `target`, in rank order.
fn hosts_of(target: &Target) -> Option<Vec<String>> {
    let Target::Controller { up, text, path, .. } = target else { return None };
    let hosts: Vec<String> = up.iter().skip_while(|word| *word != "--hosts").nth(1)?.split(',').map(str::to_owned).collect();
    let file = text.clone().or_else(|| std::fs::read_to_string(path).ok());
    let config = file.and_then(|file| crate::config::Config::parse(&file, path).ok());
    Some(hosts.into_iter().map(|host| config.as_ref().and_then(|config| config.machines.iter().find(|machine| machine.name == host)).map_or(host.clone(), |machine| machine.ssh.clone())).collect())
}

/// The Sparks in one answer line.
fn sparks_in(reply: &str) -> Result<Vec<String>, String> {
    let hosts: Vec<String> = reply.split(',').map(|host| host.trim().to_owned()).collect();
    let mut unique = std::collections::BTreeSet::new();
    if reply.is_empty() || hosts.iter().any(|host| !crate::config::ssh_destination(host) || !unique.insert(host.clone())) {
        return Err("Give one or more distinct SSH destinations, such as spark-a,spark-b; nothing was written. Run vade setup again.".into());
    }
    Ok(hosts)
}

/// `vade up` options for this list with no setup code, for a group's own refusal.
fn hosts_start(options: &SetupOptions) -> Vec<String> {
    ["--hosts".to_owned(), options.hosts.join(",")].into_iter().chain(options.fp8.filter(|fp8| *fp8).map(|_| "--fp8".to_owned())).chain(options.up.iter().cloned()).collect()
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

/// Whether the options describe a new setup.
fn answers(options: &SetupOptions) -> bool {
    !options.hosts.is_empty() || options.fp8.is_some() || options.model.is_some() || !options.links.is_empty()
}

/// The saved local defaults, the free name that setup moves them to, and
/// whether a local engine runs from them.
#[derive(Debug, PartialEq)]
pub(super) struct LocalSetup { pub file: PathBuf, pub aside: PathBuf, pub running: bool }

impl LocalSetup {
    /// The local setup in `root`.
    pub(super) fn find(root: &Path) -> Result<Self, String> {
        let file = root.join("config");
        let aside = (1..).map(|number| root.join(if number == 1 { "config.local".to_owned() } else { format!("config.local-{number}") }))
            .find(|path| std::fs::symlink_metadata(path).is_err()).unwrap_or_else(|| root.join("config.local"));
        Ok(Self { file, aside, running: root.join("state/run.conf").exists() })
    }

    /// Move the file aside, under the lock that every local start takes.
    pub(super) fn move_aside(&self, root: &Path) -> Result<(), String> {
        let _lock = crate::state::Lock::acquire(root)?;
        let now = Self::find(root)?;
        if now.running || now.aside != self.aside { return Err("The local setup changed while setup asked its questions. Nothing was moved. Run the same command again.".into()); }
        // An engine record with no saved defaults has no file to move.
        if !self.file.exists() { return Ok(()); }
        std::fs::rename(&self.file, &self.aside).map_err(|_| format!("Cannot move {} aside. Check its permissions, then run the same command again.", text::quoted(&self.file.to_string_lossy())))
    }

    /// Put the file back. A failure leaves it at the path that setup printed.
    pub(super) fn restore(&self) { let _ = std::fs::rename(&self.aside, &self.file); }
}

/// Ask one question and read one answer line.
fn prompt(input: &mut impl BufRead, question: &str) -> Result<String, String> {
    output::text(format_args!("{question} "))?;
    std::io::Write::flush(&mut std::io::stdout()).map_err(|_| "Cannot show the setup question.")?;
    read_answer(input)
}

/// Answer flags conflict with a setup that already exists.
fn refuse_answers(options: &SetupOptions, existing: &str) -> Result<(), String> {
    if !answers(options) { return Ok(()); }
    Err(format!("{existing} Setup uses it as it is, so --fp8, --fp4-local, --model and --link do not apply. Run vade setup without them, or add --hosts with the Sparks to replace that setup."))
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
/// in `checkout`, so setup and the docs name the same revision. With no
/// checkout, the copy built into this program gives the same pin.
fn profile(checkout: Option<&Path>, fp8: bool) -> Result<Profile, String> {
    let name = if fp8 { "vade.conf" } else { "fp4-local.conf" };
    let (path, source) = match checkout {
        Some(checkout) => {
            let path = checkout.join("cli/examples").join(name);
            let source = std::fs::read_to_string(&path).map_err(|_| format!("Cannot read {}. Run vade setup from a complete Vade checkout.", path.display()))?;
            (path, source)
        }
        None => (PathBuf::from(name), if fp8 { include_str!("../../examples/vade.conf") } else { include_str!("../../examples/fp4-local.conf") }.to_owned()),
    };
    let config = crate::config::Config::parse(&source, &path)?;
    config.validate_model_revision()?;
    Ok(Profile { fp8, model: config.model, repo: config.model_repo, revision: config.model_revision })
}

/// Ask the questions that flags did not answer, and choose the target.
/// `--yes` takes each default and fails on a question without one. `probe`
/// reads one Spark over SSH ([`super::links::probe`]), or gives `None` on a dry
/// run, which opens no connection.
pub(super) fn answer(options: &SetupOptions, facts: &Facts, ask: &mut dyn FnMut(&str) -> Result<String, String>,
    profile: fn(Option<&Path>, bool) -> Result<Profile, String>,
    probe: &mut dyn FnMut(&str) -> Result<Option<super::links::Spark>, String>) -> Result<Target, String> {
    let local_default = facts.spark && !facts.explicit;
    let hosts: Vec<String> = if !options.hosts.is_empty() { options.hosts.clone() } else if local_default && options.yes { Vec::new() } else if options.yes {
        return Err("Setup needs the Sparks to drive from this computer. Add --hosts with what you type after ssh, such as --hosts spark-a,spark-b, in any order.".into());
    } else {
        let reply = ask(if local_default { "Which Sparks? Type what you type after ssh, in any order, separated by commas [this Spark]:" }
            else { "Which Sparks? Type what you type after ssh, in any order, separated by commas:" })?;
        if reply.is_empty() && local_default { Vec::new() } else { sparks_in(&reply)? }
    };
    crate::config::PublicNames::hosts(&hosts).run(|| {
    if hosts.is_empty() {
        if options.fp8 == Some(true) { return Err(format!("{} This Spark alone cannot serve FP8. Add --hosts with those Sparks, or omit --fp8.", crate::config::FP8_SPARKS)); }
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
    // Before any question: no answer below can change this.
    let checkout = facts.checkout.as_deref();
    if checkout.is_none() && !facts.bundle { return Err(format!("More Sparks need a Vade checkout on this computer, and this vade program has none. Run git clone {}, then run vade setup again from that folder, or set VADE_SOURCE to a checkout.", crate::repository::VADE_REPO_URL)); }
    // Reach every Spark the way vade up will, before any other question, and
    // name every Spark that fails, so one run shows every fix.
    let mut failed = Vec::new();
    let sparks: Vec<Option<super::links::Spark>> = hosts.iter().map(|host| probe(host).unwrap_or_else(|error| { failed.push(error); None })).collect();
    if !failed.is_empty() { return Err(failed.join("\n")); }
    let given = hosts.clone();
    let cables = super::links::Cables::new(&given, sparks, &options.links);
    cables.same_home()?;
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
        return Err(format!("{} This setup has {}. Use 2 or 4 Sparks in --hosts, or omit --fp8.", crate::config::FP8_SPARKS, hosts.len()));
    }
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
    // With no chain, the questions below ask for each address in the given order.
    let hosts = match cables.chain() {
        Ok(chain) => chain,
        Err((a, b)) if options.yes => return Err(format!("{a} and {b} share no cable, so check that cable or add --link {a}:{b}=ADDRESS --link {b}:{a}=ADDRESS.")),
        Err(_) => given.clone(),
    };
    let names = machine_names(&hosts.iter().map(|host| cables.hostname(host).unwrap_or_default().to_owned()).collect::<Vec<_>>());
    if cables.read() && hosts.len() > 1 { output::line(format_args!("Cable order: {}.", names.join(" - ")))?; }
    let mut links = Vec::new();
    let mut neighbors: Vec<(&String, &String)> = hosts.windows(2).map(|pair| (&pair[0], &pair[1])).collect();
    // Keep a closing cable when the inventory or explicit cable ends name it.
    // A path still works. Setup does not ask for an extra cable.
    if let [first, _, .., last] = hosts.as_slice()
        && (cables.address(last, first).is_some() || options.links.iter().any(|(from, to, _)| (from == last && to == first) || (from == first && to == last))) {
        neighbors.push((last, first));
    }
    for (a, b) in neighbors {
        for (from, to) in [(a, b), (b, a)] {
            let given = options.links.iter().find(|(f, t, _)| f == from && t == to).map(|(_, _, address)| address.clone());
            let address = match given.or_else(|| cables.address(from, to)) {
                Some(address) => address,
                // A dry run opens no connection, so it has no address to show.
                None if !cables.read() => continue,
                None if options.yes => return Err(format!("Setup cannot tell the address of {from} on its cable to {to}: {}. Add --link {from}:{to}=ADDRESS for each cable end.", cables.unclear(from, to))),
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
    let up: Vec<String> = up_options(fp8, &names).into_iter().chain(options.up.iter().cloned()).collect();
    let parsed = crate::config::Config::parse(&text, &facts.path)?;
    if cables.read() || links.len() >= 2 * (hosts.len() - 1) {
        check_start(&parsed, &up).map_err(|error| format!("Nothing was written: vade up would refuse this setup. {error}"))?;
    }
    Ok(Target::Controller { path: facts.path.clone(), text: Some(text), up, installed: false })
    })
}

/// Use each short hostname, or a numbered label when it is absent or private.
pub(super) fn machine_names(hosts: &[String]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for (position, host) in hosts.iter().enumerate() {
        let mut name = if crate::config::public_hostname(host) { host.chars().take(56).collect() } else { format!("spark-{}", position + 1) };
        let base = name.clone();
        let mut index = 2;
        while names.contains(&name) { name = format!("{base}-{index}"); index += 1; }
        names.push(name);
    }
    names
}

/// The controller config: `checkout` as the source, or the installed bundle
/// when setup gives no checkout, Vade's private folders under the Vade home, and
/// one section per Spark with its cable addresses. Every other key keeps the
/// parser's default.
pub(super) fn render(checkout: Option<&Path>, home: &Path, profile: &Profile, hosts: &[String], names: &[String],
    links: &[(String, String, String)], model: &str) -> String {
    let start = format!("vade {}", std::iter::once("up".to_owned()).chain(up_options(profile.fp8, names)).collect::<Vec<_>>().join(" "));
    let mut out = format!("# Written by vade setup. Start: {start}\n# Edit this file to change the setup. A new list of Sparks moves it aside.\n[global]\n");
    match checkout {
        Some(checkout) => for (key, folder) in [("engine_source", "engine"), ("harness_source", "harness"), ("cli_source", "cli")] {
            out += &format!("{key} = {}\n", checkout.join(folder).display());
        },
        None => out += "source = installed-bundle\n",
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
    out += &crate::sets::sections(names);
    out
}

/// The `vade up` options for a config that is already there. The checkpoint
/// comes from the config's model, so an FP8 config starts with `--fp8` even
/// without setup's first line. The Sparks come from that line when it names
/// machines of this config. A config without it (written by hand) starts
/// every machine it lists, in its order, except `vade verify` engines.
pub(super) fn start_options(file: &str, config: &crate::config::Config) -> Vec<String> {
    let names: Vec<String> = match file.lines().next().and_then(|line| line.strip_prefix(WRITTEN)) {
        Some(rest) => match rest.split_whitespace().collect::<Vec<_>>().as_slice() {
            ["--hosts", names] | ["--fp8", "--hosts", names] => names.split(',').map(str::to_owned).collect(),
            _ => Vec::new(),
        },
        None => config.machines.iter().filter(|machine| machine.verify_url.is_none()).map(|machine| machine.name.clone()).collect(),
    };
    let mut options = Vec::new();
    if config.model.ends_with("-fp8") { options.push("--fp8".to_owned()); }
    // Even one name: a plain vade up takes the first machine, which may be a verify engine.
    if !names.is_empty() && names.iter().all(|name| config.machines.iter().any(|machine| &machine.name == name || &machine.ssh == name)) {
        let names: Vec<&str> = names.iter().filter_map(|name| config.machines.iter().find(|machine| &machine.name == name || &machine.ssh == name).map(|machine| machine.name.as_str())).collect();
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

/// The `vade up` options for a profile on these Sparks, in cable order, as the
/// user typed them, so the printed command can be pasted.
fn up_options(fp8: bool, hosts: &[String]) -> Vec<String> {
    let mut options = Vec::new();
    if fp8 { options.push("--fp8".to_owned()); }
    if hosts.len() > 1 { options.extend(["--hosts".to_owned(), hosts.join(",")]); }
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

/// What one step does, the same sentence in a run and in a dry run.
fn step_note(step: &str, target: &Target) -> &'static str {
    match (step, target) {
        ("install", Target::Local { .. }) => "This step checks this Spark and installs Vade here.",
        ("install", Target::Controller { .. }) => "This step checks SSH, the GPU, the disk and the model on each Spark, then installs Vade there.",
        ("up", _) => "This step starts the engine.",
        _ => "This step checks the running engine.",
    }
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
