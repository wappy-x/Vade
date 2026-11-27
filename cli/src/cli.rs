//! Argument parsing and per-command help. [`parse`] turns arguments into a
//! [`Cli`] without reading config, files or the network; [`help`] returns the
//! usage text for `vade help <command>`.
/// A parsed command line: the command, the global `--dry-run` flag and
/// `--json` for the read-only commands that take it (`status --json` keeps
/// its own [`Command::StatusJson`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cli {
    pub command: Command,
    pub dry_run: bool,
    pub json: bool,
}

/// One `vade` command, with its parsed options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Bare `vade`: a one-screen summary of what is running.
    Menu,
    /// `vade up`: start or attach to an engine.
    Up(UpOptions),
    /// `vade run qwen3-8b`: the separate Qwen3-8B BF16 bootstrap.
    Run(RunOptions),
    /// `vade down`: stop what vade started.
    Down,
    /// `vade status`.
    Status,
    /// `vade status --json`.
    StatusJson,
    /// `vade logs`.
    Logs,
    /// `vade chat`, with an optional saved session to resume and a name.
    Chat(ChatOptions),
    /// `vade sessions`: the harness arguments for list, export or delete.
    Sessions(Vec<String>),
    /// `vade doctor`.
    Doctor,
    /// `vade doctor --fabric [--ring]`: every configured cable, port and
    /// clock; `ring` also expects one closed ring of every machine.
    DoctorFabric { ring: bool },
    /// `vade install`.
    Install,
    /// `vade setup`: ask a few questions, then install, start and check.
    Setup(SetupOptions),
    /// `vade detect`.
    Detect,
    /// `vade config`: which setup is selected and what it holds.
    Config,
    /// `vade launch`: install if needed, start on this Spark, then chat.
    Launch(UpOptions),
    /// `vade update`.
    Update,
    /// `vade help [command]` or `<command> --help`.
    Help(Option<String>),
    /// `vade --version`.
    Version,
    /// `vade pull`, with its arguments (parsed by [`crate::pull`]).
    Pull(Vec<String>),
    /// `vade models`, with its arguments.
    Models(Vec<String>),
    /// `vade connect`, with its arguments (parsed by [`crate::connect`]).
    Connect(Vec<String>),
    /// The private `__worker` helper, with its arguments.
    Worker(Vec<String>),
}

/// Options for `vade chat`: a saved session to resume (ID, name or `last`)
/// and a display name for the session. The harness checks both values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChatOptions {
    pub resume: Option<String>,
    pub name: Option<String>,
}

/// Answers for `vade setup` given as flags. `--yes` takes the defaults for
/// the rest and never asks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SetupOptions {
    pub yes: bool,
    /// `--replace-local`: an older word, kept for scripts. A list of Sparks replaces a setup by itself.
    pub replace_local: bool,
    /// SSH destinations in any order; empty means this Spark (or ask).
    pub hosts: Vec<String>,
    /// `Some(true)` for `--fp8`, `Some(false)` for `--fp4-local`.
    pub fp8: Option<bool>,
    pub model: Option<String>,
    /// `--link FROM:TO=ADDRESS`: FROM's own address on its cable to TO.
    pub links: Vec<(String, String, String)>,
    /// More `vade up` options for the start, from `vade up --sparks a,b`.
    pub up: Vec<String>,
    /// This setup comes from `vade up --sparks a,b`.
    pub from_up: bool,
}

impl SetupOptions {
    /// The command to run again after a fix.
    pub fn retry(&self) -> String {
        if self.from_up { format!("vade up --sparks {}", self.hosts.join(",")) } else { "vade setup".into() }
    }
}

/// Options for `vade run qwen3-8b`: source checkouts or a release, local
/// weights and the local port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunOptions {
    pub engine_source: Option<std::path::PathBuf>,
    pub harness_source: Option<std::path::PathBuf>,
    pub weights_source: Option<std::path::PathBuf>,
    pub release: Option<std::path::PathBuf>,
    pub port: u16,
}
impl Default for RunOptions {
    fn default() -> Self {
        Self {
            engine_source: None,
            harness_source: None,
            weights_source: None,
            release: None,
            port: 8080,
        }
    }
}

/// Why a text service that named no depth runs at depth 0. Nobody chose that depth: a later start tries the
/// default depth again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftFall {
    /// The model folder has no drafting layer.
    NoLayer,
    /// The default depth did not fit in free memory at this start.
    Memory,
}

impl DraftFall {
    /// The word in the session record.
    pub fn word(self) -> &'static str {
        match self { Self::NoLayer => "no_layer", Self::Memory => "memory" }
    }
    pub fn parse(word: &str) -> Result<Self, String> {
        match word {
            "no_layer" => Ok(Self::NoLayer),
            "memory" => Ok(Self::Memory),
            _ => Err(format!("The session record names an unknown draft_fall: {word}.")),
        }
    }
    /// The line `vade up` prints before its second start.
    pub fn line(self) -> &'static str {
        match self {
            Self::NoLayer => "This model has no drafting layer. Starting without drafting.",
            Self::Memory => "Not enough free memory for drafting. Starting without drafting.",
        }
    }
    /// The depth as status and the ready line show it.
    pub fn label(self) -> &'static str {
        match self {
            Self::NoLayer => "0 (off: this model has no drafting layer)",
            Self::Memory => "0 (off: not enough free memory for drafting)",
        }
    }
}

/// Options for `vade up` and `vade launch`, after defaults are applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpOptions {
    /// Optional short model alias resolved to its canonical `<org>/<name>` repository
    /// (for example `vade up inkling` -> `thinkingmachines/Inkling-2-Mini`).
    pub model: Option<String>,
    /// Launch flags supplied on the command line, independent of defaults or environment.
    pub supplied_options: std::collections::BTreeSet<String>,
    pub sparks: Option<usize>,
    /// `--sparks` named the Sparks. `hosts` holds them, as `--hosts` would.
    pub listed: bool,
    /// With `listed`: `--link FROM:TO=ADDRESS` cable ends, as `vade setup` takes them.
    pub links: Vec<(String, String, String)>,
    /// With `listed`: the options as given, without `--sparks`, `--link` and the checkpoint flag.
    pub rest: Vec<String>,
    pub hosts: Vec<String>,
    /// Independent servers behind a local gateway: one per machine, or one per pair.
    pub replicas: bool,
    /// Sparks in each replica: 1, or 2 with `--pairs` (a tensor-split pair each).
    pub replica_sparks: usize,
    pub mtp: u32,
    pub mtp_explicit: bool,
    pub fp4_local: bool,
    pub context: u32,
    pub output: u32,
    pub output_explicit: bool,
    pub chats: Option<usize>,
    /// The legacy `--fast-prompts` flag: accepted and recorded, with no effect on
    /// arithmetic.
    pub fast_prompts: bool,
    pub memory_bytes: Option<u64>,
    pub cache_bytes: Option<u64>,
    /// Optional explicit HTTP bind address (for example `0.0.0.0:18080`). Defaults to `127.0.0.1:<port>`.
    pub bind: Option<std::net::SocketAddr>,
    /// Optional persistent NVMe disk prefix cache directory (defaults to `$VADE_HOME/cache/prefix` for Inkling).
    pub disk_cache_dir: Option<String>,
    /// Optional explicit persistent NVMe disk prefix cache byte budget (defaults to hardware-derived NVMe budget).
    pub disk_cache_bytes: Option<u64>,
    /// Disable persistent NVMe disk prefix caching (`--no-disk-cache`).
    pub no_disk_cache: bool,
    /// The engine's decode_fuse option: Some(true) runs one Spark's decode steps as a chain.
    pub decode_fuse: Option<bool>,
    pub diagnostic_unqualified_text: bool,
    /// Never set by a flag. The checkpoint plan, read before the session is built.
    /// `None` means that plan has not been read yet.
    pub checkpoint_service: Option<crate::text_service::TextService>,
    /// Never set by a flag. A dry run sets it when the checkpoint is on the Spark
    /// and the preview got no text-service reply. The Spark count is not refused.
    pub text_width_unread: bool,
    /// Never set by a flag. `vade up` sets it for a second start, after the text service's default depth met a
    /// model folder with no drafting layer or too little free memory: this start runs at depth 0, and the record
    /// says that nobody asked for 0.
    pub draft_fall: Option<DraftFall>,
}

impl UpOptions {
    /// Whether this command explicitly named the Inkling 2 Mini alias.
    pub fn is_inkling(&self) -> bool {
        self.model.as_deref() == Some(crate::pull::alias::INKLING_REPO)
    }
}

impl Default for UpOptions {
    fn default() -> Self {
        Self {
            model: None,
            supplied_options: Default::default(),
            sparks: None,
            listed: false,
            links: Vec::new(),
            rest: Vec::new(),
            hosts: Vec::new(),
            replicas: false,
            replica_sparks: 1,
            mtp: 2,
            mtp_explicit: false,
            fp4_local: true,
            context: FP4_CONTEXT,
            output: fp4_output(FP4_CONTEXT),
            output_explicit: false,
            chats: None,
            fast_prompts: false,
            memory_bytes: None,
            cache_bytes: None,
            bind: None,
            disk_cache_dir: None,
            disk_cache_bytes: None,
            no_disk_cache: false,
            decode_fuse: None,
            diagnostic_unqualified_text: false,
            checkpoint_service: None,
            text_width_unread: false,
            draft_fall: None,
        }
    }
}

/// Parse the arguments after the program name. Errors name the fix and the
/// help command to read.
pub fn parse(args: &[String]) -> Result<Cli, String> {
    if args.first().is_some_and(|arg| arg == "__worker") {
        return Ok(Cli {
            command: Command::Worker(args[1..].to_vec()),
            dry_run: false,
            json: false,
        });
    }
    let dry_run = args.iter().any(|arg| arg == "--dry-run");
    let mut args: Vec<&str> = args
        .iter()
        .filter(|arg| arg.as_str() != "--dry-run")
        .map(String::as_str)
        .collect();
    // A leading --json reads as if it followed the command: vade --json doctor.
    if args.len() > 1 && args[0] == "--json" { args.rotate_left(1); }
    // --json belongs to the read-only commands; `status --json` keeps its own form.
    let json = args.first().is_some_and(|command| matches!(*command, "doctor" | "detect" | "config" | "version" | "--version" | "-V"))
        && args[1..].contains(&"--json");
    if json { args.retain(|arg| *arg != "--json"); }
    let Some(command) = args.first().copied() else {
        return Ok(Cli {
            command: Command::Menu,
            dry_run,
            json,
        });
    };
    if matches!(command, "version" | "--version" | "-V") {
        if args.len() > 1 && args[1..].iter().all(|arg| matches!(*arg, "--help" | "-h")) {
            return Ok(Cli { command: Command::Help(Some("version".into())), dry_run, json });
        }
        if args.len() != 1 { return Err("Use vade --version without additional arguments.".into()); }
        return Ok(Cli { command: Command::Version, dry_run, json });
    }
    if matches!(command, "--help" | "-h" | "help") {
        // Help is text only, so --json changes nothing.
        args.retain(|arg| *arg != "--json");
        let topic = |name: &str| known_command(name) || matches!(name, "bench" | "verify" | "test");
        if args.len() > 2 || (args.len() == 2 && !topic(args[1])) {
            return Err(match args.get(1) {
                Some(topic) if args.len() == 2 => format!("No help topic is named {}. Run vade --help to see the commands.", crate::text::argument(topic)),
                _ => "Use vade help, or vade help <command>.".into(),
            });
        }
        return Ok(Cli {
            command: Command::Help(args.get(1).map(|s| (*s).into())),
            dry_run,
            json,
        });
    }
    if !known_command(command) {
        // bench, verify and test parse their own options before this parser runs,
        // so they reach it only with an option before their name.
        if matches!(command, "bench" | "verify" | "test") {
            return Err(if dry_run || args.contains(&"--json") {
                format!("vade {command} takes no --dry-run or --json. Run vade {command} --help to see its options.")
            } else {
                format!("vade {command} reads only the options after its name. Run vade {command} --help to see them.")
            });
        }
        if command.starts_with('-') {
            return Err(format!("Unknown option {}. Put options after the command name. Run vade --help.", crate::text::argument(command)));
        }
        if command == "serve" {
            return Err("Unknown command \"serve\". Run vade up to start the engine in the background, or vade launch to start it and open chat.".into());
        }
        return Err(format!("Unknown command {}. Run vade --help to see the commands.", crate::text::argument(command)));
    }
    if args[1..].iter().take_while(|arg| **arg != "--").any(|arg| matches!(*arg, "--help" | "-h")) {
        return Ok(Cli {
            command: Command::Help(Some(command.into())),
            dry_run,
            json,
        });
    }
    let parsed = if command == "run" {
        if args.get(1).copied() != Some("qwen3-8b") {
            return Err(
                "Use vade launch for Flash-Next FP4-Local, or vade run qwen3-8b for the separate official Qwen3-8B BF16 catalog.".into(),
            );
        }
        let mut options = RunOptions::default();
        let mut seen = std::collections::BTreeSet::new();
        let mut index = 2;
        while index < args.len() {
            let flag = args[index];
            if !matches!(flag, "--engine-source" | "--harness-source" | "--weights-source" | "--release" | "--port") {
                return Err(unknown_option(flag, "run", &["--engine-source", "--harness-source", "--weights-source", "--release", "--port"]));
            }
            if !seen.insert(flag) {
                return Err(format!("Use {flag} only once. Run vade run --help."));
            }
            let value = option_value(&args, index)
                .ok_or_else(|| format!("Give {flag} a value. Run vade run --help."))?;
            match flag {
                "--engine-source" => options.engine_source = Some(value.into()),
                "--harness-source" => options.harness_source = Some(value.into()),
                "--weights-source" => options.weights_source = Some(value.into()),
                "--release" => options.release = Some(value.into()),
                "--port" => options.port = number(value, flag, 1024, 65535)? as u16,
                _ => unreachable!(),
            }
            index += 2;
        }
        if options.release.is_some()
            && (options.engine_source.is_some() || options.harness_source.is_some())
        {
            return Err("Choose --release or source checkouts.".into());
        }
        Command::Run(options)
    } else if matches!(command, "pull" | "models" | "connect") {
        let rest = args[1..].iter().map(|arg| (*arg).to_owned()).collect();
        match command {
            "pull" => Command::Pull(rest),
            "models" => Command::Models(rest),
            _ => Command::Connect(rest),
        }
    } else if command == "status" && args.get(1) == Some(&"--json") {
        if args.len() != 2 { return Err("Use vade status --json [--dry-run]. No other status options are supported.".into()); }
        Command::StatusJson
    } else if command == "chat" {
        Command::Chat(parse_chat(&args[1..])?)
    } else if command == "sessions" {
        Command::Sessions(parse_sessions(&args[1..])?)
    } else if command == "doctor" && args.get(1) == Some(&"--fabric") {
        let ring = args.get(2) == Some(&"--ring");
        if args.len() != 2 + usize::from(ring) { return Err("Use vade doctor --fabric [--ring] [--json] [--dry-run]. No other doctor options are supported.".into()); }
        Command::DoctorFabric { ring }
    } else if command == "setup" {
        Command::Setup(parse_setup(&args[1..])?)
    } else if matches!(command, "up" | "launch") {
        let options = parse_up(&args[1..]).map_err(|error| error.replace("vade up", &format!("vade {command}")))?;
        if command == "launch" && options.listed {
            return Err("vade launch starts this Spark. To start Sparks over SSH, run vade up --sparks with their SSH names.".into());
        }
        if command == "launch" { Command::Launch(options) } else { Command::Up(options) }
    } else {
        if args.len() != 1 {
            return Err(format!(
                "Unexpected argument {} for vade {command}. This command takes no options except {}. Run vade {command} --help.", crate::text::argument(args[1]),
                if matches!(command, "doctor" | "detect" | "config") { "--json and --dry-run" } else { "--dry-run" }
            ));
        }
        match command {
            "down" => Command::Down,
            "status" => Command::Status,
            "logs" => Command::Logs,
            "doctor" => Command::Doctor,
            "install" => Command::Install,
            "detect" => Command::Detect,
            "config" => Command::Config,
            "update" => Command::Update,
            _ => unreachable!(),
        }
    };
    Ok(Cli {
        command: parsed,
        dry_run,
        json,
    })
}

fn known_command(command: &str) -> bool {
    matches!(
        command,
        "run" | "up" | "down" | "status" | "logs" | "chat" | "sessions" | "doctor" | "install" | "update" | "version" | "detect" | "launch" | "config"
            | "pull" | "models" | "setup" | "connect"
    )
}

fn parse_chat(args: &[&str]) -> Result<ChatOptions, String> {
    let mut options = ChatOptions::default();
    let mut rest = args.iter().enumerate();
    // The option whose value came last: a bare word after it is most likely
    // the rest of an unquoted value, such as --name login bug.
    let mut after: Option<&str> = None;
    while let Some((index, &flag)) = rest.next() {
        let slot = match flag {
            "--resume" => &mut options.resume,
            "--name" => &mut options.name,
            "--json" => return Err("vade chat has no --json output; it is an interactive chat. Run vade status --json for the engine state as JSON.".into()),
            _ if !flag.starts_with('-') => return Err(match after {
                Some(option) => format!("Unexpected word {} after the {option} value. Put a value with spaces in quotes, for example vade chat {option} \"login bug\".", crate::text::argument(flag)),
                None => format!("vade chat takes no prompt on the command line ({}). Run vade chat, then type the message and press Enter.",
                    crate::text::argument(&args[index..].join(" "))),
            }),
            _ => return Err(format!("Unknown option {} for vade chat. Run vade chat --help.", crate::text::argument(flag))),
        };
        let value = rest.next().map(|(_, v)| v).filter(|v| !v.is_empty() && !v.starts_with("--"))
            .ok_or_else(|| format!("Give {flag} a value. Run vade chat --help."))?;
        after = Some(flag);
        if slot.replace((*value).into()).is_some() {
            return Err(format!("Use {flag} only once. Run vade chat --help."));
        }
    }
    Ok(options)
}

/// The harness arguments for `vade sessions`: `list [--all]` (the default),
/// `export ID [--markdown]` or `delete ID`.
fn parse_sessions(args: &[&str]) -> Result<Vec<String>, String> {
    let usage = "Use vade sessions [list] [--all], vade sessions export ID|NAME|last [--markdown] or vade sessions delete ID. Run vade sessions --help.";
    let id = |value: Option<&&str>| value.filter(|v| !v.is_empty() && !v.starts_with('-')).map(|v| (*v).to_string()).ok_or(usage);
    Ok(match args {
        [] | ["list"] => vec!["list".into()],
        ["--all"] | ["list", "--all"] => vec!["list".into(), "--all".into()],
        ["export", rest @ ..] if matches!(rest.get(1..), Some([]) | Some(["--markdown"])) => vec!["export".into(), id(rest.first())?, "--markdown".into()],
        ["delete", value] => vec!["delete".into(), id(Some(value))?],
        _ => return Err(usage.into()),
    })
}

fn parse_setup(args: &[&str]) -> Result<SetupOptions, String> {
    let mut options = SetupOptions::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index];
        if flag != "--link" && !seen.insert(flag) {
            return Err(format!("Use {flag} only once. Run vade setup --help."));
        }
        match flag {
            "--yes" => options.yes = true,
            "--replace-local" => options.replace_local = true,
            "--fp8" | "--fp4-local" => {
                if options.fp8.replace(flag == "--fp8").is_some() { return Err("Choose --fp4-local or --fp8. Run vade setup --help.".into()); }
            }
            "--hosts" | "--model" | "--link" => {
                let value = option_value(args, index).ok_or_else(|| format!("Give {flag} a value. Run vade setup --help."))?;
                match flag {
                    "--hosts" => options.hosts = ssh_targets(value).ok_or(HOSTS)?,
                    "--model" => options.model = Some(value.into()),
                    _ => {
                        let link = link(value, &options.links, "setup")?;
                        options.links.push(link);
                    }
                }
                index += 1;
            }
            _ => return Err(unknown_option(flag, "setup", &["--hosts", "--model", "--link"])),
        }
        index += 1;
    }
    Ok(options)
}

/// The error for a bad `--hosts`, which means the same in `vade up` and `vade setup`.
const HOSTS: &str = "Give --hosts distinct SSH destinations (what you type after ssh), separated by commas, with no spaces.";

/// Distinct SSH destinations from a comma list, in the order given.
fn ssh_targets(value: &str) -> Option<Vec<String>> {
    let hosts: Vec<String> = value.split(',').map(str::to_owned).collect();
    let mut unique = std::collections::BTreeSet::new();
    hosts.iter().all(|host| crate::config::ssh_destination(host) && unique.insert(host)).then(|| hosts.clone())
}

fn parse_up(args: &[&str]) -> Result<UpOptions, String> {
    let mut options = UpOptions::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut replicas = 0;
    let mut listed = false;
    let mut index = 0;
    let inkling_named = {
        let mut scan = 0;
        let mut found = false;
        while scan < args.len() {
            let word = args[scan];
            if UP_VALUE_OPTIONS.contains(&word) || word == "--link" {
                scan += 2;
                continue;
            }
            if !word.starts_with('-') && crate::pull::alias::is_inkling(word) {
                found = true;
            }
            scan += 1;
        }
        found
    };
    let deep_mtp_ok = args.contains(&"--diagnostic-unqualified-text")
        || inkling_named
        || (!args.contains(&"--fp4-local") && !args.contains(&"--fp8") && !args.contains(&"--fp8-legacy"));
    while index < args.len() {
        // --fp8-legacy is the older spelling of --fp8; both name the same profile.
        let flag = if args[index] == "--fp8-legacy" { "--fp8" } else { args[index] };
        if !flag.starts_with('-') {
            if let Some(alias) = crate::pull::alias::resolve(flag) {
                if options.model.replace(alias.repo.to_owned()).is_some() {
                    return Err("Give one model name. Run vade up --help.".into());
                }
                index += 1;
                continue;
            }
            return Err(format!("Unknown model {} for vade up. Use inkling or qwen, or set VADE_MODELS=<folder>. Run vade up --help.", crate::text::argument(flag)));
        }
        if !seen.insert(flag) {
            return Err(format!("Use {flag} only once. Run vade up --help."));
        }
        if matches!(flag, "--fp4-local" | "--fp8") {
            if seen.contains("--fp4-local") && seen.contains("--fp8") {
                return Err("Choose --fp4-local or --fp8. Run vade up --help.".into());
            }
            options.fp4_local = flag == "--fp4-local";
            index += 1;
            continue;
        }
        if flag == "--diagnostic-unqualified-text" {
            options.diagnostic_unqualified_text = true;
            index += 1;
            continue;
        }
        if flag == "--no-disk-cache" {
            options.no_disk_cache = true;
            index += 1;
            continue;
        }
        if flag == "--fast-prompts" {
            options.fast_prompts = true;
            index += 1;
            continue;
        }
        if flag == "--pairs" {
            options.replica_sparks = 2;
            index += 1;
            continue;
        }
        // One --link per cable end, as vade setup takes it.
        if flag == "--link" {
            seen.remove(flag);
            let value = option_value(args, index).ok_or("Give --link a value. Run vade up --help.")?;
            let link = link(value, &options.links, "up")?;
            options.links.push(link);
            index += 2;
            continue;
        }
        if !UP_VALUE_OPTIONS.contains(&flag) {
            return Err(unknown_option(flag, "up", &UP_VALUE_OPTIONS));
        }
        let value = option_value(args, index)
            .ok_or_else(|| format!("Give {flag} a value. Run vade up --help."))?;
        match flag {
            // A count, or the Sparks themselves, which is --hosts by another name.
            "--sparks" if value.bytes().all(|byte| byte.is_ascii_digit()) => {
                options.sparks = Some(min_number(value, flag, 1).map_err(|_| sparks_error())? as usize);
            }
            "--sparks" => {
                options.hosts = ssh_targets(value).ok_or_else(sparks_error)?;
                listed = true;
            }
            "--replicas" => {
                options.replicas = true;
                replicas = min_number(value, flag, 2)? as usize;
            }
            "--hosts" => options.hosts = ssh_targets(value).ok_or(HOSTS)?,
            "--mtp" => {
                options.mtp = number(value, flag, 0, if deep_mtp_ok { 7 } else { 3 })?;
                options.mtp_explicit = true;
            },
            "--context" => options.context = min_number(value, flag, 4096)?,
            "--output" => {
                options.output = min_number(value, flag, 1)?;
                options.output_explicit = true;
            }
            "--chats" => options.chats = Some(min_number(value, flag, 1)? as usize),
            "--memory-bytes" => options.memory_bytes = Some(bytes(value, flag)?),
            "--cache-bytes" => options.cache_bytes = Some(bytes(value, flag)?),
            "--bind" => {
                let addr: std::net::SocketAddr = value.parse().map_err(|_| {
                    "Set --bind to an IP:PORT socket address (for example 127.0.0.1:18080 or 0.0.0.0:18080). Run vade up --help.".to_owned()
                })?;
                if addr.port() == 0 {
                    return Err("Set --bind port to 1..65535. Run vade up --help.".into());
                }
                options.bind = Some(addr);
            }
            "--disk-cache-dir" => {
                validate_disk_cache_path(value)?;
                options.disk_cache_dir = Some(value.to_owned());
            }
            "--disk-cache-bytes" => {
                let n = bytes(value, flag)?;
                if n == 0 {
                    return Err("Set --disk-cache-bytes to a positive whole number of bytes. Use --no-disk-cache to disable the disk prefix cache.".into());
                }
                options.disk_cache_bytes = Some(n);
            }
            "--decode-fuse" => options.decode_fuse = Some(match value {
                "on" => true,
                "off" => false,
                _ => return Err("Set --decode-fuse to off or on.".into()),
            }),
            _ => unreachable!(),
        }
        index += 2;
    }
    if options.is_inkling() && seen.contains("--fp8") {
        return Err("Inkling 2 Mini runs on one Spark; omit --fp8. Run vade up --help.".into());
    }
    if options.is_inkling() && seen.contains("--fp4-local") {
        return Err("Choose inkling or --fp4-local, not both. Run vade up --help.".into());
    }
    if options.model.as_deref() == Some(crate::pull::alias::QWEN_REPO) && options.mtp_explicit && options.mtp > 3 {
        return Err("Set --mtp to a whole number from 0 to 3.".into());
    }
    if options.no_disk_cache && (options.disk_cache_dir.is_some() || options.disk_cache_bytes.is_some()) {
        return Err("Cannot combine --no-disk-cache with --disk-cache-dir or --disk-cache-bytes. Run vade up --help.".into());
    }
    if options.cache_bytes == Some(0) && (options.disk_cache_dir.is_some() || options.disk_cache_bytes.is_some()) {
        return Err("Cannot combine --cache-bytes 0 with --disk-cache-dir or --disk-cache-bytes. Run vade up --help.".into());
    }
    if options.decode_fuse == Some(true) && (options.sparks.is_some_and(|n| n != 1) || !options.hosts.is_empty()) {
        return Err("--decode-fuse on runs on one Spark. Use --sparks 1, or leave it off.".into());
    }
    if !seen.contains("--mtp") {
        let max_mtp = if options.diagnostic_unqualified_text
            || options.is_inkling()
            || (!seen.contains("--fp4-local") && !seen.contains("--fp8") && options.model.is_none())
        { 7 } else { 3 };
        match std::env::var("MTP") {
            Ok(value) => {
                options.mtp = number(&value, "MTP", 0, max_mtp)?;
                options.mtp_explicit = true;
            },
            Err(std::env::VarError::NotPresent) => {}
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(format!("Set MTP to a whole number from 0 to {max_mtp}."));
            }
        }
    }
    if options.replica_sparks > 1 && !options.replicas {
        return Err("Use --pairs with --replicas N: N tensor-split pairs on 2N Sparks. Run vade up --help.".into());
    }
    if options.replicas {
        let machines = replicas.checked_mul(options.replica_sparks)
            .ok_or("Replica machine count overflows.")?;
        if seen.contains("--sparks") || !(options.hosts.is_empty() || options.hosts.len() == machines) {
            return Err("Use --replicas N without --sparks, and with either no --hosts or N Sparks in --hosts (2N with --pairs). Run vade up --help.".into());
        }
        if options.hosts.is_empty() { options.sparks = Some(machines); }
    }
    if options.is_inkling() {
        if !seen.contains("--context") { options.context = INKLING_CONTEXT; }
        if !options.output_explicit { options.output = inkling_output(options.context); }
    } else if options.fp4_local {
        if !seen.contains("--context") { options.context = FP4_CONTEXT; }
        if !options.output_explicit { options.output = fp4_output(options.context); }
    } else {
        if !seen.contains("--context") { options.context = 131072; }
        if !options.output_explicit { options.output = 65536; }
    }
    if seen.contains("--sparks") && seen.contains("--hosts") {
        return Err("Choose --sparks or --hosts. Run vade up --help.".into());
    }
    if !options.links.is_empty() && !listed {
        return Err("Use --link with the Sparks in --sparks, for the config that vade up writes. Run vade up --help.".into());
    }
    // From here a Spark list is --hosts in every way.
    if listed {
        seen.remove("--sparks");
        seen.insert("--hosts");
    }
    options.listed = listed;
    let mut index = 0;
    while listed && index < args.len() {
        match args[index] {
            "--sparks" | "--link" => index += 2,
            "--fp8" | "--fp8-legacy" | "--fp4-local" => index += 1,
            arg => { options.rest.push(arg.to_owned()); index += 1; }
        }
    }
    if !seen.contains("--output") {
        options.output = options.output.min(options.context);
    }
    if options.output > options.context {
        return Err(
            "Output must fit in context. Set --output no larger than --context, then run vade up."
                .into(),
        );
    }
    if options.memory_bytes == Some(0)
        || matches!((options.memory_bytes, options.cache_bytes), (Some(memory), Some(cache)) if cache >= memory)
    {
        return Err("Set --memory-bytes above zero and --cache-bytes below the memory budget. Use --cache-bytes 0 to disable retained prompt cache.".into());
    }
    options.supplied_options = seen.into_iter().map(str::to_owned).collect();
    Ok(options)
}

/// FP4-Local defaults for every route, on this Spark (Route A) and from a
/// controller (Route B): the published preset. The engine admits concurrent
/// chats from context and memory (eight at this context on one Spark); local
/// launch lowers a default or saved context only when the engine refuses it for memory.
/// `--context 262144` selects the
/// single-chat long-context preset explicitly.
pub const FP4_CONTEXT: u32 = 32768;
/// The FP4-Local default for a group of two or more Sparks, which `vade up` sets
/// once it knows the group. A start that does not fit falls to [`FP4_CONTEXT`].
pub const FP4_LONG_CONTEXT: u32 = 131072;
/// Default FP4-Local output limit: `min(FP4_OUTPUT, context)`.
pub const FP4_OUTPUT: u32 = 8192;
/// Default Inkling 2 Mini context window (512K with Paged KV).
pub const INKLING_CONTEXT: u32 = 524_288;
/// Default Inkling 2 Mini output limit (16K, capped at context).
pub const INKLING_OUTPUT: u32 = 16384;

/// Default FP4-Local output limit for a context.
pub fn fp4_output(context: u32) -> u32 { context.min(FP4_OUTPUT) }
/// Default Inkling 2 Mini output limit for a context.
pub fn inkling_output(context: u32) -> u32 { context.min(INKLING_OUTPUT) }

/// The error for a bad `--sparks`: the count form and the list form.
fn sparks_error() -> String {
    "Set --sparks to a positive whole number, or to distinct SSH destinations separated by commas, with no spaces. Run vade up --help.".into()
}

fn validate_disk_cache_path(value: &str) -> Result<(), String> {
    let rest = value.strip_prefix("~/").or_else(|| value.strip_prefix('/'));
    if rest.is_none_or(|r| r.is_empty() || r.split('/').any(|part| part == "..") || r.chars().any(char::is_control)) {
        return Err("Set --disk-cache-dir to an absolute or ~/ path without parent traversal. Run vade up --help.".into());
    }
    Ok(())
}

fn bytes(value: &str, flag: &str) -> Result<u64, String> {
    if !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit())
        && let Ok(bytes) = value.parse() {
        return Ok(bytes);
    }
    Err(format!(
        "Set {flag} to a whole number of bytes that fits in 64 bits."
    ))
}

/// `--link FROM:TO=ADDRESS`: FROM's own address on its cable to TO.
fn link(value: &str, links: &[(String, String, String)], command: &str) -> Result<(String, String, String), String> {
    let link = value.split_once('=').and_then(|(ends, address)| ends.split_once(':').map(|(from, to)| (from.to_owned(), to.to_owned(), address.to_owned())));
    match link {
        Some(link) if crate::config::fabric_address(&link.2) => {
            if links.iter().any(|(from, to, _)| *from == link.0 && *to == link.1) {
                return Err(format!("Give --link {}:{} only once: each cable end has one address.", link.0, link.1));
            }
            Ok(link)
        }
        _ => Err(format!("Give --link as FROM:TO=ADDRESS: FROM's own address on its cable to TO, without a port. Run vade {command} --help.")),
    }
}

/// The `vade up` options that take a value.
const UP_VALUE_OPTIONS: [&str; 13] = [
    "--sparks", "--hosts", "--replicas", "--mtp", "--context", "--output", "--chats",
    "--memory-bytes", "--cache-bytes", "--bind", "--disk-cache-dir", "--disk-cache-bytes", "--decode-fuse",
];

/// The word after the option at `index`. A word that starts with `-` is the
/// next option, unless a digit follows: `--context -5` then names the range.
fn option_value<'a>(args: &[&'a str], index: usize) -> Option<&'a str> {
    args.get(index + 1).copied().filter(|value| !value.starts_with('-') || value[1..].starts_with(|c: char| c.is_ascii_digit()))
}

/// The error for an unknown option of `vade <command>`. `--context=32768`
/// names the known option and asks for a space.
fn unknown_option(flag: &str, command: &str, known: &[&str]) -> String {
    match flag.split_once('=') {
        Some((name, _)) if known.contains(&name) => format!("Unknown option {} for vade {command}. Put a space between {name} and its value, not an equals sign. Run vade {command} --help.", crate::text::argument(flag)),
        _ => format!("Unknown option {} for vade {command}. Run vade {command} --help.", crate::text::argument(flag)),
    }
}

fn number(value: &str, flag: &str, min: u32, max: u32) -> Result<u32, String> {
    if !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit())
        && let Ok(value) = value.parse::<u32>()
        && (min..=max).contains(&value) {
        return Ok(value);
    }
    Err(format!("Set {flag} to a whole number from {min} to {max}."))
}

fn min_number(value: &str, flag: &str, min: u32) -> Result<u32, String> {
    if !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit())
        && let Ok(value) = value.parse::<u32>()
        && value >= min {
        return Ok(value);
    }
    Err(format!("Set {flag} to a whole number of at least {min}."))
}

/// A machine or host name: 1 to 64 ASCII letters, digits, `-` or `_`, starting
/// with a letter or digit.
pub fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_'))
}

pub use crate::help::help;

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;
    fn cli(args: &str) -> Result<Cli, String> { parse(&args.split_whitespace().map(str::to_owned).collect::<Vec<_>>()) }
    #[test]
    fn chat_and_sessions_options_pass_to_the_harness() {
        let chat = |args: &[&str]| parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>()).map(|c| c.command);
        assert_eq!(chat(&["chat"]).unwrap(), Command::Chat(ChatOptions::default()));
        assert_eq!(chat(&["chat", "--resume", "last", "--name", "bug fix", "--dry-run"]).unwrap(),
            Command::Chat(ChatOptions { resume: Some("last".into()), name: Some("bug fix".into()) }));
        for bad in [&["chat", "--resume"][..], &["chat", "--resume", "a", "--resume", "b"], &["chat", "--name", "--resume"], &["chat", "-x"]] {
            assert!(chat(bad).unwrap_err().contains("vade chat --help"), "{bad:?}");
        }
        // A prompt or --json says what vade chat does instead (snapshots pin the exit code).
        assert_eq!(chat(&["chat", "fix", "tests"]).unwrap_err(), "vade chat takes no prompt on the command line (\"fix tests\"). Run vade chat, then type the message and press Enter.");
        assert_eq!(chat(&["chat", "--name", "login", "bug"]).unwrap_err(), "Unexpected word \"bug\" after the --name value. Put a value with spaces in quotes, for example vade chat --name \"login bug\".");
        assert_eq!(chat(&["chat", "--json"]).unwrap_err(), "vade chat has no --json output; it is an interactive chat. Run vade status --json for the engine state as JSON.");
        let sessions = |args: &str| match cli(args).map(|c| c.command) { Ok(Command::Sessions(a)) => Ok(a.join(" ")), Ok(other) => panic!("{other:?}"), Err(e) => Err(e) };
        assert_eq!(sessions("sessions").unwrap(), "list");
        assert_eq!(sessions("sessions list --all").unwrap(), "list --all");
        assert_eq!(sessions("sessions --all").unwrap(), "list --all");
        assert_eq!(sessions("sessions export last").unwrap(), "export last --markdown");
        assert_eq!(sessions("sessions export session-1-2-3 --markdown").unwrap(), "export session-1-2-3 --markdown");
        assert_eq!(sessions("sessions delete session-1-2-3").unwrap(), "delete session-1-2-3");
        for bad in ["sessions delete", "sessions export", "sessions export --markdown", "sessions delete a b", "sessions export a --json", "sessions rm a", "sessions delete -x", "sessions export -x"] {
            assert!(sessions(bad).unwrap_err().contains("vade sessions --help"), "{bad}");
        }
        assert_eq!(cli("sessions --help").unwrap().command, Command::Help(Some("sessions".into())));
        // -h as a target is vade sessions help, never the harness help.
        for help in ["sessions delete -h", "sessions export -h"] { assert_eq!(cli(help).unwrap().command, Command::Help(Some("sessions".into())), "{help}"); }
    }
    #[test]
    fn fp4_profile_uses_current_limits_and_keeps_explicit_choices() {
        let Command::Up(options) = cli("up").unwrap().command else { panic!() };
        assert!(options.fp4_local);
        assert_eq!((options.context, options.output, options.chats), (32768, 8192, None));
        assert_eq!(options.mtp_explicit, std::env::var_os("MTP").is_some());
        // Controller defaults equal Route A on any Spark or chat count; chats never pick a context.
        for args in [
            "up --fp4-local", "up --fp4-local --chats 1", "up --fp4-local --chats 4", "up --fp4-local --chats 8",
            "up --fp4-local --chats 32", "up --fp4-local --sparks 1", "up --fp4-local --sparks 2",
            "up --fp4-local --sparks 2 --chats 1", "up --fp4-local --hosts a,b", "up --fp4-local --hosts a --chats 1",
            "up --replicas 2", "up --replicas 2 --chats 1", "up --replicas 2 --hosts a,b --chats 8", "up --replicas 2 --pairs", "up --pairs --replicas 2 --hosts a,b,c,d",
        ] {
            let Command::Up(options) = cli(args).unwrap().command else { panic!("{args}") };
            assert_eq!((options.context, options.output), (FP4_CONTEXT, fp4_output(FP4_CONTEXT)), "{args}");
        }
        let Command::Up(options) = cli("up --fp4-local --chats 4 --context 8192").unwrap().command else { panic!() };
        assert_eq!((options.context, options.output), (8192, 8192));
        // The single-chat long-context preset stays behind an explicit --context.
        let Command::Up(options) = cli("up --context 262144").unwrap().command else { panic!() };
        assert_eq!((options.context, options.output, options.chats), (262144, 8192, None));
        let Command::Up(options) = cli("up --fp4-local --chats 1 --context 262144 --output 16384").unwrap().command else { panic!() };
        assert_eq!((options.context, options.output), (262144, 16384));
        let Command::Up(options) = cli("up --fp4-local --context 131072 --chats 8").unwrap().command else { panic!() };
        assert_eq!((options.context, options.output), (131072, 8192));
        let Command::Up(options) = cli("up --output 16384").unwrap().command else { panic!() };
        assert_eq!((options.context, options.output), (32768, 16384));
        assert!(help(Some("up")).contains("Default: Qwen3.8 Flash-Next FP4-Local"));
        let Command::Up(options) = cli("up --mtp 0 --context 8192 --output 4096 --fp4-local").unwrap().command else { panic!() };
        assert!(options.mtp_explicit);
        assert_eq!((options.mtp, options.context, options.output), (0, 8192, 4096));
        let Command::Up(options) = cli("up --fp4-local --context 4096").unwrap().command else { panic!() };
        assert_eq!((options.context, options.output), (4096, 4096));
        assert!(cli("up --fp4-local --fp4-local").is_err());
        // Rule 17: no arbitrary CLI context/output ceilings; real hardware admission decides.
        let Command::Up(wide) = cli("up --fp4-local --context 524288 --output 32768").unwrap().command else { panic!() };
        assert_eq!((wide.context, wide.output), (524_288, 32768));
        let Command::Up(wide_fp8) = cli("up --fp8 --context 524288 --output 131072").unwrap().command else { panic!() };
        assert_eq!((wide_fp8.context, wide_fp8.output), (524_288, 131_072));
    }
    #[test]
    fn a_spark_list_keeps_its_links_and_the_other_options() {
        let up = |args: &str| match cli(args).map(|cli| cli.command) { Ok(Command::Up(options)) => Ok(options), Ok(other) => panic!("{other:?}"), Err(error) => Err(error) };
        let options = up("up --context 8192 --sparks you@spark-a,spark-b --fp8 --link spark-b:you@spark-a=192.0.2.2 --chats 4").unwrap();
        assert!(options.listed);
        assert_eq!(options.links, [("spark-b".to_owned(), "you@spark-a".to_owned(), "192.0.2.2".to_owned())]);
        assert_eq!(options.rest, ["--context", "8192", "--chats", "4"]);
        assert!(!up("up --sparks 2").unwrap().listed);
        assert!(!up("up --hosts a,b").unwrap().listed);
        for (args, error) in [("up --sparks 2 --link a:b=192.0.2.1", "Use --link with the Sparks in --sparks"), ("up --hosts a,b --link a:b=192.0.2.1", "Use --link with the Sparks in --sparks"),
            ("up --sparks a,b --link a:b", "Give --link as FROM:TO=ADDRESS"), ("up --sparks a,b --link a:b=192.0.2.1 --link a:b=192.0.2.9", "Give --link a:b only once"),
            ("up --sparks a,b --link", "Give --link a value"), ("launch --sparks a,b", "vade launch starts this Spark")] {
            assert!(up(args).unwrap_err().contains(error), "{args}: {:?}", up(args));
        }
    }

    #[test]
    fn parser_records_every_supplied_launch_option() {
        for args in [
            "--sparks 2", "--hosts a,b", "--mtp 2", "--fp4-local", "--fp8",
            "--context 131072", "--output 16384", "--chats 1", "--fast-prompts",
            "--memory-bytes 128000000000", "--cache-bytes 0",
        ] {
            let Command::Up(options) = cli(&format!("up {args}")).unwrap().command else { panic!() };
            assert_eq!(options.supplied_options, [args.split_whitespace().next().unwrap().to_owned()].into(), "{args}");
        }
        let Command::Up(options) = cli("up --dry-run").unwrap().command else { panic!() };
        assert!(options.supplied_options.is_empty());
    }
    #[test]
    fn defaults_and_global_flag() {
        assert_eq!(cli("").unwrap().command, Command::Menu);
        let Command::Up(fp8) = cli("up --fp8").unwrap().command else { panic!() };
        assert!(!fp8.fp4_local); assert_eq!((fp8.context, fp8.output), (131072, 65536));
        // The older spelling selects the same FP8 profile and records the same option.
        for args in ["up --fp8-legacy", "launch --fp8-legacy"] {
            let (Command::Up(older) | Command::Launch(older)) = cli(args).unwrap().command else { panic!("{args}") };
            assert_eq!(older, fp8, "{args}");
        }
        let Command::Launch(local) = cli("launch --context 65536 --output 16384").unwrap().command else { panic!() };
        assert!(local.fp4_local && local.supplied_options.contains("--context")); assert_eq!((local.context, local.output), (65536, 16384));
        assert_eq!(cli("--dry-run up --mtp 3 --dry-run").unwrap(), Cli {
            command: Command::Up(UpOptions { supplied_options: ["--mtp".into()].into(), mtp: 3, mtp_explicit: true, ..UpOptions::default() }), dry_run: true, json: false
        });
        assert!(matches!(cli("chat -h").unwrap().command, Command::Help(Some(_))));
        assert!(cli("launch --context").unwrap_err().contains("vade launch --help"));
        assert_eq!(cli("doctor --fabric --json --dry-run").unwrap(), Cli { command: Command::DoctorFabric { ring: false }, dry_run: true, json: true });
        assert_eq!(cli("doctor --fabric --ring --dry-run").unwrap(), Cli { command: Command::DoctorFabric { ring: true }, dry_run: true, json: false });
        for bad in ["doctor --fabric --ring --ring", "doctor --fabric extra", "doctor --ring --fabric"] {
            assert!(cli(bad).is_err(), "{bad}");
        }
        assert!(cli("doctor --fabric extra").unwrap_err().contains("vade doctor --fabric [--ring] [--json] [--dry-run]"));
        assert!(cli("doctor --sparks 2").unwrap_err().contains("Unexpected argument"));
    }
    #[test]
    fn fp8_default_depth_is_two_without_mtp() {
        // MTP overrides the default depth, so check the default in a child process without MTP.
        const CASE: &str = "VADE_TEST_FP8_DEPTH";
        if std::env::var_os(CASE).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "cli::tests::fp8_default_depth_is_two_without_mtp", "--nocapture"])
                .env(CASE, "1").env_remove("MTP").output().unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(output.status.success() && stdout.contains("1 passed"), "{stdout}{}", String::from_utf8_lossy(&output.stderr));
            return;
        }
        for args in ["up --fp8", "up --fp8-legacy"] {
            let Command::Up(fp8) = cli(args).unwrap().command else { panic!("{args}") };
            assert_eq!((fp8.mtp, fp8.mtp_explicit), (2, false), "{args}");
        }
    }
    #[test]
    fn deep_mtp_is_only_for_the_text_service() {
        for depth in 4..=7 {
            for args in [format!("up --mtp {depth} --diagnostic-unqualified-text"),format!("up --diagnostic-unqualified-text --mtp {depth}"), format!("up inkling --mtp {depth}"), format!("up --mtp {depth}")] {
                let Command::Up(options)=cli(&args).unwrap().command else {panic!("{args}")};
                assert_eq!(options.mtp,depth);assert!(options.mtp_explicit);
            }
            for model in ["--fp4-local", "--fp8", "qwen"] {assert!(cli(&format!("up {model} --mtp {depth}")).is_err());}
        }
        assert!(cli("up --diagnostic-unqualified-text --mtp 8").is_err());
        const CASE:&str="VADE_TEST_DEEP_MTP";
        if std::env::var_os(CASE).is_none() {
            let output=std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact","cli::tests::deep_mtp_is_only_for_the_text_service","--nocapture"])
                .env(CASE,"1").env("MTP","7").output().unwrap();
            assert!(output.status.success(),"{}{}",String::from_utf8_lossy(&output.stdout),String::from_utf8_lossy(&output.stderr));
        } else {
            let Command::Up(options)=cli("up --diagnostic-unqualified-text").unwrap().command else {panic!()};
            assert_eq!(options.mtp,7);assert!(cli("up --fp8").is_err());assert!(cli("up --fp4-local").is_err());
        }
    }
    #[test]
    fn bad_options_are_actionable() {
        for args in [
            "up --hosts a,a", "up --hosts a,", "up --sparks 2 --hosts a,b", "up --fp4-local --mtp 4",
            "up --context 4095", "up --output 0", "up --mtp", "up --mtp 1 --mtp 2", "down --force", "unknown",
            "up --context 4096 --output 8192", "up --fp4-local --fp8", "up --fp4-local --fp8-legacy", "up --fp8 --fp8-legacy",
            "up --replicas 1", "up --replicas 2 --sparks 2", "up --replicas 3 --hosts a,b", "up --pairs", "up --replicas 2 --pairs --pairs", "up --replicas 2 --pairs --hosts a,b",
            "up --bind not-an-addr", "up --bind 0.0.0.0:0", "up --disk-cache-dir relative/path", "up --disk-cache-dir /tmp/../escape",
            "up --disk-cache-bytes 0", "up --no-disk-cache --disk-cache-dir /tmp/cache", "up --no-disk-cache --disk-cache-bytes 1024",
            "up --cache-bytes 0 --disk-cache-dir /tmp/cache",
        ] { assert!(cli(args).is_err(), "{args}"); }
        // Rule 17: replicas, sparks, chats, bind and disk cache options scale without arbitrary caps.
        for ok in [
            "up inkling --replicas 2", "up inkling --replicas 4 --hosts a,b,c,d", "up --fp8 --replicas 2 --pairs",
            "up --replicas 17", "up --replicas 13 --pairs", "up --chats 64",
            "up inkling --bind 0.0.0.0:18080 --disk-cache-dir /tmp/vade-prefix --disk-cache-bytes 1073741824",
            "up inkling --no-disk-cache",
        ] { assert!(cli(ok).is_ok(), "{ok}"); }
        assert_eq!(cli("up --hosts a,b --chats 3 --fast-prompts").unwrap().command, Command::Up(UpOptions {
            supplied_options: ["--hosts".into(), "--chats".into(), "--fast-prompts".into()].into(),
            hosts: vec!["a".into(), "b".into()], chats: Some(3), fast_prompts: true, context: 32768, ..UpOptions::default()
        }));
    }
    #[test]
    fn setup_takes_replace_local_once() {
        let Command::Setup(options) = cli("setup --yes --replace-local --hosts a").unwrap().command else { panic!() };
        assert!(options.replace_local && options.yes);
        assert!(cli("setup --replace-local --replace-local").unwrap_err().contains("only once"));
    }
    #[test]
    fn sparks_takes_a_count_or_the_sparks_and_hosts_means_the_same_as_in_setup() {
        let up = |args: &str| match cli(args).unwrap().command { Command::Up(options) => options, other => panic!("{other:?}") };
        let list = up("up --sparks spark-a,user@spark-b,192.0.2.7");
        assert_eq!((list.sparks, list.hosts.join(" ")), (None, "spark-a user@spark-b 192.0.2.7".to_owned()));
        // A list is --hosts in every way, also in what the start records.
        assert_eq!(list.supplied_options, ["--hosts".to_owned()].into());
        assert_eq!(UpOptions { listed: false, ..list }, up("up --hosts spark-a,user@spark-b,192.0.2.7"));
        assert_eq!(up("up --sparks spark-a").hosts, ["spark-a"]);
        assert_eq!((up("up --sparks 24").sparks, up("up --sparks 64").sparks, up("up --sparks 1").sparks), (Some(24), Some(64), Some(1)));
        let setup = |args: &str| match cli(args).unwrap().command { Command::Setup(options) => options.hosts, other => panic!("{other:?}") };
        assert_eq!(up("up --hosts user@spark-a,spark-b.lan").hosts, setup("setup --hosts user@spark-a,spark-b.lan"));
        let sparks = "Set --sparks to a positive whole number, or to distinct SSH destinations separated by commas, with no spaces. Run vade up --help.";
        for bad in ["up --sparks 0", "up --sparks a,a", "up --sparks a,", "up --sparks ,a", "up --sparks -5", "up --sparks a;b", "launch --sparks 0"] {
            assert_eq!(cli(bad).unwrap_err(), sparks.replace("vade up", if bad.starts_with("launch") { "vade launch" } else { "vade up" }), "{bad}");
        }
        let hosts = "Give --hosts distinct SSH destinations (what you type after ssh), separated by commas, with no spaces.";
        for bad in ["up --hosts a,a", "up --hosts a,", "setup --hosts a,a", "setup --hosts a,"] { assert_eq!(cli(bad).unwrap_err(), hosts, "{bad}"); }
        assert_eq!(cli("up --sparks a,b --hosts a,b").unwrap_err(), "Choose --sparks or --hosts. Run vade up --help.");
        assert_eq!(cli("up --hosts a,b --sparks 2").unwrap_err(), "Choose --sparks or --hosts. Run vade up --help.");
        assert!(cli("up --replicas 2 --sparks a,b").unwrap_err().contains("Use --replicas N"));
    }
    #[test]
    fn argument_errors_name_the_command_to_run_instead() {
        for (args, expected) in [
            ("help up down", "Use vade help, or vade help <command>."),
            ("help nonsense", "No help topic is named \"nonsense\". Run vade --help to see the commands."),
            ("--json verify", "vade verify takes no --dry-run or --json. Run vade verify --help to see its options."),
            ("--dry-run bench", "vade bench takes no --dry-run or --json. Run vade bench --help to see its options."),
            // What is left of `vade --api-key-file KEY bench` once the key file is taken out.
            ("bench", "vade bench reads only the options after its name. Run vade bench --help to see them."),
            ("--bogus up", "Unknown option \"--bogus\". Put options after the command name. Run vade --help."),
            ("version --json now", "Use vade --version without additional arguments."),
            ("run", "Use vade launch for Flash-Next FP4-Local, or vade run qwen3-8b for the separate official Qwen3-8B BF16 catalog."),
            ("run qwen3-8b --release r --engine-source e", "Choose --release or source checkouts."),
            ("up --memory-bytes 0", "Set --memory-bytes above zero and --cache-bytes below the memory budget. Use --cache-bytes 0 to disable retained prompt cache."),
            ("up --memory-bytes 100 --cache-bytes 100", "Set --memory-bytes above zero and --cache-bytes below the memory budget. Use --cache-bytes 0 to disable retained prompt cache."),
            ("up --memory-bytes 1e9", "Set --memory-bytes to a whole number of bytes that fits in 64 bits."),
            ("up --cache-bytes 18446744073709551616", "Set --cache-bytes to a whole number of bytes that fits in 64 bits."),
        ] {
            assert_eq!(cli(args).unwrap_err(), expected, "{args}");
        }
        assert_eq!(cli("help up").unwrap().command, Command::Help(Some("up".into())));
        for args in ["--json --help", "help --json", "--help --json"] {
            assert_eq!(cli(args).unwrap().command, Command::Help(None), "{args}: help ignores --json");
        }
        for topic in ["bench", "verify", "test"] {
            assert_eq!(cli(&format!("--dry-run help {topic}")).unwrap().command, Command::Help(Some(topic.into())));
        }
        assert!(cli("--json doctor").unwrap().json, "a leading --json reads as a trailing one");
        assert_eq!(cli("-V").unwrap().command, Command::Version);
        let Command::Run(options) = cli("run qwen3-8b --release r --port 9000").unwrap().command else { panic!() };
        assert_eq!((options.release.as_deref(), options.port), (Some(std::path::Path::new("r")), 9000));
        let Command::Up(options) = cli("up --memory-bytes 100 --cache-bytes 0").unwrap().command else { panic!() };
        assert_eq!((options.memory_bytes, options.cache_bytes), (Some(100), Some(0)));
    }
    #[test]
    fn negative_numbers_and_equals_signs_get_the_error_that_names_the_fix() {
        // A negative number is a bad value, not a missing one; the next option is still missing.
        assert_eq!(cli("up --context -5").unwrap_err(), "Set --context to a whole number of at least 4096.");
        assert_eq!(cli("up --mtp -1").unwrap_err(), "Set --mtp to a whole number from 0 to 7.");
        let Command::Up(fused) = cli("up --decode-fuse on").unwrap().command else { panic!() };
        assert_eq!(fused.decode_fuse, Some(true));
        assert_eq!(cli("up --decode-fuse 1").unwrap_err(), "Set --decode-fuse to off or on.");
        for args in ["up --sparks 2 --decode-fuse on", "up --hosts a,b --decode-fuse on"] {
            assert_eq!(cli(args).unwrap_err(), "--decode-fuse on runs on one Spark. Use --sparks 1, or leave it off.");
        }
        assert_eq!(cli("run qwen3-8b --port -1").unwrap_err(), "Set --port to a whole number from 1024 to 65535.");
        assert_eq!(cli("up --context --fp4-local").unwrap_err(), "Give --context a value. Run vade up --help.");
        assert_eq!(cli("up --context -").unwrap_err(), "Give --context a value. Run vade up --help.");
        assert_eq!(cli("up --context=32768").unwrap_err(),
            "Unknown option \"--context=32768\" for vade up. Put a space between --context and its value, not an equals sign. Run vade up --help.");
        assert!(cli("run qwen3-8b --port=9000").unwrap_err().contains("Put a space between --port and its value"));
        assert_eq!(cli("up --nope=1").unwrap_err(), "Unknown option \"--nope=1\" for vade up. Run vade up --help.");
    }
    #[test]
    fn inkling_alias_and_connect_parse_cleanly() {
        for cmd in ["up inkling", "up INKLING", "up inkling-2-mini", "launch inkling"] {
            let (Command::Up(o) | Command::Launch(o)) = cli(cmd).unwrap().command else { panic!("{cmd}") };
            assert_eq!(o.model.as_deref(), Some(crate::pull::alias::INKLING_REPO), "{cmd}");
            assert!(o.is_inkling());
            assert_eq!((o.context, o.output), (INKLING_CONTEXT, INKLING_OUTPUT), "{cmd}");
        }
        let Command::Up(o) = cli("up inkling --context 1048576 --output 65536 --mtp 7").unwrap().command else { panic!() };
        assert_eq!((o.context, o.output, o.mtp, o.mtp_explicit), (1_048_576, 65536, 7, true));
        assert!(cli("up inkling --mtp 8").is_err());
        let Command::Up(qwen_wide) = cli("up qwen --context 1048576").unwrap().command else { panic!() };
        assert_eq!(qwen_wide.context, 1_048_576);
        let Command::Up(fp4_wide) = cli("up --fp4-local --context 1048576").unwrap().command else { panic!() };
        assert_eq!(fp4_wide.context, 1_048_576);
        let Command::Up(qwen) = cli("up qwen").unwrap().command else { panic!() };
        assert_eq!(qwen.model.as_deref(), Some(crate::pull::alias::QWEN_REPO));
        assert_eq!((qwen.context, qwen.output), (FP4_CONTEXT, FP4_OUTPUT));
        assert!(cli("up inkling qwen").is_err());
        assert!(cli("up unknown-model").unwrap_err().contains("Unknown model"));
        assert_eq!(cli("connect --help").unwrap().command, Command::Help(Some("connect".into())));
        let parsed = cli("connect codex --dry-run -- --help").unwrap();
        assert!(parsed.dry_run);
        let Command::Connect(args) = parsed.command else { panic!() };
        assert_eq!(args, vec!["codex", "--", "--help"]);
    }
}
