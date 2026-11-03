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

/// Options for `vade up` and `vade launch`, after defaults are applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpOptions {
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
    /// The engine's decode_fuse option: Some(true) runs one Spark's decode steps as a chain.
    pub decode_fuse: Option<bool>,
}

impl Default for UpOptions {
    fn default() -> Self {
        Self {
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
            decode_fuse: None,
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
    if args[1..].iter().any(|arg| matches!(*arg, "--help" | "-h")) {
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
    } else if matches!(command, "pull" | "models") {
        let rest = args[1..].iter().map(|arg| (*arg).to_owned()).collect();
        if command == "pull" { Command::Pull(rest) } else { Command::Models(rest) }
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
            | "pull" | "models" | "setup"
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
    while index < args.len() {
        // --fp8-legacy is the older spelling of --fp8; both name the same profile.
        let flag = if args[index] == "--fp8-legacy" { "--fp8" } else { args[index] };
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
            let value = option_value(&args, index).ok_or("Give --link a value. Run vade up --help.")?;
            let link = link(value, &options.links, "up")?;
            options.links.push(link);
            index += 2;
            continue;
        }
        if !UP_VALUE_OPTIONS.contains(&flag) {
            return Err(unknown_option(flag, "up", &UP_VALUE_OPTIONS));
        }
        let value = option_value(&args, index)
            .ok_or_else(|| format!("Give {flag} a value. Run vade up --help."))?;
        match flag {
            // A count, or the Sparks themselves, which is --hosts by another name.
            "--sparks" if value.bytes().all(|byte| byte.is_ascii_digit()) => {
                options.sparks = Some(number(value, flag, 1, crate::config::MAX_FP4_SPARKS as u32).map_err(|_| sparks_error())? as usize);
            }
            "--sparks" => {
                options.hosts = ssh_targets(value).ok_or_else(sparks_error)?;
                listed = true;
            }
            "--replicas" => {
                options.replicas = true;
                replicas = number(value, flag, 2, crate::config::MAX_REPLICAS as u32)? as usize;
            }
            "--hosts" => options.hosts = ssh_targets(value).ok_or(HOSTS)?,
            "--mtp" => {
                options.mtp = number(value, flag, 0, 3)?;
                options.mtp_explicit = true;
            },
            "--context" => options.context = number(value, flag, 4096, 262144)?,
            "--output" => {
                options.output = number(value, flag, 1024, 262144)?;
                options.output_explicit = true;
            }
            "--chats" => options.chats = Some(number(value, flag, 1, 32)? as usize),
            "--memory-bytes" => options.memory_bytes = Some(bytes(value, flag)?),
            "--cache-bytes" => options.cache_bytes = Some(bytes(value, flag)?),
            "--decode-fuse" => options.decode_fuse = Some(match value {
                "on" => true,
                "off" => false,
                _ => return Err("Set --decode-fuse to off or on.".into()),
            }),
            _ => unreachable!(),
        }
        index += 2;
    }
    if options.decode_fuse == Some(true) && (options.sparks.is_some_and(|n| n != 1) || !options.hosts.is_empty()) {
        return Err("--decode-fuse on runs on one Spark. Use --sparks 1, or leave it off.".into());
    }
    if !seen.contains("--mtp") {
        match std::env::var("MTP") {
            Ok(value) => {
                options.mtp = number(&value, "MTP", 0, 3)?;
                options.mtp_explicit = true;
            },
            Err(std::env::VarError::NotPresent) => {}
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err("Set MTP to a whole number from 0 to 3.".into());
            }
        }
    }
    if options.replica_sparks > 1 && !options.replicas {
        return Err("Use --pairs with --replicas N: N tensor-split pairs on 2N Sparks. Run vade up --help.".into());
    }
    if options.replicas {
        let machines = replicas * options.replica_sparks;
        if machines > crate::config::MAX_FP4_SPARKS {
            return Err(format!("Use --replicas 2..{} with --pairs: each pair takes two of the {} Sparks FP4-Local supports. Run vade up --help.",
                crate::config::MAX_FP4_SPARKS / options.replica_sparks, crate::config::MAX_FP4_SPARKS));
        }
        if !options.fp4_local || seen.contains("--sparks") || !(options.hosts.is_empty() || options.hosts.len() == machines) {
            return Err("Use --replicas N with FP4-Local alone, or with N Sparks in --hosts (2N with --pairs). Run vade up --help.".into());
        }
        if options.hosts.is_empty() { options.sparks = Some(machines); }
    }
    if options.fp4_local {
        if !seen.contains("--context") { options.context = FP4_CONTEXT; }
        if !options.output_explicit { options.output = fp4_output(options.context); }
        if options.output > FP4_OUTPUT_LIMIT {
            return Err("FP4-Local output is limited to 16384 tokens. Set --output 1024..16384.".into());
        }
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
/// Largest explicit FP4-Local output limit.
pub const FP4_OUTPUT_LIMIT: u32 = 16384;

/// Default FP4-Local output limit for a context.
pub fn fp4_output(context: u32) -> u32 { context.min(FP4_OUTPUT) }

/// The error for a bad `--sparks`: the range of the help, and the list form.
fn sparks_error() -> String {
    format!("Set --sparks to a whole number from 1 to {}, or to distinct SSH destinations separated by commas, with no spaces. Run vade up --help.", crate::config::MAX_FP4_SPARKS)
}

fn bytes(value: &str, flag: &str) -> Result<u64, String> {
    if !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit()) {
        if let Ok(bytes) = value.parse() {
            return Ok(bytes);
        }
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
const UP_VALUE_OPTIONS: [&str; 10] = ["--sparks", "--hosts", "--replicas", "--mtp", "--context", "--output", "--chats", "--memory-bytes", "--cache-bytes", "--decode-fuse"];

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
    if !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit()) {
        if let Ok(value) = value.parse::<u32>() {
            if (min..=max).contains(&value) {
                return Ok(value);
            }
        }
    }
    Err(format!("Set {flag} to a whole number from {min} to {max}."))
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
        assert!(cli("up --fp4-local --output 32768").unwrap_err().contains("limited to 16384"));
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
    fn bad_options_are_actionable() {
        for args in [
            "up --hosts a,a", "up --hosts a,", "up --sparks 2 --hosts a,b", "up --mtp 4",
            "up --context 4095", "up --output 262145", "up --mtp", "up --mtp 1 --mtp 2", "down --force", "unknown",
            "up --output 1023", "up --context 4096 --output 8192", "up --fp4-local --fp8", "up --fp4-local --fp8-legacy", "up --fp8 --fp8-legacy",
            "up --replicas 1", "up --replicas 17", "up --replicas 2 --sparks 2", "up --replicas 3 --hosts a,b", "up --fp8 --replicas 2", "up --pairs", "up --replicas 2 --pairs --pairs", "up --replicas 2 --pairs --hosts a,b", "up --replicas 13 --pairs",
        ] { assert!(cli(args).is_err(), "{args}"); }
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
        assert_eq!((up("up --sparks 24").sparks, up("up --sparks 1").sparks), (Some(24), Some(1)));
        let setup = |args: &str| match cli(args).unwrap().command { Command::Setup(options) => options.hosts, other => panic!("{other:?}") };
        assert_eq!(up("up --hosts user@spark-a,spark-b.lan").hosts, setup("setup --hosts user@spark-a,spark-b.lan"));
        let sparks = "Set --sparks to a whole number from 1 to 24, or to distinct SSH destinations separated by commas, with no spaces. Run vade up --help.";
        for bad in ["up --sparks 0", "up --sparks 25", "up --sparks 64", "up --sparks a,a", "up --sparks a,", "up --sparks ,a", "up --sparks -5", "up --sparks a;b", "launch --sparks 25"] {
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
        assert_eq!(cli("up --context -5").unwrap_err(), "Set --context to a whole number from 4096 to 262144.");
        assert_eq!(cli("up --mtp -1").unwrap_err(), "Set --mtp to a whole number from 0 to 3.");
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
}
