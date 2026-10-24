#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cli {
    pub command: Command,
    pub dry_run: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Menu,
    Up(UpOptions),
    Run(RunOptions),
    Down,
    Status,
    StatusJson,
    Logs,
    Chat,
    Doctor,
    Install,
    Detect,
    Launch(UpOptions),
    Update,
    Help(Option<String>),
    Version,
    Worker(Vec<String>),
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpOptions {
    /// Launch flags supplied on the command line, independent of defaults or environment.
    pub supplied_options: std::collections::BTreeSet<String>,
    pub sparks: Option<usize>,
    pub hosts: Vec<String>,
    /// One independent one-Spark server per selected machine, behind a local gateway.
    pub replicas: bool,
    pub mtp: u32,
    pub mtp_explicit: bool,
    pub fp4_local: bool,
    pub context: u32,
    pub output: u32,
    pub output_explicit: bool,
    pub chats: Option<usize>,
    pub fast_prompts: bool,
    pub memory_bytes: Option<u64>,
    pub cache_bytes: Option<u64>,
}

impl Default for UpOptions {
    fn default() -> Self {
        Self {
            supplied_options: Default::default(),
            sparks: None,
            hosts: Vec::new(),
            replicas: false,
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
        }
    }
}

pub fn parse(args: &[String]) -> Result<Cli, String> {
    if args.first().is_some_and(|arg| arg == "__worker") {
        return Ok(Cli {
            command: Command::Worker(args[1..].to_vec()),
            dry_run: false,
        });
    }
    let dry_run = args.iter().any(|arg| arg == "--dry-run");
    let args: Vec<&str> = args
        .iter()
        .filter(|arg| arg.as_str() != "--dry-run")
        .map(String::as_str)
        .collect();
    let Some(command) = args.first().copied() else {
        return Ok(Cli {
            command: Command::Menu,
            dry_run,
        });
    };
    if matches!(command, "version" | "--version" | "-V") {
        if args.len() != 1 { return Err("Use vade --version without additional arguments.".into()); }
        return Ok(Cli { command: Command::Version, dry_run });
    }
    if matches!(command, "--help" | "-h" | "help") {
        if args.len() > 2 || (args.len() == 2 && !known_command(args[1])) {
            return Err("Use vade help, or vade help <command>.".into());
        }
        return Ok(Cli {
            command: Command::Help(args.get(1).map(|s| (*s).into())),
            dry_run,
        });
    }
    if !known_command(command) {
        return Err(format!("Unknown command {}. Run vade --help to see the commands.", crate::text::argument(command)));
    }
    if args[1..].iter().any(|arg| matches!(*arg, "--help" | "-h")) {
        return Ok(Cli {
            command: Command::Help(Some(command.into())),
            dry_run,
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
                return Err(format!("Unknown option {} for vade run. Run vade run --help.", crate::text::argument(flag)));
            }
            if !seen.insert(flag) {
                return Err(format!("Use {flag} only once. Run vade run --help."));
            }
            let value = args
                .get(index + 1)
                .filter(|v| !v.starts_with('-'))
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
    } else if command == "status" && args.get(1) == Some(&"--json") {
        if args.len() != 2 { return Err("Use vade status --json [--dry-run]. No other status options are supported.".into()); }
        Command::StatusJson
    } else if matches!(command, "up" | "launch") {
        let options = parse_up(&args[1..]).map_err(|error| error.replace("vade up", &format!("vade {command}")))?;
        if command == "launch" { Command::Launch(options) } else { Command::Up(options) }
    } else {
        if args.len() != 1 {
            return Err(format!(
                "Unexpected argument {} for vade {command}. This command takes no options except --dry-run. Run vade {command} --help.", crate::text::argument(args[1])
            ));
        }
        match command {
            "down" => Command::Down,
            "status" => Command::Status,
            "logs" => Command::Logs,
            "chat" => Command::Chat,
            "doctor" => Command::Doctor,
            "install" => Command::Install,
            "detect" => Command::Detect,
            "update" => Command::Update,
            _ => unreachable!(),
        }
    };
    Ok(Cli {
        command: parsed,
        dry_run,
    })
}

fn known_command(command: &str) -> bool {
    matches!(
        command,
        "run" | "up" | "down" | "status" | "logs" | "chat" | "doctor" | "install" | "update" | "version" | "detect" | "launch"
    )
}

fn parse_up(args: &[&str]) -> Result<UpOptions, String> {
    let mut options = UpOptions::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut replicas = 0;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index];
        if !seen.insert(flag) {
            return Err(format!("Use {flag} only once. Run vade up --help."));
        }
        if matches!(flag, "--fp4-local" | "--fp8-legacy") {
            if seen.contains("--fp4-local") && seen.contains("--fp8-legacy") {
                return Err("Choose --fp4-local or --fp8-legacy. Run vade up --help.".into());
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
        if !matches!(
            flag,
            "--sparks" | "--hosts" | "--replicas" | "--mtp" | "--context" | "--output" | "--chats" | "--memory-bytes" | "--cache-bytes"
        ) {
            return Err(format!("Unknown option {} for vade up. Run vade up --help.", crate::text::argument(flag)));
        }
        let value = args
            .get(index + 1)
            .copied()
            .filter(|value| !value.starts_with('-'))
            .ok_or_else(|| format!("Give {flag} a value. Run vade up --help."))?;
        match flag {
            "--sparks" => options.sparks = Some(number(value, flag, 1, 64)? as usize),
            "--replicas" => {
                options.replicas = true;
                replicas = number(value, flag, 2, crate::config::MAX_REPLICAS as u32)? as usize;
            }
            "--hosts" => {
                options.hosts = value.split(',').map(str::to_owned).collect();
                let mut names = std::collections::BTreeSet::new();
                if options
                    .hosts
                    .iter()
                    .any(|name| !identifier(name) || !names.insert(name))
                {
                    return Err(
                        "Give --hosts distinct machine names separated by commas, with no spaces."
                            .into(),
                    );
                }
            }
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
            _ => unreachable!(),
        }
        index += 2;
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
    if options.replicas {
        if !options.fp4_local || options.sparks.is_some() || !(options.hosts.is_empty() || options.hosts.len() == replicas) {
            return Err("Use --replicas N with FP4-Local alone, or with N machine names in --hosts. Run vade up --help.".into());
        }
        if options.hosts.is_empty() { options.sparks = Some(replicas); }
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
    if options.sparks.is_some() && !options.hosts.is_empty() {
        return Err("Choose --sparks or --hosts. Run vade up --help.".into());
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
/// Default FP4-Local output limit: `min(FP4_OUTPUT, context)`.
pub const FP4_OUTPUT: u32 = 8192;
/// Largest explicit FP4-Local output limit.
pub const FP4_OUTPUT_LIMIT: u32 = 16384;

/// Default FP4-Local output limit for a context.
pub fn fp4_output(context: u32) -> u32 { context.min(FP4_OUTPUT) }

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

pub fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_'))
}

pub fn help(command: Option<&str>) -> String {
    let mut text: String = match command {
        Some("verify") => crate::verify::HELP.into(),
        Some("detect") => "vade detect\nRead this Spark's identity, available memory, disk, NVIDIA driver and existing model. No files are changed.\n".into(),
        Some("launch") => "vade launch [--context 4096..262144] [--output 1024..16384] [--mtp 0..3] [--cache-bytes 0] [--dry-run]\nOn this Spark: find the FP4-Local checkpoint, save defaults, install if missing, start, then open chat. Repeat to attach. Unset VADE_CONFIG for local lifecycle commands.\nLocal defaults match vade up from a controller: context 32768, output min(8192, context), engine-selected MTP. If the engine refuses the default or saved context for memory, launch retries 16384, 8192 then 4096, saves the one that starts and says which; an explicit --context the engine refuses is an error.\nUses the saved checkpoint, then explicit VADE_MODELS, then the FP4-Local checkpoint under ~/models; asks if missing. Installs under ~/.vade (or VADE_HOME). Local --chats and positive --cache-bytes are unsupported; --cache-bytes 0 disables prefix retention even with an inherited opt-in. See vade up --help for memory options.\n".into(),
        Some("version") => "vade --version (or vade version, vade -V)\nPrint the CLI package version without loading configuration or opening connections.\n".into(),
        Some("run") => "vade run qwen3-8b [--dry-run] [--api-key-file PATH] [--port 8080] [--weights-source DIR] [--release DIR | --engine-source DIR --harness-source DIR]\nRun locally on one Linux/aarch64 NVIDIA GB10. Detect prerequisites before downloading, resume and hash official pinned weights, start an owned server, answer, run doctor.\n--weights-source verifies an existing official BF16 directory read-only, without weight downloads. Other directory names are imported as Qwen3-8B inside VADE_HOME.\nVADE_HOME selects a dedicated private data directory. Unset VADE_CONFIG; use a fresh VADE_HOME if local launch already has defaults. Keep the same selection for chat/status/logs/down. No system tool installation.\n".into(),
        Some("up") => "vade up [--fp4-local | --fp8-legacy] [--sparks N | --hosts a,b] [--replicas N] [--mtp 0..3]\n        [--context 4096..262144] [--output 1024..16384] [--chats 1..32]\n        [--memory-bytes N] [--cache-bytes N] [--dry-run]\nDefault: Qwen3.8 Flash-Next FP4-Local on one Spark; --fp4-local is optional.\nLocal up/launch defaults: context 32768, output min(8192, context); the engine's memory admission decides, and only a refused default or saved context is lowered (16384, 8192, 4096) and saved once it starts. Supported explicit limits are saved; an explicit --context the engine refuses is an error; local --chats and positive --cache-bytes are refused, --cache-bytes 0 disables retention.\nController FP4 defaults match local ones on any Spark count: context 32768, output min(8192, context); the engine admits concurrent chats from context and memory. --context 262144 selects the single-chat long-context preset.\nMTP is engine-selected. FP4-Local one-Spark concurrent chats can draft; the default request depth is 3 with strict verification. Legacy FP8 concurrent chats decode plainly. Per-request draft=plain disables drafting; draft=strict requests it when supported. --mtp overrides MTP, and 0 disables server drafting.\nFP4 supports 1..24 Sparks. Spark counts above 2 require a current install and linked hosts in cable order.\n--fp8-legacy explicitly selects two-Spark FP8: context 131072, output min(65536, context), MTP 2; its output ceiling is 262144.\nExamples: vade up; vade up --context 262144 --output 16384; VADE_CONFIG=cluster.conf vade up --sparks 2.\n--replicas N (2..16, FP4-Local) starts one independent one-Spark server on each of the first N machines (or the N --hosts) and a local gateway on local_port; replica tunnels use the next N local ports. Each replica uses the FP4 defaults above. Use it when the model fits one Spark and many chats run at once; answers are the same on every replica.\nOne Spark needs no fabric. Controller pairs/chains use RDMA; SEQPAR_RDMA=0 selects TCP. Keep VADE_CONFIG and VADE_HOME consistent through chat/status/down.\n".into(),
        Some("down") => "vade down [--dry-run]\nStop the engine and tunnel started by vade.\n".into(),
        Some("status") => "vade status [--json] [--dry-run]\nShow the engine, machines, build, settings, and health.\n--json emits one vade.status.v1 object; exit 0 for stopped/ready, 1 for errors or unhealthy state.\nChecks share a 15-second deadline. --dry-run reads saved state only; unchecked values are null.\n".into(),
        Some("logs") => "vade logs [--dry-run]\nShow recent engine logs.\n".into(),
        Some("chat") => "vade chat [--dry-run] [--api-key-file PATH]\nOpen the Vade agent on the resident engine; local contexts below 8192 tokens use direct text chat. Start once with vade launch or vade up.\nLocal Linux agent mode disables automatic workspace checks with a notice.\nTyping during a reply interrupts it and keeps its partial answer in context.\n/queue on queues follow-ups; /queue off restores steering. /exit leaves the engine running.\nChat history is kept in ~/.vade/projects (VADE_HOME changes the root), never in this folder; VADE_CHAT_IN_PROJECT=1 keeps it in a git-ignored ./.vade.\n".into(),
        Some("doctor") => "vade doctor [--dry-run] [--api-key-file PATH]\nCheck local driver, memory, disk, model, installed binaries and engine health, or the controller harness and configured machines; explain next steps.\n".into(),
        Some("install") => "vade install [--dry-run]\nFor a fresh local installation or saved owner defaults without VADE_CONFIG: detect existing weights, save defaults once in ~/.vade/config, and stage a local engine and chat program. No hand editing is needed.\nWith VADE_CONFIG, or a default controller config before local setup: check prerequisites, copy an exact-source SHA256-verified prebuilt_dir bundle or build source, then run doctor. Model inventory checks overlap copy/build and are cached.\nA generated run qwen3-8b installation uses run again; install requires an explicit VADE_CONFIG to select a controller.\nModel downloads stay in your hands.\n".into(),
        Some("update") => "vade update [--dry-run]\nLocally, restage the current bundle or rebuild the current source; report how to restart. With VADE_CONFIG, update source, rebuild and stage binaries; offer a restart.\n".into(),
    _ => "vade — start, chat, stop, check.\n\n  launch   Install if missing, start FP4-Local on this Spark, then chat\n  up       Start FP4-Local (legacy FP8 needs --fp8-legacy)\n  chat     Open chat here (direct text below 8192 local context tokens)\n  bench    Measure a server from the client\n  verify   Compare exact token IDs with drafting ON and OFF\n  test     Run build, determinism, long-context, GDN and ABI checks\n  down     Stop what vade started\n  status   Show what is running (--json for scripts)\n  logs     Show recent logs\n  doctor   Check what needs attention\n  detect   Show this Spark and check available memory and disk\n  run      Run the separate official Qwen3-8B BF16 catalog\n  install  Save local defaults and install; explicit VADE_CONFIG uses the controller\n  update   Restage local binaries or update controller source\n  version  Print the CLI version\n\nUse vade <command> --help. For bench, vade help bench also works; help aliases and global flags before bench are unsupported. Add --dry-run to preview lifecycle commands.\nLocal defaults: ~/.vade/config (VADE_HOME changes the root).\nController config: explicit VADE_CONFIG selects the controller for up/install/chat/status/logs/down/doctor, even after local setup; keep it set to the same file throughout that lifecycle. Without VADE_CONFIG, saved local setup wins over the user config directory.\n".into(),
    };
    if matches!(command, Some("install" | "doctor")) {
        text.push_str("Transport: one Spark needs no fabric or RDMA; linked Sparks use RDMA by default. Set SEQPAR_RDMA=0 for TCP using the configured fabric addresses on any reachable interface; QSFP and RDMA memlock are unnecessary in TCP mode.\n");
    }
    if matches!(command, Some("down" | "chat" | "status" | "logs" | "doctor" | "install")) {
        text.push_str("Installation: explicit VADE_CONFIG selects the controller; otherwise generated local setup wins. Keep the same VADE_CONFIG and VADE_HOME selection from start through chat/status/logs/down.\n");
    }
    text.push_str(
        "Credentials: --api-key-file PATH or VADE_API_KEY_FILE (chat, doctor, local run).\n",
    );
    text
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;
    fn cli(args: &str) -> Result<Cli, String> { parse(&args.split_whitespace().map(str::to_owned).collect::<Vec<_>>()) }
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
            "up --replicas 2", "up --replicas 2 --chats 1", "up --replicas 2 --hosts a,b --chats 8",
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
    fn parser_records_every_supplied_launch_option() {
        for args in [
            "--sparks 2", "--hosts a,b", "--mtp 2", "--fp4-local", "--fp8-legacy",
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
        let Command::Up(legacy) = cli("up --fp8-legacy").unwrap().command else { panic!() };
        assert!(!legacy.fp4_local); assert_eq!((legacy.context, legacy.output), (131072, 65536));
        let Command::Launch(local) = cli("launch --context 65536 --output 16384").unwrap().command else { panic!() };
        assert!(local.fp4_local && local.supplied_options.contains("--context")); assert_eq!((local.context, local.output), (65536, 16384));
        assert_eq!(cli("--dry-run up --mtp 3 --dry-run").unwrap(), Cli {
            command: Command::Up(UpOptions { supplied_options: ["--mtp".into()].into(), mtp: 3, mtp_explicit: true, ..UpOptions::default() }), dry_run: true
        });
        assert!(matches!(cli("chat -h").unwrap().command, Command::Help(Some(_))));
        assert!(cli("launch --context").unwrap_err().contains("vade launch --help"));
    }
    #[test]
    fn bad_options_are_actionable() {
        for args in [
            "up --hosts a,a", "up --hosts a,", "up --sparks 2 --hosts a,b", "up --mtp 4",
            "up --context 4095", "up --output 262145", "up --mtp", "up --mtp 1 --mtp 2", "down --force", "unknown",
            "up --output 1023", "up --context 4096 --output 8192", "up --fp4-local --fp8-legacy",
            "up --replicas 1", "up --replicas 17", "up --replicas 2 --sparks 2", "up --replicas 3 --hosts a,b", "up --fp8-legacy --replicas 2",
        ] { assert!(cli(args).is_err(), "{args}"); }
        assert_eq!(cli("up --hosts a,b --chats 3 --fast-prompts").unwrap().command, Command::Up(UpOptions {
            supplied_options: ["--hosts".into(), "--chats".into(), "--fast-prompts".into()].into(),
            hosts: vec!["a".into(), "b".into()], chats: Some(3), fast_prompts: true, context: 32768, ..UpOptions::default()
        }));
    }
}
