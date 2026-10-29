//! Mode-scoped command-line configuration and validation.
use std::collections::BTreeMap;
use std::fmt;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::time::Duration;
use vade_harness::{
    app_server, child_agents, context, journal, project, session::MAX_PROMPT_BYTES,
    shell::ApprovalPolicy,
};

use super::help::{AGENT_HELP, DIRECT_HELP, GATEWAY_HELP, HELP};
use crate::native_children;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Demo,
    Direct,
    Agent,
    Gateway,
    Codex,
    Hermes,
}
impl Mode {
    pub(crate) fn help(self) -> &'static str {
        match self {
            Self::Agent => AGENT_HELP,
            Self::Direct => DIRECT_HELP,
            Self::Gateway => GATEWAY_HELP,
            _ => HELP,
        }
    }
    pub(crate) fn delegated(self) -> bool {
        matches!(self, Self::Codex | Self::Hermes)
    }
    pub(crate) fn provider(self) -> &'static str {
        if self == Self::Hermes {
            "Hermes"
        } else {
            "Codex"
        }
    }
}
pub(crate) struct Config {
    pub(crate) mode: Mode,
    pub(crate) endpoint: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) prompt: Option<String>,
    pub(crate) workspace: PathBuf,
    pub(crate) session: Option<String>,
    /// Chat state location; None until `main` applies the Vade-home default.
    pub(crate) state: Option<vade_harness::tools::StateLocation>,
    pub(crate) plain: bool,
    pub(crate) verbose: bool,
    pub(crate) children: bool,
    pub(crate) child_grants: BTreeMap<String, child_agents::Scope>,
    pub(crate) bind: String,
    pub(crate) upstreams: Vec<String>,
    pub(crate) access: vade_harness::gateway::Access,
    pub(crate) codex_bin: PathBuf,
    pub(crate) hermes_bin: PathBuf,
    pub(crate) external_session: Option<String>,
    pub(crate) thread: Option<String>,
    pub(crate) permission: Option<Permission>,
    pub(crate) approval: ApprovalPolicy,
    pub(crate) resume: Option<String>,
    pub(crate) context_tokens: usize,
    pub(crate) max_run_time: Option<Duration>,
    pub(crate) auto_compact: bool,
    pub(crate) auto_continue: bool,
    pub(crate) prewarm_compaction: bool,
    pub(crate) generation: vade_harness::contract::GenerationOptions,
    pub(crate) rules_scopes: Vec<String>,
    /// Direct and agent: load project instruction files (`--no-instructions` clears it).
    pub(crate) instructions: bool,
    pub(crate) shell_exclusions: vade_harness::shell::Exclusions,
    pub(crate) shell_toolchain: Option<PathBuf>,
    /// Agent: user check command run after rounds that changed files, and its source.
    pub(crate) check: Option<(String, &'static str)>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Permission {
    ReadOnly,
    WorkspaceWrite,
    DenyCallbacks,
}
impl Permission {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DenyCallbacks => "deny-callbacks",
        }
    }
    pub(crate) fn protocol(self) -> app_server::PermissionPolicy {
        match self {
            Self::ReadOnly => app_server::PermissionPolicy::ReadOnly,
            Self::WorkspaceWrite => app_server::PermissionPolicy::WorkspaceWrite,
            Self::DenyCallbacks => unreachable!("Codex policy validated by configuration"),
        }
    }
}
pub(crate) enum Parsed {
    Run(Config),
    Help(Mode),
    Version,
}

#[derive(Debug)]
pub(crate) struct ParseError {
    pub(crate) message: &'static str,
    option: Option<String>,
    mode: Mode,
}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(option) = &self.option {
            write!(f, "{option}: ")?;
        }
        let message = self.message.trim_end_matches("; see --help");
        write!(f, "{message}")?;
        if !message.ends_with('.') { write!(f, ".")?; }
        let command = match self.mode {
            Mode::Demo => "vade-harness --help",
            Mode::Direct => "vade-harness direct --help",
            Mode::Agent => "vade-harness agent --help",
            Mode::Gateway => "vade-harness gateway --help",
            Mode::Codex | Mode::Hermes => "vade-harness --help",
        };
        write!(f, " See {command}.")
    }
}

impl Config {
    pub(crate) fn context_budget(&self, contract: &vade_harness::contract::Context) -> usize {
        usize::try_from(contract.token_limit(self.generation.context_limit)).unwrap_or(usize::MAX)
    }
    pub(crate) fn output_reserve(&self, contract: &vade_harness::contract::Context) -> usize {
        let tokens = self
            .generation
            .max_output_tokens
            .unwrap_or_else(|| match contract.input {
                vade_harness::contract::InputCount::Exact { output_tokens, .. } => output_tokens,
                _ => contract
                    .capabilities
                    .as_ref()
                    .map_or(vade_harness::contract::DEFAULT_OUTPUT_TOKENS, |caps| {
                        caps.output_limit.min(caps.context_limit.saturating_sub(256))
                    })
                    .min(vade_harness::contract::AUTOMATIC_OUTPUT_TOKENS)
                    .min((self.context_budget(contract) as u64).saturating_sub(256)),
            });
        usize::try_from(tokens).unwrap_or(usize::MAX)
    }
    pub(crate) fn parse(args: impl IntoIterator<Item = String>) -> Result<Parsed, ParseError> {
        let mut diagnostic = ParseError { message: "", option: None, mode: Mode::Demo };
        Self::parse_options(args, &mut diagnostic).map_err(|message| {
            diagnostic.message = message;
            diagnostic
        })
    }

    fn parse_options(args: impl IntoIterator<Item = String>, diagnostic: &mut ParseError) -> Result<Parsed, &'static str> {
        let mut args = args.into_iter().peekable();
        let mut config = Self {
            mode: Mode::Demo,
            endpoint: None,
            model: None,
            prompt: None,
            workspace: PathBuf::from("."),
            session: None,
            state: None,
            plain: false,
            verbose: false,
            children: false,
            child_grants: BTreeMap::new(),
            bind: "127.0.0.1:8787".into(),
            upstreams: Vec::new(),
            access: vade_harness::gateway::Access::default(),
            codex_bin: PathBuf::from("codex"),
            hermes_bin: PathBuf::from("hermes"),
            external_session: None,
            thread: None,
            permission: None,
            approval: ApprovalPolicy::Ask,
            resume: None,
            context_tokens: context::DEFAULT_CONTEXT_TOKENS,
            max_run_time: vade_harness::agent::Limits::default().max_run_time,
            auto_compact: true,
            auto_continue: true,
            prewarm_compaction: false,
            generation: vade_harness::contract::GenerationOptions::default(),
            rules_scopes: Vec::new(),
            instructions: true,
            shell_exclusions: vade_harness::shell::Exclusions::default(),
            shell_toolchain: None,
            check: None,
        };
        let mut no_check = false;
        let mut selected = None;
        let mut used = Vec::new();
        let mut config_path = None;
        let mut time_override = None;
        while let Some(arg) = args.next() {
            diagnostic.option = vade_harness::diagnostics::option_name(&arg).map(str::to_owned);
            diagnostic.mode = config.mode;
            if arg == "--help" || arg == "-h" {
                return Ok(Parsed::Help(config.mode));
            }
            if arg == "--version" {
                return Ok(Parsed::Version);
            }
            let mut select = |mode| -> Result<(), &'static str> {
                if selected.is_some_and(|old| old != mode) {
                    return Err("choose exactly one mode");
                }
                selected = Some(mode);
                config.mode = mode;
                Ok(())
            };
            match arg.as_str() {
                "demo" | "--demo" => select(Mode::Demo)?,
                "direct" => select(Mode::Direct)?,
                "agent" => select(Mode::Agent)?,
                "gateway" => select(Mode::Gateway)?,
                "codex" => select(Mode::Codex)?,
                "hermes" => select(Mode::Hermes)?,
                "--mode" => {
                    let mode = args.next().ok_or("--mode needs a value")?;
                    select(match mode.as_str() {
                        "demo" => Mode::Demo,
                        "direct" => Mode::Direct,
                        "agent" => Mode::Agent,
                        "gateway" => Mode::Gateway,
                        "codex" => Mode::Codex,
                        "hermes" => Mode::Hermes,
                        _ => {
                            return Err(
                                "mode must be demo, direct, agent, gateway, codex, or hermes",
                            );
                        }
                    })?;
                }
                "--verbose" => config.verbose = true,
                "--greedy" => {
                    if used.contains(&"--greedy") { return Err("--greedy was specified more than once"); }
                    config.generation.temperature = Some(0.0);
                    used.push("--greedy");
                }
                "--plain" => {
                    config.plain = true;
                    used.push("plain");
                }
                "--no-compact" | "--no-auto-compact" => {
                    if !config.auto_compact {
                        return Err("--no-compact was specified more than once");
                    }
                    config.auto_compact = false;
                }
                "--no-instructions" => {
                    if !config.instructions {
                        return Err("--no-instructions was specified more than once");
                    }
                    config.instructions = false;
                }
                "--no-auto-continue" => {
                    if !config.auto_continue {
                        return Err("--no-auto-continue was specified more than once");
                    }
                    config.auto_continue = false;
                    used.push("--no-auto-continue");
                }
                "--state-in-project" => {
                    if used.contains(&"--state-in-project") {
                        return Err("--state-in-project was specified more than once");
                    }
                    used.push("--state-in-project");
                    config.state = Some(vade_harness::tools::StateLocation::Project);
                }
                "--no-check" => {
                    if no_check {
                        return Err("--no-check was specified more than once");
                    }
                    no_check = true;
                    used.push("--no-check");
                }
                "--prewarm-compaction" => {
                    if config.prewarm_compaction {
                        return Err("--prewarm-compaction was specified more than once");
                    }
                    config.prewarm_compaction = true;
                }
                "--children" => {
                    if config.children {
                        return Err("--children was specified more than once");
                    }
                    config.children = true;
                }
                "--endpoint"
                | "--model"
                | "--prompt"
                | "--workspace"
                | "--session"
                | "--vade-home"
                | "--bind"
                | "--upstream"
                | "--allow-origin"
                | "--allow-host"
                | "--token"
                | "--hermes-bin"
                | "--external-session"
                | "--codex-bin"
                | "--thread"
                | "--permission"
                | "--approval"
                | "--resume"
                | "--context-tokens"
                | "--agent-time-limit-seconds"
                | "--config"
                | "--rules-scope"
                | "--shell-exclude"
                | "--shell-toolchain"
                | "--child-edit"
                | "--child-shell"
                | "--reply-tokens"
                | "--max-output-tokens"
                | "--thinking"
                | "--effort"
                | "--check"
                | "--temperature"
                | "--seed" => {
                    if used.contains(&arg.as_str())
                        && !matches!(
                            arg.as_str(),
                            "--rules-scope"
                                | "--shell-exclude"
                                | "--upstream"
                                | "--allow-origin"
                                | "--allow-host"
                                | "--child-edit"
                                | "--child-shell"
                        )
                    {
                        return Err("an option was specified more than once");
                    }
                    let value = args.next().ok_or("option needs a value; see --help")?;
                    if value.is_empty() {
                        return Err("option value cannot be empty");
                    }
                    match arg.as_str() {
                        "--config" => {
                            config_path = Some(PathBuf::from(value));
                            used.push("--config");
                        }
                        "--agent-time-limit-seconds" => {
                            time_override = Some(parse_time_limit(&value)?);
                            used.push("--agent-time-limit-seconds");
                        }
                        "--temperature" => {
                            let temperature: f64 =
                                value.parse().map_err(|_| "invalid temperature")?;
                            if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
                                return Err("temperature must be finite and between 0 and 2");
                            }
                            config.generation.temperature = Some(temperature);
                            used.push("--temperature");
                        }
                        "--seed" => {
                            config.generation.seed = Some(
                                value
                                    .parse()
                                    .map_err(|_| "seed must be an unsigned 64-bit integer")?,
                            );
                            used.push("--seed");
                        }
                        "--reply-tokens" => {
                            let n = value.parse::<u64>().map_err(|_| "invalid reply token cap")?;
                            config.generation.reply_tokens = Some(n);
                            used.push("--reply-tokens");
                        }
                        "--max-output-tokens" => {
                            let n: u64 = value.parse().map_err(|_| "invalid output token limit")?;
                            if n == 0 {
                                return Err("output token limit must be positive");
                            }
                            config.generation.max_output_tokens = Some(n);
                            used.push("--max-output-tokens");
                        }
                        "--thinking" => {
                            config.generation.thinking = Some(match value.as_str() {
                                "on" => true,
                                "off" => false,
                                _ => return Err("thinking must be on or off"),
                            });
                            used.push("--thinking");
                        }
                        "--effort" => {
                            config.generation.effort = Some(effort(&value)?);
                            used.push("--effort");
                        }
                        "--check" => {
                            config.check = Some((check_command(&value)?, "--check"));
                            used.push("--check");
                        }
                        "--shell-toolchain" => {
                            config.shell_toolchain = Some(PathBuf::from(value));
                            used.push("--shell-toolchain");
                        }
                        "--shell-exclude" => {
                            config.shell_exclusions.add(&value)?;
                            used.push("--shell-exclude");
                        }
                        "--rules-scope" => {
                            if config.rules_scopes.len() >= project::MAX_SCOPES {
                                return Err("at most eight --rules-scope options");
                            }
                            config.rules_scopes.push(value);
                            used.push("--rules-scope");
                        }
                        "--approval" => {
                            config.approval = ApprovalPolicy::parse(&value)
                                .ok_or("approval must be ask, auto, or never")?;
                            used.push("--approval");
                        }
                        "--resume" => {
                            if !journal::valid_name(&value) {
                                return Err("invalid session name");
                            }
                            config.resume = Some(value);
                            used.push("--resume");
                        }
                        "--context-tokens" => {
                            config.context_tokens =
                                value.parse().map_err(|_| "invalid context token budget")?;
                            if !(8192..=2_000_000).contains(&config.context_tokens) {
                                return Err("context token budget must be 8192–2000000");
                            }
                            used.push("--context-tokens");
                        }
                        "--child-edit" => {
                            let (name, path) = value
                                .split_once('=')
                                .ok_or("--child-edit needs NAME=PATH")?;
                            native_children::add_grant(&mut config.child_grants, name, Some(path))?;
                        }
                        "--child-shell" => {
                            native_children::add_grant(&mut config.child_grants, &value, None)?;
                        }
                        "--endpoint" => {
                            config.endpoint = Some(value);
                            used.push("--endpoint");
                        }
                        "--model" => {
                            config.model = Some(value);
                            used.push("--model");
                        }
                        "--prompt" => {
                            config.prompt = Some(value);
                            used.push("--prompt");
                        }
                        "--workspace" => {
                            config.workspace = PathBuf::from(value);
                            used.push("--workspace");
                        }
                        "--vade-home" => {
                            config.state = Some(
                                vade_harness::tools::StateLocation::home(PathBuf::from(value))
                                    .map_err(|_| "--vade-home must be an absolute path without '..'")?,
                            );
                            used.push("--vade-home");
                        }
                        "--session" => {
                            config.session = Some(value);
                            used.push("--session");
                        }
                        "--bind" => {
                            config.bind = value;
                            used.push("--bind");
                        }
                        "--upstream" => {
                            if config.upstreams.len() == 64 {
                                return Err("gateway supports at most 64 upstreams");
                            }
                            config.upstreams.push(value);
                            used.push("--upstream");
                        }
                        "--allow-origin" => {
                            config.access.allow_origin(&value)?;
                            used.push("--allow-origin");
                        }
                        "--allow-host" => {
                            config.access.allow_host(&value)?;
                            used.push("--allow-host");
                        }
                        "--token" => {
                            config.access.require_token(&value)?;
                            used.push("--token");
                        }
                        "--hermes-bin" => {
                            if value.len() > 4096 || value.chars().any(char::is_control) {
                                return Err("invalid Hermes executable path");
                            }
                            config.hermes_bin = PathBuf::from(value);
                            used.push("--hermes-bin");
                        }
                        "--external-session" => {
                            if value.len() > 512
                                || value.chars().any(|c| c.is_control() || c.is_whitespace())
                            {
                                return Err("invalid external session identifier");
                            }
                            config.external_session = Some(value);
                            used.push("--external-session");
                        }
                        "--codex-bin" => {
                            if value.len() > 4096 || value.chars().any(char::is_control) {
                                return Err("invalid Codex executable path");
                            }
                            config.codex_bin = PathBuf::from(value);
                            used.push("--codex-bin");
                        }
                        "--thread" => {
                            if value.len() > 512
                                || value.chars().any(|c| c.is_control() || c.is_whitespace())
                            {
                                return Err("invalid Codex thread identifier");
                            }
                            config.thread = Some(value);
                            used.push("--thread");
                        }
                        "--permission" => {
                            config.permission = Some(match value.as_str() {
                                "read-only" => Permission::ReadOnly,
                                "workspace-write" => Permission::WorkspaceWrite,
                                "deny-callbacks" => Permission::DenyCallbacks,
                                _ => {
                                    return Err(
                                        "permission must be read-only, workspace-write (Codex), or deny-callbacks (Hermes)",
                                    );
                                }
                            });
                            used.push("--permission");
                        }
                        _ => unreachable!(),
                    }
                }
                _ => return Err("unknown argument; see --help"),
            }
        }
        // Cross-option and configuration validation must not blame whichever
        // unrelated option happened to occur last.
        diagnostic.option = None;
        diagnostic.mode = config.mode;
        if config.mode != Mode::Agent
            && (used.contains(&"--approval")
                || used.contains(&"--resume")
                || used.contains(&"--context-tokens")
                || used.contains(&"--agent-time-limit-seconds")
                || used.contains(&"--config")
                || used.contains(&"--rules-scope")
                || used.contains(&"--shell-exclude")
                || used.contains(&"--shell-toolchain")
                || used.contains(&"--check")
                || used.contains(&"--no-check"))
        {
            return Err(
                "--approval, --resume, --context-tokens, --agent-time-limit-seconds, --config, --rules-scope, --shell-exclude, --shell-toolchain, --check and --no-check require agent mode",
            );
        }
        if used.contains(&"--vade-home") && used.contains(&"--state-in-project") {
            return Err("choose --vade-home DIR or --state-in-project, not both");
        }
        if no_check && config.check.is_some() {
            return Err("choose --check COMMAND or --no-check, not both");
        }
        if config.generation.effort.is_none() {
            if let Some(value) = std::env::var_os("VADE_HARNESS_EFFORT") {
                config.generation.effort =
                    Some(effort(value.to_str().ok_or(
                        "VADE_HARNESS_EFFORT must be low, medium, high or xhigh",
                    )?)?);
            }
        }
        if config.generation.effort.is_some() && config.generation.thinking == Some(false) {
            return Err("--effort needs thinking; remove --thinking off (or VADE_HARNESS_EFFORT)");
        }
        if config.mode == Mode::Agent && config.check.is_none() && !no_check {
            config.check =
                workspace_check(&config.workspace)?.map(|command| (command, ".vade/check"));
        }
        if config.check.is_some() {
            if !vade_harness::coding_tools::SHELL_AVAILABLE {
                return Err("Check not run: --check and .vade/check require the macOS sandbox; use --no-check on this platform.");
            }
            if !vade_harness::coding_tools::shell_offered() {
                return Err("Check not run: VADE_HARNESS_SHELL=off hides workspace_shell; unset it or use --no-check.");
            }
            if config.approval == ApprovalPolicy::Never {
                return Err("Check not run: --approval never denies commands; use --approval auto or --no-check.");
            }
            if config.approval == ApprovalPolicy::Ask
                && (config.prompt.is_some() || !std::io::stdin().is_terminal())
            {
                return Err("Check not run: --approval ask needs interactive input; use --approval auto or --no-check.");
            }
        }
        if used.contains(&"--greedy") && used.contains(&"--temperature") {
            return Err("choose --greedy or --temperature, not both");
        }
        if !matches!(config.mode, Mode::Agent | Mode::Direct)
            && (config.generation.max_output_tokens.is_some()
                || config.generation.thinking.is_some()
                || used.contains(&"--effort")
                || config.generation.temperature.is_some()
                || config.generation.seed.is_some())
        {
            return Err("generation flags require direct or agent mode");
        }
        if config.generation.reply_tokens.is_some() && config.mode != Mode::Agent {
            return Err("--reply-tokens requires agent mode");
        }
        if !config.auto_continue && !matches!(config.mode, Mode::Agent | Mode::Direct) {
            return Err("--no-auto-continue requires direct or agent mode");
        }
        if !config.instructions && !matches!(config.mode, Mode::Agent | Mode::Direct) {
            return Err("--no-instructions requires direct or agent mode");
        }
        if config.resume.is_some() && !config.rules_scopes.is_empty() {
            return Err(
                "resume preserves frozen project scopes; use a new session to select different scopes",
            );
        }
        if config.resume.is_some() && config.session.is_some() {
            return Err("choose --session for new or --resume for existing, not both");
        }
        if let Some(text) = &config.prompt {
            if text.trim().is_empty() {
                return Err("--prompt must contain text");
            }
            if text.len() > MAX_PROMPT_BYTES {
                return Err("prompt exceeds 64 KiB");
            }
        }
        if !config.auto_compact && config.mode != Mode::Agent {
            return Err("--no-compact requires agent mode");
        }
        if config.prewarm_compaction && config.mode != Mode::Agent {
            return Err("--prewarm-compaction requires agent mode");
        }
        if !config.mode.delegated() && config.permission.is_some() {
            return Err("--permission requires a delegated mode");
        }
        if !config.child_grants.is_empty() && !config.children {
            return Err("--child-edit and --child-shell require --children");
        }
        if !vade_harness::coding_tools::shell_offered()
            && config.child_grants.values().any(|scope| scope.tools.iter().any(|tool| tool == "workspace_shell"))
        {
            return Err(if vade_harness::coding_tools::SHELL_AVAILABLE {
                "--child-shell: VADE_HARNESS_SHELL=off hides workspace_shell; unset it or drop --child-shell."
            } else {
                "--child-shell needs the macOS sandbox; workspace_shell is not offered on this platform."
            });
        }
        if config.children && config.mode != Mode::Agent {
            return Err("--children requires native agent mode");
        }
        if config.mode != Mode::Codex && (config.thread.is_some() || used.contains(&"--codex-bin"))
        {
            return Err("--thread and --codex-bin require codex mode");
        }
        if config.mode != Mode::Hermes
            && (config.external_session.is_some() || used.contains(&"--hermes-bin"))
        {
            return Err("--external-session and --hermes-bin require hermes mode");
        }
        if config.mode != Mode::Gateway
            && used.iter().any(|u| matches!(*u, "--allow-origin" | "--allow-host" | "--token"))
        {
            return Err("--allow-origin, --allow-host and --token are gateway options");
        }
        if let Some(model) = &config.model {
            if model.len() > 256 || model.chars().any(char::is_control) {
                return Err("invalid model name");
            }
        }
        match config.mode {
            Mode::Demo => {
                if config.endpoint.is_some()
                    || config.model.is_some()
                    || !config.upstreams.is_empty()
                    || used.contains(&"--bind")
                {
                    return Err(
                        "demo mode does not accept upstream, endpoint, model, or bind options",
                    );
                }
            }
            Mode::Direct | Mode::Agent => {
                if config.mode == Mode::Agent && !used.contains(&"--workspace") {
                    return Err(
                        "agent mode requires explicit --workspace, --endpoint, and --model",
                    );
                }
                validate_url(
                    config
                        .endpoint
                        .as_deref()
                        .ok_or("direct/agent mode requires --endpoint and --model")?,
                )?;
                let model = config
                    .model
                    .as_deref()
                    .ok_or("direct/agent mode requires --model")?;
                if model.len() > 256 || model.chars().any(char::is_control) {
                    return Err("invalid model name");
                }
                if !config.upstreams.is_empty() || used.contains(&"--bind") {
                    return Err("--upstream and --bind are gateway options");
                }
            }
            Mode::Codex => {
                if !used.contains(&"--workspace")
                    || !matches!(
                        config.permission,
                        Some(Permission::ReadOnly | Permission::WorkspaceWrite)
                    )
                {
                    return Err(
                        "codex mode requires explicit --workspace and --permission read-only|workspace-write",
                    );
                }
                if config.endpoint.is_some()
                    || !config.upstreams.is_empty()
                    || used.contains(&"--bind")
                {
                    return Err(
                        "codex mode delegates over stdio; endpoint, upstream, and bind do not apply",
                    );
                }
            }
            Mode::Hermes => {
                if !used.contains(&"--workspace")
                    || config.permission != Some(Permission::DenyCallbacks)
                {
                    return Err(
                        "hermes mode requires explicit --workspace and --permission deny-callbacks; no external sandbox is verified",
                    );
                }
                if config.endpoint.is_some()
                    || !config.upstreams.is_empty()
                    || config.model.is_some()
                    || used.contains(&"--bind")
                {
                    return Err(
                        "Hermes ACP uses its externally configured model; endpoint, upstream, model, and bind are unsupported",
                    );
                }
            }
            Mode::Gateway => {
                if config.upstreams.is_empty() {
                    return Err("gateway mode requires --upstream");
                }
                for upstream in &config.upstreams {
                    validate_url(upstream)?;
                }
                config
                    .bind
                    .parse::<std::net::SocketAddr>()
                    .map_err(|_| "--bind must be an IP address and port")?;
                if config.endpoint.is_some()
                    || config.model.is_some()
                    || config.prompt.is_some()
                    || config.session.is_some()
                    || used.contains(&"--workspace")
                    || config.state.is_some()
                    || config.plain
                {
                    return Err("gateway accepts only --upstream, --bind, --allow-origin, --allow-host and --token");
                }
                config.access.token_from_env()?;
            }
        }
        if let Some(path) = config_path {
            let settings = read_native_config(&path)?;
            if let Some(limit) = settings.max_run_time {
                config.max_run_time = limit;
            }
            if !used.contains(&"--no-auto-continue") {
                if let Some(enabled) = settings.auto_continue {
                    config.auto_continue = enabled;
                }
            }
        }
        if let Some(limit) = time_override {
            config.max_run_time = limit;
        }
        if used.contains(&"--context-tokens") {
            config.generation.context_limit = Some(config.context_tokens as u64);
        }
        Ok(Parsed::Run(config))
    }
}

fn effort(value: &str) -> Result<&'static str, &'static str> {
    match value {
        "low" => Ok("low"),
        "medium" => Ok("medium"),
        "high" => Ok("high"),
        "xhigh" => Ok("xhigh"),
        _ => Err("effort must be low, medium, high or xhigh"),
    }
}

/// One line of shell text for `/bin/sh -c` inside the command sandbox.
fn check_command(value: &str) -> Result<String, &'static str> {
    let command = value.trim();
    if command.is_empty() || command.len() > 4096 || command.chars().any(char::is_control) {
        return Err("check command must be one nonempty line of at most 4096 bytes");
    }
    Ok(command.to_owned())
}

/// Optional `.vade/check` in the workspace: first nonblank line not starting
/// with `#`. Only a regular file is read, bounded to 4 KiB.
fn workspace_check(workspace: &std::path::Path) -> Result<Option<String>, &'static str> {
    use std::io::Read;
    let path = workspace.join(".vade").join("check");
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() => (),
        Ok(_) => return Err(".vade/check must be a regular file"),
        Err(_) => return Ok(None),
    }
    let mut text = String::new();
    std::fs::File::open(&path)
        .and_then(|file| file.take(4097).read_to_string(&mut text))
        .map_err(|_| "cannot read .vade/check as UTF-8")?;
    if text.len() > 4096 {
        return Err(".vade/check exceeds 4 KiB");
    }
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .map(check_command)
        .transpose()
}

fn parse_time_limit(value: &str) -> Result<Option<Duration>, &'static str> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err("agent time limit must be a nonnegative integer in seconds");
    }
    let seconds: u64 = value.parse().map_err(|_| "agent time limit is too large")?;
    Ok((seconds != 0).then(|| Duration::from_secs(seconds)))
}

/// Explicit native configuration only; never search user homes or read credentials.
#[derive(Default)]
struct NativeConfig {
    max_run_time: Option<Option<Duration>>,
    auto_continue: Option<bool>,
}
fn read_native_config(path: &std::path::Path) -> Result<NativeConfig, &'static str> {
    use std::io::Read;
    use vade_harness::json::{self, Value};
    const CONFIG_BYTES: u64 = 64 * 1024;
    if !std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file()) {
        return Err("native --config must be a readable regular file");
    }
    let mut open = std::fs::OpenOptions::new();
    open.read(true);
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        #[cfg(target_os = "macos")]
        const NONBLOCK: i32 = 0x4;
        #[cfg(target_os = "linux")]
        const NONBLOCK: i32 = 0x800;
        // The path may change after preflight; never block opening a FIFO.
        open.custom_flags(NONBLOCK);
    }
    let file = open
        .open(path)
        .map_err(|_| "cannot open native --config file")?;
    if !file.metadata().is_ok_and(|metadata| metadata.is_file()) {
        return Err("native --config must be a regular file");
    }
    let mut text = String::new();
    file.take(CONFIG_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|_| "cannot read native --config as UTF-8")?;
    if text.len() as u64 > CONFIG_BYTES {
        return Err("native --config exceeds 64 KiB");
    }
    let Value::Object(fields) = json::parse(&text).map_err(|_| "invalid native --config JSON")?
    else {
        return Err("native --config must be a JSON object");
    };
    let mut settings = NativeConfig::default();
    for (key, value) in fields {
        match key.as_str() {
            "agent_time_limit_seconds" => {
                let Value::Number(seconds) = value else {
                    return Err("agent_time_limit_seconds must be a nonnegative integer");
                };
                settings.max_run_time = Some(parse_time_limit(&seconds)?);
            }
            "auto_continue" => {
                let Value::Bool(enabled) = value else {
                    return Err("auto_continue must be true or false");
                };
                settings.auto_continue = Some(enabled);
            }
            _ => {
                return Err(
                    "unknown native --config key; expected agent_time_limit_seconds or auto_continue",
                );
            }
        }
    }
    Ok(settings)
}
pub(crate) fn validate_url(url: &str) -> Result<(), &'static str> {
    if url.starts_with("https://") {
        return Err("TLS is unsupported; use an explicitly configured http:// upstream");
    }
    let rest = url
        .strip_prefix("http://")
        .ok_or("upstream URL must begin with http://")?;
    let authority = rest.split('/').next().unwrap_or("");
    if authority.is_empty()
        || authority.contains('@')
        || url.contains('#')
        || url.contains('?')
        || url.chars().any(|c| c.is_control() || c.is_whitespace())
    {
        return Err(
            "upstream URL must have a host and no credentials, query, fragment, or whitespace",
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
