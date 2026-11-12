//! Opt-in homes: never merge into, overwrite, or discover installed client state.
use super::{Client, Config, Target, claude_environment, quoted, shell_quote};
use std::{
    fs::{self, DirBuilder, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
};

pub(super) const TOOL_RULE: &str = "When a task needs a tool, call the tool immediately; do not end the turn with a promise such as 'Let me check' before calling it.\n";

fn destination(path: &Path) -> Result<PathBuf, &'static str> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|_| "cannot resolve dedicated home parent")?
            .join(path)
    };
    let mut clean = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => return Err("dedicated home cannot contain parent traversal"),
            Component::CurDir => {}
            Component::Normal(name)
                if name.to_str().is_some_and(|name| {
                    [".claude", ".codex", ".hermes", ".pi", ".config"]
                        .iter()
                        .any(|reserved| name.eq_ignore_ascii_case(reserved))
                }) =>
            {
                return Err("use a fresh scratch directory outside installed client configuration");
            }
            other => clean.push(other.as_os_str()),
        }
    }
    let parent = clean
        .parent()
        .ok_or("dedicated home needs an existing parent")?;
    if fs::canonicalize(parent).map_err(|_| "dedicated home needs an existing parent")? != parent {
        return Err("dedicated home parent must be a physical path without symlinks");
    }
    Ok(clean)
}

fn directory(path: &Path) -> std::io::Result<()> {
    let mut builder = DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

fn file(path: &Path, text: &str, executable: bool) -> std::io::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(if executable { 0o700 } else { 0o600 });
    }
    #[cfg(not(unix))]
    let _ = executable;
    let mut output = options.open(path)?;
    output.write_all(text.as_bytes())?;
    output.sync_all()
}

fn configuration(config: &Config, home: &Path) -> Vec<(&'static str, String)> {
    let context = config.context_window.unwrap();
    let output = config.max_output_tokens.unwrap();
    let endpoint = if config.target == Target::Direct {
        &config.endpoint
    } else {
        &config.gateway
    };
    let model = quoted(&config.model);
    let base = quoted(endpoint);
    match config.client {
        // Read and search only, like the other homes; Claude Code denies other tools in print mode.
        // The same settings as the launcher's environment. A tool that starts Claude Code
        // with only this folder as its config directory (no launch.sh) then reaches Vade too.
        Client::Claude => vec![
            ("settings.json", format!("{{\"env\":{{{}}},\"permissions\":{{\"allow\":[\"Read\",\"Grep\",\"Glob\"],\"deny\":[\"Bash\",\"Edit\",\"Write\",\"NotebookEdit\",\"WebFetch\",\"WebSearch\"]}}}}\n",
                claude_home_environment(config).iter().map(|(key, value)| format!("{}:{}", quoted(key), quoted(value))).collect::<Vec<_>>().join(","))),
            (".claude.json", "{\"hasCompletedOnboarding\":true}\n".into()),
            ("CLAUDE.md", TOOL_RULE.into()),
        ],
        Client::Codex => vec![
            (
                "config.toml",
                format!(
                    "model_catalog_json = {}\nmodel_context_window = {context}\nmodel_auto_compact_token_limit = {}\n{}web_search = \"disabled\"\ncheck_for_update_on_startup = false\nsandbox_mode = \"read-only\"\napproval_policy = \"never\"\n{}\n[analytics]\nenabled = false\n\n[features]\nmulti_agent = false\ngoals = false\n",
                    quoted(&home.join("models.json").to_string_lossy()),
                    context - output,
                    // A model that cannot turn thinking off (GLM) refuses the "none" effort with
                    // HTTP 400 on every request. There Codex names no effort and the model thinks.
                    if crate::contract::thinking_off_supported(&config.model) { "model_reasoning_effort = \"none\"\n" } else { "" },
                    config.snippet(endpoint)
                ),
            ),
            (
                "models.json",
                format!(
                    "{{\"models\":[{{\"slug\":{model},\"display_name\":{model},\"description\":\"Explicit local provider\",\"default_reasoning_level\":null,\"supported_reasoning_levels\":[],\"shell_type\":\"shell_command\",\"visibility\":\"list\",\"supported_in_api\":true,\"priority\":1,\"upgrade\":null,\"base_instructions\":\"You are a coding assistant. Complete the user's request using the provided tools when needed. Follow workspace instructions and tool permissions.\",\"supports_reasoning_summaries\":false,\"default_reasoning_summary\":\"none\",\"support_verbosity\":false,\"default_verbosity\":null,\"apply_patch_tool_type\":\"freeform\",\"web_search_tool_type\":\"text\",\"truncation_policy\":{{\"mode\":\"tokens\",\"limit\":{output}}},\"supports_parallel_tool_calls\":false,\"supports_image_detail_original\":false,\"context_window\":{context},\"max_context_window\":{context},\"auto_compact_token_limit\":{},\"experimental_supported_tools\":[],\"input_modalities\":[\"text\"],\"supports_search_tool\":false,\"supports_experimental_context\":false,\"use_responses_lite\":false}}]}}\n",
                    context - output
                ),
            ),
            ("AGENTS.md", TOOL_RULE.into()),
        ],
        Client::Hermes => vec![
            (
                "config.yaml",
                format!(
                    "{}  api_mode: chat_completions\n  context_length: {context}\n  streaming: true\nagent:\n  max_turns: 8\n  api_max_retries: 1\ntoolsets: [file]\nfallback_providers: []\nupdates:\n  check: false\ndisplay:\n  streaming: true\n",
                    if context < 64_000 {
                        config
                            .snippet(endpoint)
                            .replace("  provider: custom\n", "  provider: lmstudio\n")
                    } else {
                        config.snippet(endpoint)
                    }
                ),
            ),
            ("SOUL.md", TOOL_RULE.into()),
        ],
        Client::OpenCode => vec![
            (
                "opencode.json",
                format!(
                    "{{\n  \"$schema\": \"https://opencode.ai/config.json\",\n  \"model\": {},\n  \"enabled_providers\": [\"vade\"],\n  \"autoupdate\": false,\n  \"share\": \"disabled\",\n  \"instructions\": [{}],\n  \"permission\": {{\"*\": \"deny\", \"read\": \"allow\", \"list\": \"allow\", \"glob\": \"allow\", \"grep\": \"allow\"}},\n  \"provider\": {{\"vade\": {{\"npm\": \"@ai-sdk/openai-compatible\", \"name\": \"Vade\",\n    \"options\": {{\"baseURL\": {base}, \"apiKey\": \"no-key-required\"}},\n    \"models\": {{{model}: {{\"name\": {model}, \"limit\": {{\"context\": {context}, \"output\": {output}}}}}}}\n  }}}}\n}}\n",
                    quoted(&format!("vade/{}", config.model)),
                    quoted(&home.join("AGENTS.md").to_string_lossy())
                ),
            ),
            ("AGENTS.md", TOOL_RULE.into()),
        ],
        Client::Pi => vec![
            (
                "models.json",
                format!(
                    "{{\"providers\":{{\"vade\":{{\"baseUrl\":{base},\"api\":\"openai-completions\",\"apiKey\":\"no-key-required\",\"compat\":{{\"supportsDeveloperRole\":false,\"supportsReasoningEffort\":false,\"maxTokensField\":\"max_tokens\"}},\"models\":[{{\"id\":{model},\"contextWindow\":{context},\"maxTokens\":{output}}}]}}}}}}\n"
                ),
            ),
            (
                "settings.json",
                format!(
                    "{{\"defaultProvider\":\"vade\",\"defaultModel\":{model},\"enableInstallTelemetry\":false,\"compaction\":{{\"reserveTokens\":{output},\"keepRecentTokens\":{}}}}}\n",
                    (context - output).min(8192)
                ),
            ),
            ("AGENTS.md", TOOL_RULE.into()),
        ],
        Client::Aider | Client::Continue | Client::OpenWebUi | Client::Generic => unreachable!(),
    }
}

/// Claude Code's settings for a home: the gateway settings and the two verified limits.
/// The launcher passes them as environment; `settings.json` carries the same values.
fn claude_home_environment(config: &Config) -> Vec<(&'static str, String)> {
    let endpoint = if config.target == Target::Direct { &config.endpoint } else { &config.gateway };
    let mut environment = claude_environment(endpoint, &config.model);
    environment.push(("CLAUDE_CODE_MAX_CONTEXT_TOKENS", config.context_window.unwrap().to_string()));
    environment.push(("CLAUDE_CODE_MAX_OUTPUT_TOKENS", config.max_output_tokens.unwrap().to_string()));
    environment
}

fn launcher(config: &Config) -> String {
    let default = match config.client {
        Client::Claude => "claude",
        Client::Codex => "codex",
        Client::Hermes => "hermes",
        Client::OpenCode => "opencode",
        Client::Pi => "pi",
        Client::Aider | Client::Continue | Client::OpenWebUi | Client::Generic => unreachable!(),
    };
    let binary = config.client_bin.as_deref().unwrap_or(default);
    let claude = if config.client == Client::Claude {
        claude_home_environment(config).iter().fold("CLAUDE_CONFIG_DIR=\"$VADE_CLIENT_HOME\"".to_owned(), |line, (key, value)| format!("{line} {key}={}", shell_quote(value)))
    } else {
        String::new()
    };
    let (environment, flags) = match config.client {
        Client::Claude => (claude.as_str(), ""),
        Client::Codex => ("CODEX_HOME=\"$VADE_CLIENT_HOME\"", "--no-daemon"),
        Client::Hermes => (
            "HERMES_HOME=\"$VADE_CLIENT_HOME\" HERMES_WRITE_SAFE_ROOT=\"$VADE_CLIENT_WORKSPACE\" PYTHONDONTWRITEBYTECODE=1",
            "",
        ),
        Client::OpenCode => (
            "OPENCODE_CONFIG=\"$VADE_CLIENT_HOME/opencode.json\" OPENCODE_CONFIG_DIR=\"$VADE_CLIENT_HOME\" OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_LSP_DOWNLOAD=1",
            "--pure",
        ),
        Client::Pi => (
            "PI_CODING_AGENT_DIR=\"$VADE_CLIENT_HOME\" PI_OFFLINE=1 PI_TELEMETRY=0",
            "--offline --no-extensions --no-skills --no-prompt-templates --tools read,grep,find,ls",
        ),
        Client::Aider | Client::Continue | Client::OpenWebUi | Client::Generic => unreachable!(),
    };
    format!(
        "#!/bin/sh\nset -eu\nVADE_CLIENT_HOME=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd -P)\nVADE_CLIENT_WORKSPACE=$(pwd -P)\nset -- {} {flags} \"$@\"\nif [ \"${{NO_COLOR+x}}\" = x ]; then set -- \"NO_COLOR=$NO_COLOR\" \"$@\"; fi\nexec env -i PATH=\"${{PATH:-/usr/bin:/bin}}\" HOME=\"$VADE_CLIENT_HOME/user\" TMPDIR=\"$VADE_CLIENT_HOME/tmp\" XDG_CONFIG_HOME=\"$VADE_CLIENT_HOME/xdg/config\" XDG_CACHE_HOME=\"$VADE_CLIENT_HOME/xdg/cache\" XDG_DATA_HOME=\"$VADE_CLIENT_HOME/xdg/data\" XDG_STATE_HOME=\"$VADE_CLIENT_HOME/xdg/state\" TERM=\"${{TERM:-dumb}}\" {environment} \"$@\"\n",
        shell_quote(binary)
    )
}

pub(super) fn write(config: &Config) -> Result<(), &'static str> {
    let home = destination(config.write.as_ref().unwrap())?;
    let mut files = configuration(config, &home);
    files.push(("launch.sh", launcher(config)));
    // Exclusive directory creation refuses real client homes, existing scratch
    // state, and final symlinks without reading any configuration there.
    directory(&home)
        .map_err(|_| "dedicated home must be a new directory with an existing parent")?;
    let directories = [
        "user",
        "tmp",
        "xdg",
        "xdg/config",
        "xdg/cache",
        "xdg/data",
        "xdg/state",
    ];
    let mut created_files = Vec::new();
    let mut created_dirs = Vec::new();
    let result = (|| {
        for relative in directories {
            let path = home.join(relative);
            directory(&path)?;
            created_dirs.push(path);
        }
        for (name, text) in &files {
            let path = home.join(name);
            file(&path, text, *name == "launch.sh")?;
            created_files.push(path);
        }
        Ok::<(), std::io::Error>(())
    })();
    if result.is_err() {
        for path in created_files.iter().rev() {
            let _ = fs::remove_file(path);
        }
        for path in created_dirs.iter().rev() {
            let _ = fs::remove_dir(path);
        }
        let _ = fs::remove_dir(&home);
        return Err("cannot finish dedicated home; only newly created entries were cleaned up");
    }
    crate::diagnostics::stderr_line(format_args!(
        "Dedicated client home created. Start from your chosen workspace:\n  {}\n{}The launcher uses a clean environment and private HOME/XDG/client state.\nTool permissions remain owned by the selected client; the launcher is not an OS sandbox.\n",
        shell_quote(&home.join("launch.sh").to_string_lossy()),
        config.notes()
    ));
    if matches!(config.client, Client::Codex | Client::Hermes) {
        crate::diagnostics::stderr_line(format_args!(
            "This client owns generation output limits; --max-output-tokens sets a local reservation, not a guaranteed request cap."
        ));
    }
    if config.client == Client::Hermes && config.context_window.unwrap() < 64_000 {
        crate::diagnostics::stderr_line(format_args!(
            "Hermes uses its LM Studio compatibility profile to honor an explicit context below its custom-provider floor of 64000; the selected endpoint and model remain exact."
        ));
    }
    Ok(())
}
