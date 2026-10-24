//! Offline, original configuration templates based on the credited client docs.
//! No client configuration, credentials, environment, or network is read.
use crate::{json, transport::Endpoint};
use std::{
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    process::ExitCode,
};

mod home;

pub const HELP: &str = "Vade client setup (offline; prints configuration only)

Usage: vade-harness setup <codex|hermes|opencode|pi|generic>
         --endpoint URL --model NAME [OPTIONS]

  --endpoint URL       Engine HTTP origin or API base (required)
  --model NAME         Exact upstream model ID (required; no discovery)
  --gateway URL        Gateway API base (default http://127.0.0.1:8787/v1)
  --target TARGET      direct, gateway, or both (default both)
  --write PATH         Create a new file containing one selected configuration;
                       requires --target direct or gateway; never overwrites
  --dedicated-home     With --write DIR, create a new private client home,
                       instructions and launch.sh; refuses existing directories
  --context-window N   Verified total context tokens (home mode; required)
  --max-output-tokens N Verified output reservation (home mode; required)
  --client-bin PATH    Client executable for launch.sh (home mode only)
  -h, --help           Show this help

Default stdout contains both configurations with instructions and sources.
With one --target, stdout is the client-ready configuration; notes go to stderr.
Without --dedicated-home, --write creates only the explicitly named private file.
Home mode creates only the requested new directory below an existing parent.
No client is launched and no real client config or credentials are read.
HTTP only; no credentials, query strings, fragments, or TLS support.
Codex requires upstream Responses support; the gateway does not convert APIs.
";

const CODEX_SOURCE: &str = "https://learn.chatgpt.com/docs/config-file/config-reference";
const HERMES_SOURCE: &str = "https://github.com/NousResearch/hermes-agent/blob/25a71a744cb9ef06950a91638e6229b4f808d461/website/docs/integrations/providers.md#custom--self-hosted-llm-providers";
const OPENCODE_SOURCE: &str = "https://github.com/anomalyco/opencode/blob/5a8c1d81b7ba527e580191e1d9638c09c6f79419/packages/web/src/content/docs/providers.mdx#custom-provider";
const PI_SOURCE: &str = "https://github.com/earendil-works/pi/blob/ce950d78f424dcaf9f5d6a03ce80ab141130eb1d/packages/coding-agent/docs/models.md#configure-a-compatible-endpoint";
const GENERIC_SOURCE: &str = "https://platform.openai.com/docs/api-reference/chat/create";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Client {
    Codex,
    Hermes,
    OpenCode,
    Pi,
    Generic,
}
impl Client {
    fn parse(value: &str) -> Result<Self, &'static str> {
        match value {
            "codex" => Ok(Self::Codex),
            "hermes" => Ok(Self::Hermes),
            "opencode" => Ok(Self::OpenCode),
            "pi" => Ok(Self::Pi),
            "generic" => Ok(Self::Generic),
            _ => Err("setup client must be codex, hermes, opencode, pi, or generic"),
        }
    }
    fn source(self) -> &'static str {
        match self {
            Self::Codex => CODEX_SOURCE,
            Self::Hermes => HERMES_SOURCE,
            Self::OpenCode => OPENCODE_SOURCE,
            Self::Pi => PI_SOURCE,
            Self::Generic => GENERIC_SOURCE,
        }
    }
    fn format(self) -> &'static str {
        match self {
            Self::Codex => "toml",
            Self::Hermes => "yaml",
            Self::OpenCode | Self::Pi => "json",
            Self::Generic => "sh",
        }
    }
    fn instructions(self, model: &str) -> String {
        match self {
            Self::Codex => "Codex config.toml provider settings. Requires /v1/responses on the upstream. HTTP/SSE only; OpenAI authentication and WebSockets are disabled for this provider.".into(),
            Self::Hermes => "Hermes config.yaml model settings. Uses the documented custom provider over Chat Completions. The literal no-key-required is a dummy value for a keyless endpoint, not a credential.".into(),
            Self::OpenCode => "OpenCode opencode.json provider settings. Uses @ai-sdk/openai-compatible (Chat Completions); the model is selected in this file. The literal no-key-required is a dummy value for a keyless endpoint.".into(),
            Self::Pi => format!("Pi models.json provider settings. Select with: pi --provider vade --model {}. Uses openai-completions (Chat Completions). The dummy apiKey makes this keyless provider available to Pi; it is not a credential.", shell_quote(model)),
            Self::Generic => "A POSIX-shell curl request using Chat Completions; no SDK or configuration file required. Running the printed command sends one synthetic request. curl -q ignores curl's configuration files.".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Direct,
    Gateway,
    Both,
}

#[derive(Debug)]
struct Config {
    client: Client,
    endpoint: String,
    gateway: String,
    model: String,
    target: Target,
    write: Option<PathBuf>,
    dedicated_home: bool,
    context_window: Option<u64>,
    max_output_tokens: Option<u64>,
    client_bin: Option<String>,
}

/// Restrict input to literal identifiers: client loaders can interpolate ${...},
/// {env:...}, {file:...}, or execute !commands even inside quoted strings.
pub(crate) fn valid_model(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 512
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:/+-".contains(&b))
        && value.as_bytes()[0].is_ascii_alphanumeric()
}

/// Derive one unambiguous API base, including reverse-proxy prefixes. Never
/// guess the meaning of a custom exact route or retain credentials/query data.
pub(crate) fn api_base(url: &str) -> Result<String, &'static str> {
    if url.starts_with("https://") {
        return Err("HTTPS is unsupported; use an HTTP endpoint");
    }
    if url.contains(['?', '#', '@']) {
        return Err("endpoint credentials, queries, and fragments are unsupported");
    }
    let endpoint = Endpoint::parse(url).map_err(|_| "invalid HTTP endpoint")?;
    if !endpoint
        .path
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"/-._~".contains(&b))
        || endpoint.path.split('/').any(|s| s == "." || s == "..")
        || endpoint.path.contains("//")
    {
        return Err(
            "endpoint must contain a literal API path without interpolation or dot segments",
        );
    }
    let path = endpoint.path.trim_end_matches('/');
    let base = path
        .strip_suffix("/chat/completions")
        .or_else(|| path.strip_suffix("/responses"))
        .or_else(|| path.strip_suffix("/capabilities"))
        .unwrap_or(path);
    if !base.ends_with("/v1") && !base.is_empty() {
        return Err("use an HTTP origin, /v1 API base, or known chat/responses route");
    }
    let base = format!("http://{}{base}", endpoint.authority);
    // Reserve room for the longest generated route, including origin-only URLs
    // whose parser supplies an implicit /v1 suffix.
    Endpoint::parse(&format!("{base}/chat/completions"))
        .map_err(|_| "normalized HTTP endpoint exceeds size limit")?;
    Ok(base)
}

impl Config {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        crate::diagnostics::parse_command("setup", |option| Self::parse_options(args, option))
    }
    fn parse_options(args: impl IntoIterator<Item = String>, option: &mut Option<String>) -> Result<Self, &'static str> {
        let mut args = args.into_iter();
        let client = Client::parse(
            &args
                .next()
                .ok_or("setup needs a client; see setup --help")?,
        )?;
        let mut endpoint = None;
        let mut model = None;
        let mut gateway = "http://127.0.0.1:8787/v1".to_owned();
        let mut target = Target::Both;
        let mut write = None;
        let mut dedicated_home = false;
        let mut context_window = None;
        let mut max_output_tokens = None;
        let mut client_bin = None;
        let mut seen = Vec::new();
        while let Some(flag) = args.next() {
            *option = crate::diagnostics::option_name(&flag).map(str::to_owned);
            if seen.contains(&flag) {
                return Err("duplicate setup option");
            }
            if flag == "--dedicated-home" {
                dedicated_home = true;
                seen.push(flag);
                continue;
            }
            let value = args
                .next()
                .ok_or("setup option needs a value; see setup --help")?;
            match flag.as_str() {
                "--endpoint" => endpoint = Some(api_base(&value)?),
                "--gateway" => gateway = api_base(&value)?,
                "--model" => {
                    if !valid_model(&value) {
                        return Err(
                            "model must be a literal ID (letters, digits, ._:/+-; start with a letter or digit)",
                        );
                    }
                    model = Some(value);
                }
                "--target" => {
                    target = match value.as_str() {
                        "direct" => Target::Direct,
                        "gateway" => Target::Gateway,
                        "both" => Target::Both,
                        _ => return Err("setup target must be direct, gateway, or both"),
                    }
                }
                "--write" if !value.is_empty() => write = Some(PathBuf::from(value)),
                "--write" => return Err("setup output path must not be empty"),
                "--context-window" | "--max-output-tokens" => {
                    let count = value
                        .parse::<u64>()
                        .ok()
                        .filter(|n| (1..=1_048_576).contains(n))
                        .ok_or("setup token limits must be integers in 1..1048576")?;
                    if flag == "--context-window" {
                        context_window = Some(count);
                    } else {
                        max_output_tokens = Some(count);
                    }
                }
                "--client-bin" if !value.is_empty() && !value.chars().any(char::is_control) => {
                    let path = PathBuf::from(&value);
                    client_bin = Some(if path.is_relative() && value.contains('/') {
                        std::env::current_dir()
                            .map_err(|_| "cannot resolve client executable")?
                            .join(path)
                            .into_os_string()
                            .into_string()
                            .map_err(|_| "client executable needs a UTF-8 path")?
                    } else {
                        value
                    });
                }
                "--client-bin" => return Err("client executable must be a nonempty path without control characters"),
                _ => return Err("unknown setup option; see setup --help"),
            }
            seen.push(flag);
        }
        *option = None;
        if write.is_some() && target == Target::Both {
            return Err(
                "--write needs --target direct or gateway to produce one client-ready file",
            );
        }
        if dedicated_home {
            if write.is_none() || client == Client::Generic {
                return Err("--dedicated-home requires --write DIR and a supported client");
            }
            if context_window.is_none() || max_output_tokens.is_none() {
                return Err(
                    "dedicated home requires verified --context-window and --max-output-tokens",
                );
            }
        } else if client_bin.is_some() || context_window.is_some() || max_output_tokens.is_some() {
            return Err("client executable and token limits require --dedicated-home");
        }
        if context_window
            .zip(max_output_tokens)
            .is_some_and(|(c, o)| o >= c)
        {
            return Err("output reservation must be smaller than the context window");
        }
        Ok(Self {
            client,
            endpoint: endpoint.ok_or("setup requires --endpoint")?,
            model: model.ok_or("setup requires --model")?,
            gateway,
            target,
            write,
            dedicated_home,
            context_window,
            max_output_tokens,
            client_bin,
        })
    }

    fn snippet(&self, endpoint: &str) -> String {
        let model = quoted(&self.model);
        let base = quoted(endpoint);
        match self.client {
            Client::Codex => format!(
                "model = {model}\nmodel_provider = \"vade\"\n\n[model_providers.vade]\nname = \"Vade\"\nbase_url = {base}\nwire_api = \"responses\"\nrequires_openai_auth = false\nsupports_websockets = false\n"
            ),
            Client::Hermes => format!(
                "model:\n  default: {model}\n  provider: custom\n  base_url: {base}\n  api_key: \"no-key-required\"\n"
            ),
            Client::OpenCode => format!(
                "{{\n  \"$schema\": \"https://opencode.ai/config.json\",\n  \"model\": {},\n  \"provider\": {{\n    \"vade\": {{\n      \"npm\": \"@ai-sdk/openai-compatible\",\n      \"name\": \"Vade\",\n      \"options\": {{\"baseURL\": {base}, \"apiKey\": \"no-key-required\"}},\n      \"models\": {{{model}: {{\"name\": {model}}}}}\n    }}\n  }}\n}}\n",
                quoted(&format!("vade/{}", self.model))
            ),
            Client::Pi => format!(
                "{{\n  \"providers\": {{\n    \"vade\": {{\n      \"baseUrl\": {base},\n      \"api\": \"openai-completions\",\n      \"apiKey\": \"no-key-required\",\n      \"models\": [{{\"id\": {model}}}]\n    }}\n  }}\n}}\n"
            ),
            Client::Generic => {
                let body = format!(
                    "{{\"model\":{model},\"messages\":[{{\"role\":\"user\",\"content\":\"Reply with OK.\"}}],\"max_tokens\":32,\"stream\":false}}"
                );
                format!(
                    "curl -q --fail-with-body --max-time 60 {} \\\n  -H 'Content-Type: application/json' \\\n  --data-binary {}\n",
                    shell_quote(&format!("{endpoint}/chat/completions")),
                    shell_quote(&body)
                )
            }
        }
    }

    fn notes(&self) -> String {
        format!(
            "{}\nSource: {}\nSetup does not connect, discover limits, launch clients, or read/write installed client configuration.\nThe upstream must implement the selected API; the gateway forwards without translation.\n",
            self.client.instructions(&self.model),
            self.client.source()
        )
    }
    fn output(&self) -> String {
        match self.target {
            Target::Direct => self.snippet(&self.endpoint),
            Target::Gateway => self.snippet(&self.gateway),
            Target::Both => format!(
                "Vade setup\n\n{}\n(a) Client -> Vade Engine directly\n```{}\n{}```\n\n(b) Client -> Vade gateway -> Engine\nStart the gateway separately (the gateway must forward to the chosen Engine):\n  vade-harness gateway --upstream {}\nThe gateway defaults to 127.0.0.1:8787; a custom --gateway URL needs a separately configured listener/proxy.\n```{}\n{}```\n",
                self.notes(),
                self.client.format(),
                self.snippet(&self.endpoint),
                shell_quote(&self.endpoint),
                self.client.format(),
                self.snippet(&self.gateway)
            ),
        }
    }
}

fn quoted(value: &str) -> String {
    let mut output = String::new();
    json::write_string(value, &mut output);
    output
}
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

pub fn cli(args: impl IntoIterator<Item = String>) -> ExitCode {
    let args: Vec<_> = args.into_iter().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return crate::diagnostics::print_stdout(HELP);
    }
    let config = match Config::parse(args) {
        Ok(config) => config,
        Err(error) => {
            crate::diagnostics::stderr_line(format_args!("vade: {error}"));
            return ExitCode::from(2);
        }
    };
    if config.dedicated_home {
        return match home::write(&config) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                crate::diagnostics::stderr_line(format_args!("vade: {error}"));
                ExitCode::FAILURE
            }
        };
    }
    let output = config.output();
    if let Some(path) = &config.write {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        // create_new refuses existing files and final symlinks; no config search,
        // merge, truncation, home-directory lookup, or parent-directory creation.
        let result = options.open(path).and_then(|mut file| {
            file.write_all(output.as_bytes())?;
            file.sync_all()
        });
        if let Err(error) = result {
            crate::diagnostics::stderr_line(format_args!(
                "vade: cannot create/write setup file. {}",
                crate::diagnostics::output_file_hint(&error)
            ));
            return ExitCode::FAILURE;
        }
        crate::diagnostics::stderr_line(format_args!(
            "Setup configuration saved to the requested new file.\n{}",
            config.notes()
        ));
    } else {
        if config.target != Target::Both {
            crate::diagnostics::stderr(format_args!("{}", config.notes()));
        }
        return crate::diagnostics::print_stdout(&output);
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn literal_bases_are_normalized_without_network() {
        for url in [
            "http://127.0.0.1:1234",
            "http://127.0.0.1:1234/",
            "http://127.0.0.1:1234/v1/",
            "http://127.0.0.1:1234/v1/chat/completions",
            "http://127.0.0.1:1234/v1/responses",
        ] {
            assert_eq!(api_base(url).unwrap(), "http://127.0.0.1:1234/v1");
        }
        assert_eq!(
            api_base("http://[::1]:8080/proxy/v1/responses").unwrap(),
            "http://[::1]:8080/proxy/v1"
        );
        for url in [
            "https://example.invalid/v1",
            "http://u:p@example.invalid/v1",
            "http://example.invalid/v1?key=x",
            "http://example.invalid/#x",
            "http://example.invalid/${HOST}/v1",
            "http://example.invalid/a/../v1",
            "http://example.invalid/unknown",
            "http://example.invalid/a%0ab/v1",
            "http://example.invalid/a//v1",
        ] {
            assert!(api_base(url).is_err(), "{url}");
        }
    }
    #[test]
    fn reject_client_interpolation_and_ambiguous_write() {
        for model in [
            "",
            "$(id)",
            "!command",
            "${MODEL}",
            "{env:MODEL}",
            "{file:secrets}",
            "m\n",
            "m\u{1b}",
            "--arg",
        ] {
            assert!(!valid_model(model));
        }
        for model in ["mock", "org/model:Q4_K_M", "model-1.2+3"] {
            assert!(valid_model(model));
        }
        assert!(
            Config::parse(
                [
                    "pi",
                    "--endpoint",
                    "http://example.invalid",
                    "--model",
                    "mock",
                    "--write",
                    "models.json"
                ]
                .map(str::to_owned)
            )
            .is_err()
        );
    }
}
