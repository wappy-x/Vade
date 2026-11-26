//! `vade connect <client>`: read `/v1/capabilities` from the running engine
//! and generate (or launch) an isolated client setup via `vade-harness setup`.
use crate::evals::json::{self, Value};
use std::{
    fs,
    io::{BufRead, BufReader, IsTerminal, Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::Duration,
};

const CLIENTS: &[&str] = &["claude", "codex", "opencode", "hermes", "pi"];
const MAX_CAPABILITIES_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Options {
    client: String,
    url: Option<String>,
    target: String,
    write: Option<PathBuf>,
    client_bin: Option<String>,
    print: bool,
    no_launch: bool,
    client_args: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Capabilities {
    model: String,
    context_limit: u64,
    output_limit: u64,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let usage = "Use vade connect <claude|codex|opencode|hermes|pi> [--print] [--dry-run] [--url URL]. Run vade connect --help.";
    let Some((first, rest)) = args.split_first() else {
        return Err(format!("Name the client to connect. {usage}"));
    };
    if first.starts_with('-') {
        return Err(format!("Name the client before options, not {}. {usage}", crate::text::argument(first)));
    }
    let client = first.to_ascii_lowercase();
    if !CLIENTS.contains(&client.as_str()) {
        return Err(format!(
            "Unknown client {}. Choose claude, codex, opencode, hermes or pi. Run vade connect --help.",
            crate::text::argument(first)
        ));
    }
    let mut options = Options {
        client,
        url: None,
        target: "direct".into(),
        write: None,
        client_bin: None,
        print: false,
        no_launch: false,
        client_args: Vec::new(),
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut index = 0;
    while index < rest.len() {
        let arg = rest[index].as_str();
        if arg == "--" {
            options.client_args = rest[index + 1..].to_vec();
            break;
        }
        if !seen.insert(arg) {
            return Err(format!("Use {arg} only once. Run vade connect --help."));
        }
        match arg {
            "--print" => {
                options.print = true;
                index += 1;
            }
            "--no-launch" => {
                options.no_launch = true;
                index += 1;
            }
            "--url" | "--target" | "--write" | "--client-bin" => {
                let value = rest
                    .get(index + 1)
                    .filter(|v| !v.is_empty() && !v.starts_with("--"))
                    .ok_or_else(|| format!("Give {arg} a value. Run vade connect --help."))?;
                match arg {
                    "--url" => options.url = Some(value.clone()),
                    "--target" => {
                        if !matches!(value.as_str(), "direct" | "gateway") {
                            return Err("Set --target to direct or gateway. Run vade connect --help.".into());
                        }
                        options.target = value.clone();
                    }
                    "--write" => options.write = Some(PathBuf::from(value)),
                    "--client-bin" => options.client_bin = Some(value.clone()),
                    _ => unreachable!(),
                }
                index += 2;
            }
            _ => return Err(format!("Unknown option {} for vade connect. Run vade connect --help.", crate::text::argument(arg))),
        }
    }
    if options.print && (options.write.is_some() || options.client_bin.is_some() || !options.client_args.is_empty()) {
        return Err("Use --print without --write, --client-bin or client arguments. Run vade connect --help.".into());
    }
    Ok(options)
}

/// Run `vade connect <client> [OPTIONS]`.
pub fn run(args: &[String], dry: bool) -> Result<(), String> {
    let options = parse(args)?;
    let (url, _) = crate::endpoint::resolve(options.url.clone(), false)?;
    let base = normalize_origin(&url)?;
    crate::auth::validate(false)?;
    let bearer = crate::auth::bearer_key()?;
    let caps = fetch_capabilities(&base, bearer)?;
    let max_output = caps.output_limit.min(caps.context_limit.saturating_sub(1)).max(1);
    if options.print {
        let setup_args = print_args(&options, &base, &caps.model, caps.context_limit, max_output);
        if dry {
            return print_dry(&base, &caps, max_output, &setup_args, None);
        }
        return run_setup(setup_args);
    }
    let home = match &options.write {
        Some(path) => path.clone(),
        None => default_home(&options.client, dry)?,
    };
    let setup_args = home_args(&options, &base, &caps.model, &home, caps.context_limit, max_output);
    if dry {
        return print_dry(&base, &caps, max_output, &setup_args, Some(&home));
    }
    if options.write.is_none() && let Some(parent) = home.parent() {
        crate::text::private_dir(parent)?;
    }
    run_setup(setup_args)?;
    let launcher = home.join("launch.sh");
    let shown = crate::runner::quote(&launcher.to_string_lossy());
    if options.no_launch || !std::io::stdin().is_terminal() {
        return crate::output::line(format_args!(
            "Ready for {} ({}, context {}, output {}).\nRun from your workspace:\n  {shown}",
            options.client, caps.model, caps.context_limit, max_output
        ));
    }
    let binary = options.client_bin.as_deref().unwrap_or(&options.client);
    if !command_exists(binary) {
        return crate::output::line(format_args!(
            "Dedicated home created for {} ({}, context {}, output {}), but {} is not on PATH.\nInstall {}, then run from your workspace:\n  {shown}",
            options.client, caps.model, caps.context_limit, max_output,
            crate::text::quoted(binary), options.client,
        ));
    }
    let status = Command::new(&launcher)
        .args(&options.client_args)
        .status()
        .map_err(|error| format!("Cannot start {}: {error}", launcher.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{} exited with status {}.", options.client, status.code().unwrap_or(1)))
    }
}

fn print_dry(base: &str, caps: &Capabilities, max_output: u64, setup_args: &[String], home: Option<&Path>) -> Result<(), String> {
    let quoted_args: Vec<String> = setup_args.iter().map(|arg| crate::runner::quote(arg)).collect();
    crate::output::line(format_args!("Endpoint: {base}"))?;
    crate::output::line(format_args!(
        "Capabilities: model={}, context_limit={}, output_limit={}",
        crate::text::terminal_text(&caps.model), caps.context_limit, max_output
    ))?;
    if let Some(home) = home {
        crate::output::line(format_args!("Dedicated home: {}", crate::text::terminal_text(&home.to_string_lossy())))?;
    }
    crate::output::line(format_args!("Command: vade-harness setup {}", quoted_args.join(" ")))?;
    crate::output::line(format_args!("Dry run: no files written and no client launched."))
}

fn print_args(options: &Options, base: &str, model: &str, context: u64, output: u64) -> Vec<String> {
    let mut args = vec![
        options.client.clone(),
        "--endpoint".into(),
        base.into(),
        "--model".into(),
        model.into(),
        "--target".into(),
        options.target.clone(),
    ];
    if options.client == "codex" {
        args.extend([
            "--context-window".into(),
            context.to_string(),
            "--max-output-tokens".into(),
            output.to_string(),
        ]);
    }
    args
}

fn home_args(options: &Options, base: &str, model: &str, home: &Path, context: u64, output: u64) -> Vec<String> {
    let mut args = vec![
        options.client.clone(),
        "--endpoint".into(),
        base.into(),
        "--model".into(),
        model.into(),
        "--target".into(),
        options.target.clone(),
        "--dedicated-home".into(),
        "--write".into(),
        home.to_string_lossy().into_owned(),
        "--context-window".into(),
        context.to_string(),
        "--max-output-tokens".into(),
        output.to_string(),
    ];
    if let Some(bin) = &options.client_bin {
        args.extend(["--client-bin".into(), bin.clone()]);
    }
    args
}

fn run_setup(args: Vec<String>) -> Result<(), String> {
    if vade_harness::setup::cli(args) == ExitCode::SUCCESS {
        Ok(())
    } else {
        Err("vade-harness setup failed; see the message above.".into())
    }
}

fn default_home(client: &str, dry: bool) -> Result<PathBuf, String> {
    let root = crate::launch::root()?;
    let clients = root.join("clients");
    let parent = if dry {
        clients
    } else {
        crate::text::private_dir(&clients)?;
        fs::canonicalize(&clients).map_err(|_| "Cannot resolve the Vade clients folder.")?
    };
    Ok(parent.join(format!("{client}-{}", crate::text::random_hex(6)?)))
}

fn command_exists(binary: &str) -> bool {
    let path = Path::new(binary);
    if binary.contains('/') {
        return path.is_file();
    }
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| dir.join(binary).is_file())
    })
}

fn normalize_origin(url: &str) -> Result<String, String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or("vade connect needs an http:// engine URL.")?;
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    if authority.is_empty()
        || authority.bytes().any(|c| c.is_ascii_control() || c.is_ascii_whitespace())
        || authority.contains(['@', '?', '#', '\\'])
    {
        return Err("vade connect needs an engine URL without credentials or a query.".into());
    }
    let trimmed = path.trim_end_matches('/');
    let trimmed = trimmed.strip_suffix("/v1").or_else(|| trimmed.strip_suffix("v1")).unwrap_or(trimmed);
    let trimmed = trimmed.trim_end_matches('/');
    Ok(if trimmed.is_empty() {
        format!("http://{authority}")
    } else {
        format!("http://{authority}/{trimmed}")
    })
}

fn fetch_capabilities(origin: &str, bearer: Option<&str>) -> Result<Capabilities, String> {
    let rest = origin.strip_prefix("http://").ok_or("Invalid engine URL.")?;
    let (authority, prefix) = rest.split_once('/').unwrap_or((rest, ""));
    let path = if prefix.is_empty() {
        "/v1/capabilities".to_owned()
    } else {
        format!("/{prefix}/v1/capabilities")
    };
    let address = socket_address(authority)?;
    let addr = address
        .to_socket_addrs()
        .map_err(|_| format!("Cannot resolve engine address {authority}."))?
        .next()
        .ok_or_else(|| format!("Cannot resolve engine address {authority}."))?;
    let timeout = Duration::from_secs(5);
    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .map_err(|_| format!("Cannot reach the engine at {origin}. Start it with vade up, then retry vade connect."))?;
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    let mut req = format!("GET {path} HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\n");
    if let Some(key) = bearer {
        req.push_str(&format!("Authorization: Bearer {key}\r\n"));
    }
    req.push_str("\r\n");
    stream
        .write_all(req.as_bytes())
        .map_err(|_| format!("Cannot send capabilities request to {origin}."))?;
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader
        .read_line(&mut status_line)
        .map_err(|_| format!("No response from {origin}{path}."))?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| format!("Invalid HTTP response from {origin}{path}."))?;
    let mut chunked = false;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|_| format!("Incomplete headers from {origin}{path}."))?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("transfer-encoding")
            && value.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
        }
    }
    let mut raw = Vec::new();
    reader
        .take(MAX_CAPABILITIES_BYTES as u64 + 1)
        .read_to_end(&mut raw)
        .map_err(|_| format!("Cannot read capabilities response from {origin}{path}."))?;
    if raw.len() > MAX_CAPABILITIES_BYTES {
        return Err(format!("Capabilities response from {origin}{path} exceeds 1 MiB."));
    }
    let body = if chunked { decode_chunked(&raw)? } else { raw };
    if status != 200 {
        return Err(format!("Engine at {origin}{path} answered HTTP {status}. Check that the engine is ready with vade status."));
    }
    let text = String::from_utf8(body).map_err(|_| "Capabilities response is not valid UTF-8.")?;
    parse_capabilities(&text)
}

fn socket_address(authority: &str) -> Result<String, String> {
    if authority.starts_with('[') {
        let end = authority.find(']').ok_or("Invalid IPv6 engine URL.")?;
        let rest = &authority[end + 1..];
        Ok(if rest.is_empty() { format!("{authority}:80") } else { authority.to_owned() })
    } else if authority.contains(':') {
        Ok(authority.to_owned())
    } else {
        Ok(format!("{authority}:80"))
    }
}

fn decode_chunked(raw: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut rest = raw;
    loop {
        let Some(pos) = rest.windows(2).position(|w| w == b"\r\n") else {
            return Err("Truncated chunked capabilities response.".into());
        };
        let size_str = std::str::from_utf8(&rest[..pos])
            .map_err(|_| "Invalid chunk size in capabilities response.")?
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        let size = usize::from_str_radix(size_str, 16)
            .map_err(|_| "Invalid chunk size in capabilities response.")?;
        rest = &rest[pos + 2..];
        if size == 0 {
            break;
        }
        if rest.len() < size + 2 || &rest[size..size + 2] != b"\r\n" {
            return Err("Truncated chunk body in capabilities response.".into());
        }
        out.extend_from_slice(&rest[..size]);
        rest = &rest[size + 2..];
    }
    Ok(out)
}

fn parse_capabilities(body: &str) -> Result<Capabilities, String> {
    let value = json::parse(body).map_err(|_| "Engine /v1/capabilities did not return valid JSON.")?;
    let model = value["model"]
        .as_str()
        .filter(|m| !m.is_empty())
        .ok_or("Engine /v1/capabilities is missing a valid model ID.")?
        .to_owned();
    let positive = |key: &str| -> Result<u64, String> {
        match &value[key] {
            Value::Int(n) if *n > 0 => Ok(*n as u64),
            Value::Float(n) if *n > 0.0 && n.fract() == 0.0 => Ok(*n as u64),
            _ => Err(format!("Engine /v1/capabilities is missing a positive {key}.")),
        }
    };
    let context_limit = positive("context_limit")?;
    let output_limit = positive("output_limit")?;
    Ok(Capabilities { model, context_limit, output_limit })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn mock_server(body: &'static str) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 2048];
                let _ = stream.read(&mut buf);
                let reply = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        (format!("http://{addr}"), handle)
    }

    #[test]
    fn parses_supported_clients_and_rejects_bad_arguments() {
        for client in CLIENTS {
            let parsed = parse(&[client.to_ascii_uppercase()]).unwrap();
            assert_eq!((parsed.client.as_str(), parsed.target.as_str(), parsed.print), (*client, "direct", false));
        }
        let parsed = parse(&[
            "codex", "--url", "http://127.0.0.1:18080", "--target", "gateway", "--no-launch", "--", "exec", "hi",
        ].map(str::to_owned)).unwrap();
        assert_eq!(parsed.target, "gateway");
        assert!(parsed.no_launch);
        assert_eq!(parsed.client_args, ["exec", "hi"]);
        for bad in [
            &[][..],
            &["--print"],
            &["unknown"],
            &["codex", "--bogus"],
            &["codex", "--target", "both"],
            &["codex", "--print", "--write", "/tmp/x"],
            &["codex", "--print", "--print"],
        ] {
            assert!(parse(&bad.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn dry_run_and_dedicated_home_read_live_capabilities_and_generate_exact_config() {
        let body = r#"{"object":"engine.capabilities","model":"inkling-2-mini","context_limit":131072,"output_limit":16384}"#;
        let (url, server) = mock_server(body);
        let capture = crate::output::testing::Capture::default();
        let res = crate::output::with_writer(capture.clone(), || {
            run(&["codex".into(), "--url".into(), url.clone()], true)
        });
        server.join().unwrap();
        res.unwrap();
        let printed = String::from_utf8(capture.bytes.borrow().clone()).unwrap();
        assert!(printed.contains("model=inkling-2-mini, context_limit=131072, output_limit=16384"), "{printed}");
        assert!(printed.contains("vade-harness setup codex --endpoint "), "{printed}");
        assert!(printed.contains("Dry run: no files written and no client launched."), "{printed}");

        // Real dedicated home generation with --no-launch writes exact Inkling config.
        let base = fs::canonicalize(std::env::temp_dir()).unwrap()
            .join(format!("vade-connect-test-{}", crate::text::random_hex(8).unwrap()));
        fs::create_dir_all(&base).unwrap();
        let dest = base.join("codex-home");
        let (url2, server2) = mock_server(body);
        let capture2 = crate::output::testing::Capture::default();
        let res2 = crate::output::with_writer(capture2.clone(), || {
            run(&["codex".into(), "--url".into(), url2, "--write".into(), dest.to_string_lossy().into_owned(), "--no-launch".into()], false)
        });
        server2.join().unwrap();
        res2.unwrap();
        let config_toml = fs::read_to_string(dest.join("config.toml")).unwrap();
        assert!(config_toml.contains("model = \"inkling-2-mini\""), "{config_toml}");
        assert!(config_toml.contains("model_context_window = 131072"), "{config_toml}");
        assert!(config_toml.contains("model_auto_compact_token_limit = 114688"), "{config_toml}");
        assert!(dest.join("launch.sh").is_file());

        // 512K Paged-KV Inkling server defaults Codex auto-compact to 307200 (300K).
        let body512 = r#"{"object":"engine.capabilities","model":"inkling-2-mini","context_limit":524288,"output_limit":16384}"#;
        let (url3, server3) = mock_server(body512);
        let dest512 = base.join("codex-home-512k");
        let capture3 = crate::output::testing::Capture::default();
        let res3 = crate::output::with_writer(capture3.clone(), || {
            run(&["codex".into(), "--url".into(), url3, "--write".into(), dest512.to_string_lossy().into_owned(), "--no-launch".into()], false)
        });
        server3.join().unwrap();
        res3.unwrap();
        let config_512 = fs::read_to_string(dest512.join("config.toml")).unwrap();
        assert!(config_512.contains("model_context_window = 524288"), "{config_512}");
        assert!(config_512.contains("model_auto_compact_token_limit = 307200"), "{config_512}");
        let models_512 = fs::read_to_string(dest512.join("models.json")).unwrap();
        assert!(models_512.contains("\"context_window\":524288"), "{models_512}");
        assert!(models_512.contains("\"auto_compact_token_limit\":307200"), "{models_512}");

        // --print dry-run passes verified context and output limits to setup codex.
        let (url4, server4) = mock_server(body512);
        let capture4 = crate::output::testing::Capture::default();
        let res4 = crate::output::with_writer(capture4.clone(), || {
            run(&["codex".into(), "--print".into(), "--url".into(), url4], true)
        });
        server4.join().unwrap();
        res4.unwrap();
        let printed4 = String::from_utf8(capture4.bytes.borrow().clone()).unwrap();
        assert!(printed4.contains("--context-window 524288 --max-output-tokens 16384"), "{printed4}");
        fs::remove_dir_all(&base).unwrap();
    }
}
