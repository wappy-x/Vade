//! Endpoint selection shared by client commands and the local lifecycle.
use std::{io::Read, path::Path};

/// Clients require an explicitly selected or recorded endpoint so they cannot
/// accidentally contact an unrelated port.
pub(crate) fn resolve(
    explicit_url: Option<String>,
    load_controller: bool,
) -> Result<(String, Option<crate::config::Config>), String> {
    let explicit_url = explicit_url.or_else(|| std::env::var("VADE_API_URL").ok());
    // Match lifecycle selection: saved owner defaults precede the older
    // qwen3-8b setup and an implicit controller config. Explicit selectors win.
    let owner_root = if explicit_url.is_none()
        && std::env::var_os("VADE_CONFIG").is_none()
        && crate::launch::owner_configured()
    {
        Some(crate::launch::root()?)
    } else {
        None
    };
    let local_root = if explicit_url.is_none()
        && owner_root.is_none()
        && std::env::var_os("VADE_CONFIG").is_none()
    {
        let root = crate::run::home()?;
        crate::run::handles(&crate::cli::Command::Status, &root).then_some(root)
    } else {
        None
    };
    let config = if load_controller
        || (explicit_url.is_none()
            && owner_root.is_none()
            && local_root.is_none()
            && (std::env::var_os("VADE_CONFIG").is_some()
                || crate::config::config_path()?.exists()))
    {
        Some(crate::config::Config::load()?)
    } else {
        None
    };
    let url = match explicit_url {
        Some(url) => url,
        None if let Some(root) = owner_root.as_deref() => owner_endpoint(root)?,
        None => {
            let port = match (&local_root, &config) {
                (None, Some(config)) => config.local_port,
                _ => {
                    let record = local_root.unwrap_or(crate::run::home()?).join("run.conf");
                    recorded_port(&record)?.ok_or("No recorded local server endpoint; run vade launch (or vade run qwen3-8b for legacy setup) or supply --url URL.")?
                }
            };
            format!("http://127.0.0.1:{port}")
        }
    };
    Ok((url, config))
}

fn recorded_port(path: &Path) -> Result<Option<u16>, String> {
    let failure = "Cannot read local engine state; run vade doctor or supply --url.";
    let file = match crate::unix_file::AnchoredPath::new(path)
        .and_then(|path| path.open(crate::unix_file::OpenMode::Read))
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(failure.into()),
    };
    if !file.metadata().map_err(|_| failure)?.is_file() { return Err(failure.into()); }
    let mut record = String::new();
    file.take(65537).read_to_string(&mut record).map_err(|_| failure)?;
    if record.len() > 65536 { return Err(failure.into()); }
    crate::text::parse(&record).map_err(|error| format!("{error} Run vade doctor or supply --url."))?
        .get("port")
        .and_then(|port| port.parse::<u16>().ok())
        .filter(|port| *port != 0)
        .map(Some)
        .ok_or_else(|| "Invalid local engine port; run vade doctor or supply --url.".into())
}

fn owner_endpoint(root: &Path) -> Result<String, String> {
    let read = |path: &Path| -> Result<crate::text::Fields, String> {
        // Never follow links or block on a FIFO while discovering an endpoint.
        let file = crate::unix_file::AnchoredPath::new(path)
            .and_then(|path| path.open(crate::unix_file::OpenMode::Read))
            .map_err(|_| "Cannot read local owner state; run vade status or supply --url.")?;
        if !file.metadata().map_err(|_| "Cannot inspect local owner state.")?.is_file() {
            return Err("Local owner state must be a regular file; run vade status or supply --url.".into());
        }
        let mut data = String::new();
        file.take(65537).read_to_string(&mut data)
            .map_err(|_| "Cannot read local owner state; run vade status or supply --url.")?;
        if data.len() > 65536 {
            return Err("Local owner state exceeds 64 KiB; run vade status or supply --url.".into());
        }
        crate::text::parse(&data)
    };
    let config = read(&root.join("config"))?;
    let record = read(&root.join("state/run.conf"))?;
    let port = record.get("port").and_then(|s| s.parse::<u16>().ok())
        .filter(|port| *port >= 1024);
    if config.get("format").map(String::as_str) != Some("vade-owner-v1")
        || record.get("version").map(String::as_str) != Some("1")
        || record.get("phase").map(String::as_str) != Some("running")
        || !record.get("token").is_some_and(|token| crate::text::hex(token, 64))
        || port.is_none()
        || config.get("port") != record.get("port")
    {
        return Err("Local owner state has no consistent running endpoint; run vade status or supply --url.".into());
    }
    Ok(format!("http://127.0.0.1:{}", port.unwrap()))
}

/// Endpoint flags and offline modes belong to the benchmark parser. Only fill
/// its missing single-server URL; do not alter explicit comparison targets.
pub(crate) fn bench_arguments(arguments: &[String]) -> Result<Vec<String>, String> {
    let mut arguments = arguments.to_vec();
    if !arguments.iter().any(|arg| matches!(arg.as_str(),
        "--url" | "--ab-urls" | "--aba-urls" | "--launch-commands"
        | "--help" | "-h" | "--self-test" | "--suite"))
    {
        let (url, _) = resolve(None, false)
            .map_err(|error| if error.contains("--url") { error } else {
                format!("{error} Supply --url URL to select a benchmark server explicitly.")
            })?;
        arguments.extend(["--url".into(), url]);
    }
    Ok(arguments)
}
