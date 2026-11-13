//! Read bounded private engine-log tails and redact sensitive data.
use super::*;
use std::io::{Seek, SeekFrom};

const MAX_LOG: u64 = 64 * 1024;

pub(super) fn read_log(store: &Store) -> Result<String, String> {
    let secrets = read_fields(&store.env())?.unwrap_or_default();
    let mut log = match private_open(&store.log(), false, false) {
        Ok(file) => file,
        Err(_) if !store.log().exists() => return Ok(String::new()),
        Err(error) => return Err(error),
    };
    let length = log
        .metadata()
        .map_err(|_| "Cannot inspect the private engine log.".to_string())?
        .len();
    log.seek(SeekFrom::Start(length.saturating_sub(MAX_LOG)))
        .map_err(|_| "Cannot read the private engine log.".to_string())?;
    let mut bytes = Vec::new();
    log.take(MAX_LOG)
        .read_to_end(&mut bytes)
        .map_err(|_| "Cannot read the private engine log.".to_string())?;
    let text = String::from_utf8_lossy(&bytes);
    // A tail can begin halfway through a secret-bearing line: discard its
    // partial first line before applying complete-line redaction.
    let text = if length > MAX_LOG {
        text.split_once('\n').map(|(_, rest)| rest).unwrap_or("")
    } else {
        &text
    };
    Ok(redact_log(text, &secrets))
}

fn redact_log(text: &str, environment: &Fields) -> String {
    let mut result = String::new();
    for line in text.lines() {
        // These exact server diagnostics contain neither paths nor credentials.
        let auth_diagnostic = matches!(line,
            "FAIL --allow-remote requires --api-key-file PATH"
            | "WARNING: --api-key-file is group- or world-readable");
        if sensitive_name(line) && !auth_diagnostic {
            result.push_str("[private environment line omitted]\n");
            continue;
        }
        let mut line = line.to_owned();
        for (key, value) in environment {
            if sensitive_environment_name(key) && !value.is_empty() {
                line = line.replace(value, "[redacted]");
            }
        }
        for character in line.chars() {
            if !character.is_control() || character == '\t' {
                result.push(character);
            }
        }
        result.push('\n');
    }
    result
}

fn sensitive_name(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    [
        "secret",
        "password",
        "credential",
        "private_key",
        "api_key",
        "api-key",
        "authorization",
        "auth_token",
        "access_token",
        "bearer ",
    ]
    .iter()
    .any(|needle| value.contains(needle))
}

pub(super) fn sensitive_environment_name(value: &str) -> bool {
    let name = value
        .strip_prefix("env.")
        .or_else(|| value.strip_prefix("env_unset."))
        .unwrap_or(value)
        .to_ascii_uppercase();
    // This is a file path, not the credential; serving receives the path in argv.
    if name == "VADE_API_KEY_FILE" { return false; }
    sensitive_name(&name)
        || [
            "_TOKEN",
            "_KEY",
            "_PASSWORD",
            "_SECRET",
            "_CREDENTIAL",
            "_CREDENTIALS",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
        || matches!(name.as_str(), "TOKEN" | "KEY" | "PASSWORD" | "SECRET")
}

#[cfg(test)]
mod tests;
