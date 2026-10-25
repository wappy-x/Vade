//! Actionable terminal diagnostics built only from trusted codes and categories.
use std::{io::{self, Write}, process::ExitCode};

/// A bounded option name for diagnostics, excluding any attached value.
pub fn option_name(argument: &str) -> Option<&str> {
    let name = argument.split('=').next()?;
    (name.starts_with('-') && name.len() <= 64
        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-'))
        .then_some(name)
}

/// Attach only the option label being parsed, never its value. Parsers clear
/// the label before cross-option validation so the last flag is not blamed.
pub(crate) fn parse_command<T>(
    command: &'static str,
    parse: impl FnOnce(&mut Option<String>) -> Result<T, &'static str>,
) -> Result<T, String> {
    let mut option = None;
    parse(&mut option).map_err(|message| {
        let message = message.split("; see ").next().unwrap_or(message).trim_end_matches('.');
        format!("{}{message}. See vade-harness {command} --help.",
            option.map_or_else(String::new, |option| format!("{option}: ")))
    })
}

pub(crate) fn output_file_hint(error: &io::Error) -> &'static str {
    match error.kind() {
        io::ErrorKind::AlreadyExists => "The destination already exists. Keep it and choose a new output path.",
        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory => "Choose an output path under an existing directory.",
        io::ErrorKind::IsADirectory => "The destination is a directory. Choose a new file path.",
        io::ErrorKind::PermissionDenied => "Choose an output directory you can write to, or correct its permissions.",
        io::ErrorKind::StorageFull => "Free space on the output volume, then retry with a new file path.",
        io::ErrorKind::ReadOnlyFilesystem => "Choose an output path on a writable filesystem.",
        _ => "Check the output directory permissions and free space; preserve any partial file before retrying.",
    }
}

/// Utility output treats a consumer closing its pipe as normal completion.
/// Other write failures still fail; generation and gateway streams do not use
/// this helper, so their cancellation and transport errors remain unchanged.
pub fn write_stdout(text: &str) -> io::Result<()> {
    let mut output = io::stdout().lock();
    match output.write_all(text.as_bytes()).and_then(|()| output.flush()) {
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        result => result,
    }
}

/// [`write_stdout`], turned into an exit code; a failed write prints one
/// actionable line on standard error.
pub fn print_stdout(text: &str) -> ExitCode {
    match write_stdout(text) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => {
            crate::diagnostics::stderr_line(format_args!("vade: cannot write standard output. Check the destination's permissions and available space, then retry."));
            ExitCode::FAILURE
        }
    }
}

/// A next step for the common local-engine failures; never the raw error.
pub fn error_hint(error: &io::Error) -> &'static str {
    if error.to_string().starts_with("cannot write replay statistics") {
        return " Replay stopped. Check the output destination. Inspect any --stats file before explicitly replaying remaining requests.";
    }
    if error.to_string() == "direct text chat does not execute upstream tool calls" {
        return " The model asked to run a tool, which plain chat cannot do; nothing ran. Use agent mode for tools.";
    }
    if error.to_string() == "no output room in context" {
        return " The chat is too long for another reply; history is kept. Use /new.";
    }
    if error.to_string() == "requested output exceeds context budget" {
        return " The reply length you asked for does not fit; lower --max-output-tokens or use /new.";
    }
    if error.kind() == io::ErrorKind::PermissionDenied
        && error.to_string() == "HTTP 401 Unauthorized: authentication failed"
    {
        return " Check --api-key-file or VADE_API_KEY_FILE and the engine's authentication settings, then retry.";
    }
    if let Some(status) = crate::transport::http_status(error) {
        return match status {
            401 | 403 => " Check the engine's authentication settings and your API key, then retry.",
            404 | 405 => " Check --endpoint and the engine's supported API routes; the usual base is http://127.0.0.1:18080/v1.",
            408 | 504 => " The upstream timed out. Check vade status and retry when it is ready.",
            413 => " The request is too large. Shorten the prompt or use /new, then retry.",
            429 | 503 => " The engine is busy or unavailable. Check vade status, wait for capacity, then retry.",
            400 | 409 | 422 => " The engine rejected the request. Check the model, context and output limits with vade doctor before retrying.",
            _ => " Check vade status and the engine logs before retrying.",
        };
    }
    match error.kind() {
        io::ErrorKind::ConnectionRefused => {
            " Is the engine running? Check with vade status."
        }
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => {
            " The engine did not answer in time. Check vade status, then retry."
        }
        io::ErrorKind::AddrInUse => " Another process is using the gateway address. Choose a free loopback port with --bind 127.0.0.1:PORT, or stop the gateway you started.",
        io::ErrorKind::AddrNotAvailable => " Use a local loopback address with --bind 127.0.0.1:PORT.",
        io::ErrorKind::NotFound => " Check the executable or resource path in your configuration, then retry.",
        io::ErrorKind::Interrupted => " The request was interrupted; review any partial output before sending it again.",
        io::ErrorKind::InvalidInput => " Check the command options with --help, then retry.",
        io::ErrorKind::InvalidData => " The upstream reply was invalid. Check --endpoint and the engine logs before retrying.",
        io::ErrorKind::Unsupported => " Use an HTTP loopback endpoint; connect remote engines through an SSH tunnel.",
        io::ErrorKind::PermissionDenied => " Check the configured file permissions or credentials, then retry.",
        io::ErrorKind::UnexpectedEof | io::ErrorKind::ConnectionReset | io::ErrorKind::BrokenPipe => " The reply ended early; review any partial output and check vade status before retrying.",
        _ => " Check vade status and the engine logs before retrying.",
    }
}
/// A short, trusted name for an error (an HTTP status or an I/O category),
/// safe to show without the untrusted upstream text.
pub fn error_category(error: &io::Error) -> &'static str {
    if let Some(status) = crate::transport::http_status(error) {
        return match status {
            400 => "HTTP 400 Bad Request",
            401 => "HTTP 401 Unauthorized",
            403 => "HTTP 403 Forbidden",
            404 => "HTTP 404 Not Found",
            405 => "HTTP 405 Method Not Allowed",
            408 => "HTTP 408 Request Timeout",
            409 => "HTTP 409 Conflict",
            413 => "HTTP 413 Content Too Large",
            422 => "HTTP 422 Unprocessable Content",
            429 => "HTTP 429 Too Many Requests",
            500 => "HTTP 500 Internal Server Error",
            502 => "HTTP 502 Bad Gateway",
            503 => "HTTP 503 Service Unavailable",
            504 => "HTTP 504 Gateway Timeout",
            _ => "upstream HTTP error",
        };
    }
    if error.kind() == io::ErrorKind::PermissionDenied
        && error.to_string() == "HTTP 401 Unauthorized: authentication failed"
    {
        return "HTTP 401 Unauthorized: authentication failed";
    }
    match error.kind() {
        io::ErrorKind::ConnectionRefused => "connection refused",
        io::ErrorKind::AddrInUse => "address already in use",
        io::ErrorKind::AddrNotAvailable => "address not available",
        io::ErrorKind::NotFound => "executable or resource not found",
        io::ErrorKind::Interrupted => "interrupted",
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => "timeout",
        io::ErrorKind::InvalidInput => "invalid configuration",
        io::ErrorKind::InvalidData => "invalid upstream response",
        io::ErrorKind::Unsupported => "unsupported transport",
        io::ErrorKind::PermissionDenied => "access denied",
        io::ErrorKind::UnexpectedEof => "incomplete upstream response",
        io::ErrorKind::ConnectionReset | io::ErrorKind::BrokenPipe => "connection closed",
        _ => "transport error",
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_common_io_category_has_a_safe_next_step() {
        for kind in [
            io::ErrorKind::ConnectionRefused, io::ErrorKind::AddrInUse,
            io::ErrorKind::AddrNotAvailable, io::ErrorKind::NotFound,
            io::ErrorKind::Interrupted, io::ErrorKind::TimedOut,
            io::ErrorKind::WouldBlock, io::ErrorKind::InvalidInput,
            io::ErrorKind::InvalidData, io::ErrorKind::Unsupported,
            io::ErrorKind::PermissionDenied, io::ErrorKind::UnexpectedEof,
            io::ErrorKind::ConnectionReset, io::ErrorKind::BrokenPipe,
            io::ErrorKind::Other,
        ] {
            let error = io::Error::new(kind, "PRIVATE_URL?key=secret\x1b[2J");
            assert!(!error_hint(&error).trim().is_empty(), "{kind:?}");
            for text in [error_hint(&error), error_category(&error)] {
                assert!(!text.contains("PRIVATE"));
                assert!(!text.contains("secret"));
                assert!(!text.contains('\x1b'));
            }
        }
    }
}

/// Diagnostics must not panic when a consumer also closes standard error.
pub fn stderr(args: std::fmt::Arguments<'_>) { let _ = io::stderr().lock().write_fmt(args); }
/// [`stderr`] followed by a newline.
pub fn stderr_line(args: std::fmt::Arguments<'_>) { let _ = writeln!(io::stderr().lock(), "{args}"); }

#[cfg(test)]
mod file_error_tests {
    use super::*;
    #[test]
    fn output_file_failures_have_specific_actions_and_never_echo_raw_errors() {
        for (kind, action) in [
            (io::ErrorKind::PermissionDenied, "correct its permissions"),
            (io::ErrorKind::AlreadyExists, "Keep it and choose a new output path"),
            (io::ErrorKind::StorageFull, "Free space on the output volume"),
            (io::ErrorKind::ReadOnlyFilesystem, "writable filesystem"),
        ] {
            let hint = output_file_hint(&io::Error::new(kind, "PRIVATE error payload\x1b[2J"));
            assert!(hint.contains(action), "{kind:?}: {hint}");
            assert!(!hint.contains("PRIVATE") && !hint.contains('\x1b'));
        }
    }
}
