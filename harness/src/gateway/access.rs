//! Browser guards shared by every client-facing gateway listener. A loopback bind
//! alone does not stop a web page: browsers send cross-site requests to 127.0.0.1,
//! and DNS rebinding gives a hostile name a loopback address. Host must name a
//! loopback authority of the listener (421), an Origin must be allow-listed (403),
//! and an optional token must arrive as `Authorization: Bearer` (401).
//! Command-line clients send a loopback Host and no Origin, so they pass unchanged.
use crate::transport::Head;
use std::io::{self, Write};
use std::net::{IpAddr, SocketAddr};

/// Environment fallback for `--token`; the flag overrides it.
pub const TOKEN_ENV: &str = "VADE_GATEWAY_TOKEN";
const MAX_VALUE: usize = 256;
/// A shorter token is too easy to guess, like the engine key.
const MIN_TOKEN: usize = 16;
const MAX_TOKEN: usize = 1024;

/// No Debug: the token must never reach logs or diagnostics.
#[derive(Clone, Default)]
pub struct Access {
    origins: Vec<String>,
    hosts: Vec<String>,
    token: Option<String>,
}

/// A request the listener answers itself, before reading a body or contacting upstream.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    Status(u16, &'static str),
    /// CORS preflight from an allow-listed Origin; carries the CORS header lines.
    Preflight(String),
}

impl Access {
    /// `--allow-origin`: an exact browser Origin (`scheme://host[:port]`, or `null`).
    pub fn allow_origin(&mut self, value: &str) -> Result<(), &'static str> {
        let value = value.to_ascii_lowercase();
        let valid = value == "null"
            || value.split_once("://").is_some_and(|(scheme, authority)| {
                scheme.bytes().next().is_some_and(|b| b.is_ascii_alphabetic())
                    && scheme.bytes().all(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b))
                    && authority_chars(authority)
            });
        if !valid || value.len() > MAX_VALUE {
            return Err("--allow-origin needs scheme://host[:port] without a path, or null");
        }
        self.origins.push(value);
        Ok(())
    }

    /// `--allow-host`: an extra Host value, with or without the listener port.
    pub fn allow_host(&mut self, value: &str) -> Result<(), &'static str> {
        let value = value.to_ascii_lowercase();
        if !authority_chars(&value) || value.len() > MAX_VALUE {
            return Err("--allow-host needs a host name or address, optionally with :port");
        }
        self.hosts.push(value);
        Ok(())
    }

    /// `--token`: require `Authorization: Bearer TOKEN` on every request.
    pub fn require_token(&mut self, value: &str) -> Result<(), &'static str> {
        if value.len() < MIN_TOKEN || value.len() > MAX_TOKEN || !value.bytes().all(|b| b.is_ascii_graphic()) {
            return Err("--token needs 16 to 1024 visible ASCII characters");
        }
        self.token = Some(value.to_owned());
        Ok(())
    }

    /// Applies VADE_GATEWAY_TOKEN unless `--token` already set a token.
    pub fn token_from_env(&mut self) -> Result<(), &'static str> {
        if self.token.is_some() {
            return Ok(());
        }
        match std::env::var(TOKEN_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(()),
            Ok(value) if self.require_token(&value).is_ok() => Ok(()),
            _ => Err("VADE_GATEWAY_TOKEN needs 16 to 1024 visible ASCII characters"),
        }
    }

    /// Host, Origin, preflight, then token. `Ok` carries the allow-listed Origin
    /// that the relayed response must name (see [`cors`]).
    pub(crate) fn check(&self, head: &Head, local: SocketAddr) -> Result<Option<String>, Refusal> {
        match one(head, "host") {
            Err(()) => return Err(Refusal::Status(400, "duplicate Host is unsupported")),
            Ok(None) => return Err(Refusal::Status(400, "the gateway requires a Host header")),
            Ok(Some(host)) if !self.host_allowed(host, local) => {
                return Err(Refusal::Status(421, "Host must name the gateway loopback address; see --allow-host"));
            }
            Ok(Some(_)) => {}
        }
        let origin = match one(head, "origin") {
            Err(()) => return Err(Refusal::Status(400, "duplicate Origin is unsupported")),
            Ok(Some(origin)) if !self.origins.iter().any(|o| o.eq_ignore_ascii_case(origin)) => {
                return Err(Refusal::Status(403, "cross-origin requests are refused; see --allow-origin"));
            }
            Ok(origin) => origin,
        };
        // Browsers send preflights without credentials, so answer before the token.
        if let Some(origin) = origin
            && head.first.starts_with("OPTIONS ")
            && let Some(method) = head.get("access-control-request-method")
        {
            let method = if token_list(method) { method } else { "GET, POST" };
            let headers = head
                .get("access-control-request-headers")
                .filter(|v| token_list(v))
                .unwrap_or("authorization, content-type");
            return Err(Refusal::Preflight(format!(
                "{}Access-Control-Allow-Methods: {method}\r\nAccess-Control-Allow-Headers: {headers}\r\nAccess-Control-Max-Age: 600\r\n",
                cors(Some(origin))
            )));
        }
        if let Some(token) = &self.token {
            let presented = match one(head, "authorization") {
                Ok(Some(value)) => value
                    .split_once(' ')
                    .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                    .map(|(_, credential)| credential.trim()),
                _ => None,
            };
            if !same(token.as_bytes(), presented.unwrap_or("").as_bytes()) {
                return Err(Refusal::Status(401, "gateway token required: Authorization: Bearer TOKEN"));
            }
        }
        Ok(origin.map(str::to_owned))
    }

    /// The gateway token is consumed here; it is never forwarded upstream.
    pub(crate) fn consumes_authorization(&self) -> bool {
        self.token.is_some()
    }

    fn host_allowed(&self, value: &str, local: SocketAddr) -> bool {
        let value = value.to_ascii_lowercase();
        let port = local.port().to_string();
        let (name, explicit) = match value.rsplit_once(':') {
            Some((name, tail)) if !tail.contains(']') => (name, Some(tail)),
            _ => (value.as_str(), None),
        };
        if self.hosts.iter().any(|host| *host == value || (*host == name && explicit == Some(port.as_str()))) {
            return true;
        }
        // A browser always names a non-default port, so a bare loopback name cannot
        // come from a page addressing this listener. No DNS lookup happens here.
        if explicit.is_some_and(|tail| tail != port) {
            return false;
        }
        let bound = match local.ip() {
            IpAddr::V4(ip) => ip.to_string(),
            IpAddr::V6(ip) => format!("[{ip}]"),
        };
        matches!(name, "127.0.0.1" | "localhost" | "[::1]") || name == bound
    }
}

/// Response header lines for an allow-listed Origin; empty otherwise.
pub(crate) fn cors(origin: Option<&str>) -> String {
    origin.map_or_else(String::new, |origin| format!("Access-Control-Allow-Origin: {origin}\r\nVary: Origin\r\n"))
}

pub(crate) fn write_preflight(writer: &mut dyn Write, marker: &str, headers: &str) -> io::Result<()> {
    write!(writer, "HTTP/1.1 204 No Content\r\n{marker}{headers}Content-Length: 0\r\nConnection: close\r\n\r\n")?;
    writer.flush()
}

fn one<'a>(head: &'a Head, name: &str) -> Result<Option<&'a str>, ()> {
    let mut values = head.fields.iter().filter(|(key, _)| key == name).map(|(_, v)| v.as_str());
    match (values.next(), values.next()) {
        (_, Some(_)) => Err(()),
        (value, None) => Ok(value),
    }
}

fn authority_chars(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || b"-.:[]_".contains(&b))
}

fn token_list(value: &str) -> bool {
    value.len() <= MAX_VALUE
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~, ".contains(&b))
}

/// Work is fixed by the configured token length, also on a length mismatch,
/// as in the engine key check. black_box keeps the fold from becoming an early exit.
fn same(expected: &[u8], supplied: &[u8]) -> bool {
    let mut difference = expected.len() ^ supplied.len();
    for (index, byte) in expected.iter().enumerate() {
        difference |= std::hint::black_box(usize::from(byte ^ supplied.get(index).copied().unwrap_or(0)));
    }
    !expected.is_empty() && std::hint::black_box(difference) == 0
}

#[cfg(test)]
#[path = "access_tests.rs"]
mod tests;
