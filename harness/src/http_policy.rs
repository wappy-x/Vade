//! The process's explicit choice for unencrypted upstream HTTP.
use std::{collections::BTreeMap, io, net::{SocketAddr, TcpStream}, sync::{Mutex, OnceLock}};
use crate::transport::Endpoint;

pub const WARNING: &str = "Warning: --allow-plain-http is on. Links to other machines are not encrypted. Anyone on the network can read and change your chat, your tool results and your API key.";
static ALLOW: OnceLock<()> = OnceLock::new();
static WARNED: OnceLock<()> = OnceLock::new();

static REPORTS: Mutex<Reports> = Mutex::new(Reports::new());
struct Reports { latest: BTreeMap<String, bool>, printed: std::collections::BTreeSet<String>, lines: Vec<String>, interactive: bool, first: Option<String> }
impl Reports {
    const fn new() -> Self { Self { latest: BTreeMap::new(), printed: std::collections::BTreeSet::new(), lines: Vec::new(), interactive: false, first: None } }
    fn note(&mut self, endpoint: &Endpoint, refused: bool) {
        self.latest.insert(endpoint.authority.clone(), refused);
        if refused && self.interactive {
            self.first.get_or_insert_with(|| full_reason(endpoint));
        } else if refused && self.printed.insert(endpoint.authority.clone()) {
            let message = full_reason(endpoint);
            crate::diagnostics::stderr_line(format_args!("{message}"));
            self.lines.push(message);
        }
    }
}

/// Called by command parsers, only for a flag, never for an option's value.
pub fn allow_plain_http() {
    warn_plain_http();
    let _ = ALLOW.set(());
}

/// Announce a parsed flag even if another option later fails validation.
pub fn warn_plain_http() {
    if WARNED.set(()).is_ok() {
        crate::diagnostics::stderr_line(format_args!("{WARNING}"));
    }
}

/// The binary selects this before starting interactive workers. Library and
/// gateway callers retain per-upstream reporting and receive ordinary errors.
pub fn begin_gateway() { with_reports(|reports| reports.interactive = false); }
pub fn begin_interactive() { with_reports(|reports| reports.interactive = true); }

/// A refusal is terminal even for workers whose result type keeps only an error
/// kind. Keep the first refusal until shutdown; do not retry or start more work.
pub fn interactive_result() -> io::Result<()> {
    with_reports(|reports| match &reports.first {
        Some(_) => Err(io::Error::new(io::ErrorKind::PermissionDenied, Refused("plain HTTP refused".into()))),
        None => Ok(()),
    })
}

/// Called after the command's workers, child processes and terminal are closed.
pub fn finish_interactive(code: std::process::ExitCode) -> std::process::ExitCode {
    with_reports(|reports| match reports.first.take() {
        Some(message) => {
            crate::diagnostics::stderr_line(format_args!("{message}"));
            std::process::ExitCode::FAILURE
        }
        None => code,
    })
}

/// Only register outputs newly created by this run, never pre-existing paths.
/// Retain other partial results, but a refused run must leave no output behind.
pub(crate) struct RefusalOutput(pub std::path::PathBuf);
impl Drop for RefusalOutput {
    fn drop(&mut self) {
        if interactive_result().is_err() {
            let result = if self.0.is_dir() { std::fs::remove_dir_all(&self.0) } else { std::fs::remove_file(&self.0) };
            if let Err(error) = result {
                crate::diagnostics::stderr_line(format_args!("vade: cannot remove refused run output ({})", error.kind()));
            }
        }
    }
}

/// Help does not parse values or connect, but an explicit switch still warns.
pub fn help_requested(args: &[String]) -> bool {
    let help = args.iter().any(|a| a == "--help" || a == "-h");
    if help && args.iter().any(|a| a == "--allow-plain-http") { warn_plain_http(); }
    help
}

/// Decide from the connected addresses, not the configured name or request data.
fn allowed(peer: SocketAddr, local: SocketAddr, allow: bool) -> bool {
    let peer = peer.ip().to_canonical();
    // Test aid: disabling the two local exceptions can only close links that
    // the normal rule allows. It cannot open a link that the rule would close.
    let remote = std::env::var_os("VADE_HTTP_TREAT_PEERS_AS_REMOTE").is_some_and(|v| v == "1");
    allow || (!remote && (peer.is_loopback() || peer == local.ip().to_canonical()))
}

fn full_reason(endpoint: &Endpoint) -> String {
    let authority = target(endpoint);
    let host = &endpoint.host;
    let port = typed_port(endpoint);
    // The local end of the tunnel: a port under 1024 needs administrator rights, so the hint uses 8080 then.
    // The last sentence names no option: the doctor tool takes the address as its first word.
    let local = if port.parse::<u16>().is_ok_and(|number| number < 1024) { "8080" } else { port };
    format!("Vade sent nothing to {authority}.\n{host} is another machine, and this link is plain HTTP. Plain HTTP is not encrypted. Anyone on the network between the two machines can read and change your chat, your tool results and your API key.\nDo one of these:\n  1. Use an SSH tunnel. It encrypts the link. `vade up` sets one up for `vade chat`. By hand: run  ssh -N -L {local}:127.0.0.1:{port} USER@{host}  in another terminal. Replace USER with your login name on that machine. Then start Vade again and use http://127.0.0.1:{local} as the address.\n  2. Accept the risk for this run: add --allow-plain-http")
}
fn typed_port(endpoint: &Endpoint) -> &str {
    let authority = &endpoint.authority;
    if authority.starts_with('[') { authority.split_once("]:").map(|(_, port)| port) }
    else { authority.split_once(':').map(|(_, port)| port) }.unwrap_or("80")
}
fn target(endpoint: &Endpoint) -> String {
    if endpoint.authority.ends_with(']') || !endpoint.authority.contains(':') {
        format!("{}:80", endpoint.authority)
    } else { endpoint.authority.clone() }
}
fn short_reason(authority: &str) -> String {
    format!("The gateway sent nothing to {authority}. It is another machine and the link is not encrypted. Use an SSH tunnel or start the gateway with --allow-plain-http.")
}
#[derive(Debug)]
struct Refused(String);
impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for Refused {}
pub fn refusal(error: &io::Error) -> Option<&str> {
    error.get_ref()?.downcast_ref::<Refused>().map(|e| e.0.as_str())
}

// Each test gets its own report state. Production shares it across workers.
fn with_reports<T>(f: impl FnOnce(&mut Reports) -> T) -> T {
    #[cfg(test)]
    if let Some(reports) = testing::reports() { return f(&mut reports.lock().unwrap()); }
    f(&mut REPORTS.lock().unwrap_or_else(|e| e.into_inner()))
}
pub(crate) fn last_refusal(endpoint: &Endpoint) -> Option<String> {
    with_reports(|reports| (reports.latest.get(&endpoint.authority) == Some(&true)).then(|| short_reason(&target(endpoint))))
}

/// Own the new socket so refusal closes it before returning an error.
/// Only the returned socket may enter an HTTP writer or the connection pool.
pub(crate) fn check(stream: TcpStream, endpoint: &Endpoint) -> io::Result<TcpStream> {
    let (peer, local, allow) = (stream.peer_addr()?, stream.local_addr()?, ALLOW.get().is_some());
    #[cfg(test)]
    let (peer, local, allow) = testing::addresses().unwrap_or((peer, local, allow));
    if allowed(peer, local, allow) {
        // Do not allocate entries for ordinary same-machine connections.
        with_reports(|reports| { if let Some(refused) = reports.latest.get_mut(&endpoint.authority) { *refused = false; } });
        return Ok(stream);
    }
    drop(stream);
    with_reports(|reports| reports.note(endpoint, true));
    Err(io::Error::new(io::ErrorKind::PermissionDenied, Refused(short_reason(&target(endpoint)))))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use std::{cell::RefCell, sync::Arc};
    #[derive(Clone)]
    struct Override { peer: SocketAddr, local: SocketAddr, allow: bool, log: Arc<Mutex<Reports>> }
    thread_local! { static OVERRIDE: RefCell<Option<Override>> = const { RefCell::new(None) }; }
    struct Guard(Option<Override>);
    impl Drop for Guard { fn drop(&mut self) { OVERRIDE.with(|s| *s.borrow_mut() = self.0.take()); } }
    pub(super) fn addresses() -> Option<(SocketAddr, SocketAddr, bool)> {
        OVERRIDE.with(|s| s.borrow().as_ref().map(|s| (s.peer, s.local, s.allow)))
    }
    pub(super) fn reports() -> Option<Arc<Mutex<Reports>>> {
        OVERRIDE.with(|s| s.borrow().as_ref().map(|s| s.log.clone()))
    }
    pub(crate) fn remote<T>(allow: bool, f: impl FnOnce() -> T) -> (T, Vec<String>) {
        let log = Arc::new(Mutex::new(Reports::new()));
        let state = Override { peer: "192.0.2.10:80".parse().unwrap(), local: "192.0.2.20:50000".parse().unwrap(), allow, log: log.clone() };
        let guard = Guard(OVERRIDE.with(|s| s.replace(Some(state))));
        let result = f();
        drop(guard);
        let lines = log.lock().unwrap().lines.clone();
        (result, lines)
    }
}

#[cfg(test)]
#[path = "http_policy_tests.rs"]
pub(crate) mod tests;
