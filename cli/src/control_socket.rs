//! OpenSSH control paths are relative to the private state directory so long
//! checkout paths do not exceed sockaddr_un. Never change the parent's cwd.
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Whether no SSH master is listening on the control socket `path`. A path too
/// long to connect to directly is probed from its folder by a child process.
pub fn closed(path: &Path) -> bool {
    match connect(path) {
        Some(closed) => closed,
        None if path.as_os_str().len() >= 104 => probe_child(path),
        None => false,
    }
}

fn connect(path: &Path) -> Option<bool> {
    match std::os::unix::net::UnixStream::connect(path) {
        Ok(_) => Some(false),
        Err(error) if matches!(error.kind(), std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound) => Some(true),
        Err(_) => None,
    }
}

/// Private subprocess entry. A live socket or any uncertainty refuses cleanup.
pub fn probe(name: &str) -> i32 {
    let valid = name.strip_prefix("tunnel-").and_then(|s| s.strip_suffix(".sock"))
        .is_some_and(|token| crate::text::hex(token, 16));
    if valid && connect(Path::new(name)) == Some(true) { 0 } else { 1 }
}

fn probe_child(path: &Path) -> bool {
    let Some(parent) = path.parent() else { return false; };
    let Some(name) = path.file_name() else { return false; };
    let Ok(executable) = std::env::current_exe() else { return false; };
    let Ok(mut child) = Command::new(executable).arg("__socket-probe").arg(name)
        .current_dir(parent).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn()
    else { return false; };
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
            _ => { let _ = child.kill(); let _ = child.wait(); return false; }
        }
    }
}
