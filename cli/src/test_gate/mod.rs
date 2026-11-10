//! Build/export and legacy flash-bench checks, driven without shell dispatch.
//! `build` runs host build/export checks; `determinism` retains flash-bench's
//! logit and HTTP comparison receipts. Its fixtures and backend determine
//! coverage; accepting world one does not establish FP4-Local qualification.
mod build;
mod determinism;
use crate::sha256;

use std::ffi::CString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::{
    fs::DirBuilderExt,
    process::{CommandExt, ExitStatusExt},
};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

static INTERRUPTED: AtomicI32 = AtomicI32::new(0);
unsafe extern "C" {
    fn signal(number: i32, handler: usize) -> usize;
    fn kill(pid: i32, number: i32) -> i32;
    fn access(path: *const std::ffi::c_char, mode: i32) -> i32;
}
extern "C" fn interrupt(number: i32) {
    INTERRUPTED.store(number, Ordering::Relaxed);
}
struct Signals(Vec<(i32, usize)>);
impl Signals {
    fn install() -> Self {
        INTERRUPTED.store(0, Ordering::Relaxed);
        Self(
            [2, 15]
                .into_iter()
                .map(|n| (n, unsafe { signal(n, interrupt as *const () as usize) }))
                .collect(),
        )
    }
}
impl Drop for Signals {
    fn drop(&mut self) {
        for &(n, previous) in &self.0 {
            unsafe {
                signal(n, previous);
            }
        }
    }
}
/// The signal number that interrupted the gate, or 0.
pub(super) fn interrupted() -> i32 {
    INTERRUPTED.load(Ordering::Relaxed)
}

/// `vade test build` or `vade test determinism`; returns the exit code.
pub fn run(args: &[String]) -> i32 {
    let _signals = Signals::install();
    match args.first().map(String::as_str) {
        None | Some("--help" | "-h") => {
            crate::public_output(HELP);
            0
        }
        Some("build") => build::run(&args[1..]),
        Some("determinism") => determinism::run(&args[1..]),
        _ => {
            crate::output::stderr_line(format_args!("FAIL inputs: use vade test build or vade test determinism (see --help)"));
            2
        }
    }
}
/// Usage for the gate commands, printed after the evaluation commands' usage
/// (hence the leading blank line).
pub const HELP: &str = "\nUsage: vade test build [OPTIONS]\n       vade test determinism [OPTIONS] -- MODEL FIXTURES ADDR0 ADDR1 RANK [--world 1|2|4]\n\nBuild checks the current engine checkout and a private release snapshot.\nDeterminism drives the existing flash-bench byte comparisons and receipts.\nUse vade test build --help or vade test determinism --help for exact inputs.\nNative Spark qualification remains separate from host checks.\n";

fn engine_root() -> Result<Option<PathBuf>, &'static str> {
    if let Some(selected) = std::env::var_os("VADE_ROOT") {
        if selected.is_empty() {
            return Err("FAIL inputs: VADE_ROOT must select an engine checkout or its repository");
        }
        let root = PathBuf::from(selected);
        return [root.join("engine"), root]
            .into_iter()
            .find_map(|p| {
                // Cargo must not discover an ancestor manifest for a selected
                // marker-only directory nested inside another checkout.
                (p.join("Cargo.toml").is_file() && p.join("tools/export/Cargo.toml").is_file())
                    .then(|| fs::canonicalize(p).ok())
                    .flatten()
            })
            .map(Some)
            .ok_or("FAIL inputs: VADE_ROOT must select an engine checkout or its repository");
    }
    let Ok(start) = std::env::current_dir() else {
        return Ok(None);
    };
    for parent in start.ancestors() {
        let component = parent.join("engine");
        if component.join("tools/export/Cargo.toml").is_file() {
            return Ok(fs::canonicalize(component).ok());
        }
        if parent.join("tools/export/Cargo.toml").is_file() {
            return Ok(fs::canonicalize(parent).ok());
        }
    }
    Ok(None)
}

/// A private temporary directory, removed when dropped.
pub(super) struct TempDir {
    pub path: PathBuf,
}
impl TempDir {
    /// Create `<prefix>.<random>` (mode 0700) in the system temporary directory.
    pub fn new(prefix: &str) -> Result<Self, String> {
        let root = fs::canonicalize(std::env::temp_dir())
            .map_err(|_| "Cannot open temporary directory. Check TMPDIR.".to_string())?;
        let path = root.join(format!("{prefix}.{}", crate::text::random_hex(8)?));
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .map_err(|_| "Cannot create gate temporary directory.".to_string())?;
        Ok(Self { path })
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn status_code(status: std::process::ExitStatus) -> i32 {
    status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(1))
}
fn wait_child(child: &mut Child, deadline: Option<Instant>) -> Result<i32, String> {
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            return Ok(if interrupted() != 0 {
                128 + interrupted()
            } else {
                status_code(status)
            });
        }
        let caught = interrupted();
        let timed_out = deadline.is_some_and(|d| Instant::now() >= d);
        if caught != 0 || timed_out {
            // Match timeout --foreground: a deadline signals only this child.
            // An operator interrupt reaches the owned command's whole group.
            // Neither path signals an external server or another rank.
            let target = if caught != 0 {
                -(child.id() as i32)
            } else {
                child.id() as i32
            };
            unsafe {
                kill(target, if caught != 0 { caught } else { 15 });
            }
            let grace = Instant::now() + Duration::from_secs(5);
            loop {
                if child.try_wait().map_err(|e| e.to_string())?.is_some() {
                    return Ok(if caught != 0 { 128 + caught } else { 124 });
                }
                if Instant::now() >= grace {
                    unsafe {
                        kill(target, 9);
                    }
                    let _ = child.wait();
                    return Ok(if caught != 0 { 128 + caught } else { 137 });
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn spawn(command: &mut Command) -> std::io::Result<Child> {
    // Gate receipts use C/POSIX diagnostics, including forwarded child output.
    // Keep this local to gate children without changing the caller's locale.
    command.env("LC_ALL", "C");
    command.process_group(0);
    unsafe {
        command.pre_exec(|| {
            for n in [2, 15] {
                signal(n, 0);
            }
            Ok(())
        });
    }
    command.spawn()
}
fn spawn_code(error: &std::io::Error) -> i32 {
    if error.kind() == std::io::ErrorKind::NotFound {
        127
    } else {
        126
    }
}
/// Run `command` with stdout and stderr appended to `log`; returns its exit
/// code (128 + signal when interrupted).
pub(super) fn exec_logged(
    command: &mut Command,
    log: &Path,
    deadline: Option<Instant>,
) -> Result<i32, String> {
    if interrupted() != 0 {
        return Ok(128 + interrupted());
    }
    let out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .map_err(|e| e.to_string())?;
    command
        .stdout(out.try_clone().map_err(|e| e.to_string())?)
        .stderr(out);
    match spawn(command) {
        Ok(mut child) => wait_child(&mut child, deadline),
        Err(error) => {
            let mut file = OpenOptions::new()
                .append(true)
                .open(log)
                .map_err(|e| e.to_string())?;
            writeln!(file, "Cannot execute gate command: {error}").map_err(|e| e.to_string())?;
            Ok(spawn_code(&error))
        }
    }
}
/// Run `command` on this terminal's stdout and stderr; returns its exit code.
pub(super) fn exec_inherited(
    command: &mut Command,
    deadline: Option<Instant>,
) -> Result<i32, String> {
    if interrupted() != 0 {
        return Ok(128 + interrupted());
    }
    command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    match spawn(command) {
        Ok(mut child) => wait_child(&mut child, deadline),
        Err(error) => {
            crate::output::stderr_line(format_args!("Cannot execute gate command: {error}"));
            Ok(spawn_code(&error))
        }
    }
}
/// Run `command` with stdout written to `path`; returns its exit code.
pub(super) fn exec_stdout(
    command: &mut Command,
    path: &Path,
    deadline: Option<Instant>,
) -> Result<i32, String> {
    if interrupted() != 0 {
        return Ok(128 + interrupted());
    }
    command
        .stdout(fs::File::create(path).map_err(|e| e.to_string())?)
        .stderr(Stdio::inherit());
    match spawn(command) {
        Ok(mut child) => wait_child(&mut child, deadline),
        Err(error) => {
            crate::output::stderr_line(format_args!("Cannot execute gate command: {error}"));
            Ok(spawn_code(&error))
        }
    }
}
/// Whether this user may execute `path`.
pub(super) fn executable(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes()).is_ok_and(|p| unsafe { access(p.as_ptr(), 1) } == 0)
}
