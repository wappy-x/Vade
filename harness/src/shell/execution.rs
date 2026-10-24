//! OS confinement, bounded process execution and workspace verification.
use super::*;

#[cfg(target_os = "macos")]
pub(super) fn profile(
    root: &std::path::Path,
    exclusions: &Exclusions,
    build: &std::path::Path,
    toolchain: Option<&std::path::Path>,
    read_only: bool,
    git: bool,
) -> io::Result<String> {
    let root = root
        .to_str()
        .ok_or_else(|| io::Error::other("workspace path must be UTF-8"))?;
    if root.chars().any(char::is_control) {
        return Err(io::Error::other("invalid workspace path"));
    }
    let root = crate::json::quote(root);
    let quote_path = |path: &std::path::Path| -> io::Result<String> {
        let path = path
            .to_str()
            .ok_or_else(|| io::Error::other("sandbox paths must be UTF-8"))?;
        if path.chars().any(char::is_control) {
            return Err(io::Error::other("invalid sandbox path"));
        }
        Ok(crate::json::quote(path))
    };
    let build = quote_path(build)?;
    let mut private_reads = format!("(subpath {build})");
    if git {
        private_reads.push_str(&format!(" (subpath (string-append {root} \"/.git\"))"));
    }
    let read_only_policy = if read_only {
        format!("(deny file-write* (subpath {root}))")
    } else {
        String::new()
    };
    // Attributes in .git/info can select a clean/process filter even for
    // status and diff. Read-only git must never execute those workspace- or
    // config-selected helpers. /usr/bin/git is Apple's launcher; the Command
    // Line Tools binary is its fixed system implementation.
    let process_policy = if read_only && git {
        "(allow process-fork)\n(allow process-exec (literal \"/usr/bin/git\") (literal \"/Library/Developer/CommandLineTools/usr/bin/git\"))"
    } else {
        "(allow process-exec process-fork)"
    };
    let mut toolchain_policy = String::new();
    if let Some(toolchain) = toolchain {
        let bin = quote_path(&toolchain.join("bin"))?;
        let lib = quote_path(&toolchain.join("lib"))?;
        private_reads.push_str(&format!(" (subpath {bin}) (subpath {lib})"));
        toolchain_policy = format!(
            "(allow file-read* (subpath {bin}) (subpath {lib}))\n(deny file-write* (subpath {bin}) (subpath {lib}))"
        );
    }
    let excluded_access = exclusions.access_denial();
    let credential_paths = ascii_case_pattern(&format!(
        "/({})(/|$)",
        crate::tools::CREDENTIAL_NAMES
            .iter()
            .map(|n| n.replace('.', "\\."))
            .collect::<Vec<_>>()
            .join("|")
    ));
    let credential_extensions = ascii_case_pattern(&format!(
        "\\.({})(/|$)",
        crate::tools::CREDENTIAL_SUFFIXES
            .iter()
            .map(|s| s[1..].replace('.', "\\."))
            .collect::<Vec<_>>()
            .join("|")
    ));
    // First-party minimal policy, using the OS sandbox language (not copied
    // profiles). No default-allow fallback. .vade journals cannot be accessed.
    // Explicit foreign process-info/procargs denies are necessary: on this OS
    // deny-default plus named sysctl grants alone still exposes KERN_PROCARGS2.
    // Self queries are needed by executable startup; real Rust fixtures check
    // parent-environment denial without printing the queried buffer.
    Ok(format!(
        r#"(version 1)
(deny default)
{process_policy}
(deny process-info* (require-not (target self)))
(deny sysctl-read (sysctl-name-prefix "kern.procargs"))
(allow sysctl-read (sysctl-name "hw.pagesize") (sysctl-name "hw.pagesize_compat") (sysctl-name "hw.ncpu") (sysctl-name "hw.activecpu") (sysctl-name "hw.physicalcpu") (sysctl-name "hw.logicalcpu") (sysctl-name "hw.memsize") (sysctl-name "kern.ostype") (sysctl-name "kern.osrelease") (sysctl-name "kern.osversion") (sysctl-name "kern.version") (sysctl-name "kern.argmax"))
(allow signal (target self))
(allow file-read-metadata)
(allow file-read* (literal "/") (subpath "/System/Library") (subpath "/System/iOSSupport") (subpath "/System/Cryptexes") (subpath "/System/Volumes/Preboot/Cryptexes") (subpath "/usr/bin") (subpath "/usr/sbin") (subpath "/usr/lib") (subpath "/usr/libexec") (subpath "/usr/share") (subpath "/bin") (subpath "/sbin") (subpath "/Library/Developer") (subpath "/opt/homebrew/bin") (subpath "/opt/homebrew/sbin") (subpath "/opt/homebrew/lib") (subpath "/opt/homebrew/opt") (subpath "/opt/homebrew/Cellar") (subpath "/opt/homebrew/share") (subpath "/opt/homebrew/include") (subpath "/opt/homebrew/libexec") (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random") (literal "/private/etc/localtime") (literal "/private/etc/ssl/openssl.cnf") (subpath {root}))
(allow file-write* (subpath {root}) (literal "/dev/null"))
(allow file-read* file-write* (subpath {build}))
(deny file-write* (literal {build}))
{toolchain_policy}
{read_only_policy}
(deny file-link)
{excluded_access}
(deny file-read-data file-write* (require-all (regex #"/\.[^/]+(/|$)") (require-not (require-any {private_reads}))))
(deny file-read-data file-write* (regex #"{credential_paths}" #"{credential_extensions}"))
"#
    ))
}

/// Fail closed on non-macOS. Linux needs its own verified OS sandbox before
/// commands are enabled there. Read/edit tools have their own directory guards.
pub fn run(
    workspace: &Workspace,
    spec: &CommandSpec,
    cap: usize,
    cancel: &AtomicBool,
) -> io::Result<CommandOutput> {
    run_with_exclusions(workspace, spec, cap, cancel, &Exclusions::default())
}

pub fn run_with_exclusions(
    workspace: &Workspace,
    spec: &CommandSpec,
    cap: usize,
    cancel: &AtomicBool,
    exclusions: &Exclusions,
) -> io::Result<CommandOutput> {
    run_in_session(
        workspace,
        spec,
        cap,
        cancel,
        exclusions,
        &BuildSession::default(),
    )
}

/// Reuse only this live session's freshly created build state. Exclusions and
/// the OS boundary remain authoritative even if argv changes build variables.
pub fn run_in_session(
    workspace: &Workspace,
    spec: &CommandSpec,
    cap: usize,
    cancel: &AtomicBool,
    exclusions: &Exclusions,
    session: &BuildSession,
) -> io::Result<CommandOutput> {
    run_in_session_observing(workspace, spec, cap, cancel, exclusions, session, false)
        .map(|(output, _)| output)
}

/// Detect file changes with metadata only, reusing the bounded preflight scan.
/// Read-only sandbox commands need no second scan. Hidden/credential paths and
/// protected build trees never enter the manifest; directory mtimes are omitted
/// so private journals and build outputs cannot make a read look like an edit.
pub(crate) fn run_in_session_observing(
    workspace: &Workspace,
    spec: &CommandSpec,
    cap: usize,
    cancel: &AtomicBool,
    exclusions: &Exclusions,
    session: &BuildSession,
    detect_changes: bool,
) -> io::Result<(CommandOutput, bool)> {
    spec.validate()?;
    if cap == 0 || cap > 64 * 1024 {
        return Err(io::Error::other("invalid command output cap"));
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (workspace, cancel, exclusions, session, detect_changes);
        Err(io::Error::other(
            "command sandbox is supported only on macOS",
        ))
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        use std::os::unix::process::{CommandExt, ExitStatusExt};
        if cancel.load(Ordering::Relaxed) {
            return Ok((
                CommandOutput {
                    outcome: Outcome::Cancelled,
                    stdout: String::new(),
                    stderr: String::new(),
                    stdout_bytes: 0,
                    stderr_bytes: 0,
                    stdout_omitted: 0,
                    stderr_omitted: 0,
                    elapsed_ms: 0,
                    exit_code: None,
                    signal: None,
                },
                false,
            ));
        }
        workspace.verify_root()?;
        let read_only_argv = spec.read_only_argv();
        let git = read_only_argv
            .as_ref()
            .is_some_and(|argv| argv[0] == "/usr/bin/git");
        let detect_changes = detect_changes && read_only_argv.is_none();
        let mut before = WorkspaceState::new();
        if git {
            verify_tree_with_git(workspace.root(), cancel, exclusions, true, None)?;
        } else if detect_changes {
            verify_tree_with_git(
                workspace.root(), cancel, exclusions, false, Some(&mut before),
            )?;
        } else {
            verify_tree(workspace.root(), cancel, exclusions)?;
        }
        let mut build_guard = session.prepare(workspace.root())?;
        let build = build_guard.as_mut().unwrap();
        let profile = profile(
            workspace.root(),
            exclusions,
            &build.path,
            session.toolchain.as_deref(),
            read_only_argv.is_some(),
            git,
        )?;
        let mut command = Command::new("/usr/bin/sandbox-exec");
        command
            .args(["-p", &profile, "--"])
            .args(read_only_argv.as_ref().unwrap_or(&spec.argv))
            .current_dir(workspace.root())
            .env_clear()
            .env("LANG", "en_US.UTF-8")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        build.environment(&mut command, session.toolchain.as_deref());
        // SAFETY: async-signal-safe system calls only between fork and exec.
        unsafe {
            command.pre_exec(|| {
                for (resource, value) in [(0, 120u64), (1, 64 * 1024 * 1024), (7, 1024), (8, 128)] {
                    let limit = Rlimit {
                        current: value,
                        maximum: value,
                    };
                    if setrlimit(resource, &limit) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        workspace.verify_root()?;
        let child = command
            .spawn()
            .map_err(|_| io::Error::other("cannot start sandboxed command"))?;
        let mut owned = OwnedChild(child);
        let mut stdout = owned.0.stdout.take().unwrap();
        let mut stderr = owned.0.stderr.take().unwrap();
        for fd in [stdout.as_raw_fd(), stderr.as_raw_fd()] {
            let flags = unsafe { fcntl(fd, 3) }; // F_GETFL
            if flags < 0 || unsafe { fcntl(fd, 4, flags | 4) } < 0 {
                return Err(io::Error::other("cannot make command pipes nonblocking"));
            }
        }
        // Reserve stderr separately so verbose stdout cannot crowd out the
        // compiler's final diagnostics. Both pipes keep being drained after
        // the display fills; only cancellation/deadline/resource limits stop
        // a running command. Capture memory remains bounded by the display cap.
        let mut out = Capture::new(cap / 2);
        let mut errors = Capture::new(cap - cap / 2);
        let mut out_eof = false;
        let mut err_eof = false;
        let mut pipes = [stdout.as_raw_fd(), stderr.as_raw_fd()].map(|fd| PollFd {
            fd,
            events: 1, // POLLIN; hangup/error events are reported automatically.
            revents: 0,
        });
        let start = Instant::now();
        let mut exit_code = None;
        let mut signal = None;
        let mut outcome = loop {
            if cancel.load(Ordering::Relaxed) {
                break Outcome::Cancelled;
            }
            if start.elapsed() >= Duration::from_millis(spec.timeout_ms) {
                break Outcome::Timeout;
            }
            drain(&mut stdout, &mut out, &mut out_eof)?;
            drain(&mut stderr, &mut errors, &mut err_eof)?;
            if let Some(status) = owned.0.try_wait()? {
                exit_code = status.code();
                signal = status.signal();
                // Kill leftover group members even if the direct child exited.
                owned.kill_group();
                break Outcome::Exited(status.code());
            }
            // Wake as soon as output is ready. Retain the 10 ms cancellation
            // bound, and never wait beyond the command's remaining deadline.
            // Disable EOF pipes so their persistent hangup cannot spin.
            if out_eof {
                pipes[0].fd = -1;
            }
            if err_eof {
                pipes[1].fd = -1;
            }
            let remaining = Duration::from_millis(spec.timeout_ms).saturating_sub(start.elapsed());
            // SAFETY: macOS poll receives two live, correctly laid-out entries.
            if unsafe { poll(pipes.as_mut_ptr(), 2, remaining.as_millis().min(10) as i32) }
                < 0
                && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted
            {
                return Err(io::Error::other("cannot wait for command output"));
            }
        };
        drop(owned); // Kill group and reap before returning any outcome.
        drain(&mut stdout, &mut out, &mut out_eof)?;
        drain(&mut stderr, &mut errors, &mut err_eof)?;
        build.verify()?;
        if build.limits_exceeded()? {
            outcome = Outcome::BuildLimit;
        }
        let changed = if detect_changes && matches!(outcome, Outcome::Exited(_)) {
            workspace.verify_root()?;
            let mut after = WorkspaceState::new();
            verify_tree_with_git(
                workspace.root(), cancel, exclusions, false, Some(&mut after),
            )?;
            before != after
        } else {
            false
        };
        Ok((
            CommandOutput {
                outcome,
                stdout: out.render(),
                stderr: errors.render(),
                stdout_bytes: out.total,
                stderr_bytes: errors.total,
                stdout_omitted: out.omitted(),
                stderr_omitted: errors.omitted(),
                elapsed_ms: start.elapsed().as_millis().min(u64::MAX as u128) as u64,
                exit_code,
                signal,
            },
            changed,
        ))
    }
}
#[cfg(target_os = "macos")]
struct Capture {
    head: Vec<u8>,
    tail: std::collections::VecDeque<u8>,
    cap: usize,
    total: u64,
}
#[cfg(target_os = "macos")]
impl Capture {
    fn new(cap: usize) -> Self {
        Self {
            head: Vec::new(),
            tail: std::collections::VecDeque::new(),
            cap,
            total: 0,
        }
    }
    fn append(&mut self, bytes: &[u8]) {
        self.total = self.total.saturating_add(bytes.len() as u64);
        let head_cap = self.cap / 4;
        let keep = bytes.len().min(head_cap.saturating_sub(self.head.len()));
        self.head.extend_from_slice(&bytes[..keep]);
        let tail_cap = self.cap - head_cap;
        let rest = &bytes[keep..];
        if rest.len() >= tail_cap {
            self.tail.clear();
            self.tail.extend(&rest[rest.len() - tail_cap..]);
        } else {
            let discard = (self.tail.len() + rest.len()).saturating_sub(tail_cap);
            self.tail.drain(..discard);
            self.tail.extend(rest);
        }
    }
    fn omitted(&self) -> u64 {
        self.total
            .saturating_sub((self.head.len() + self.tail.len()) as u64)
    }
    fn render(&self) -> String {
        let mut bytes = self.head.clone();
        if self.omitted() > 0 {
            bytes.extend(format!("\n[... {} middle bytes omitted ...]\n", self.omitted()).bytes());
        }
        bytes.extend(self.tail.iter());
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
#[cfg(target_os = "macos")]
fn drain(pipe: &mut impl Read, capture: &mut Capture, eof: &mut bool) -> io::Result<()> {
    if *eof {
        return Ok(());
    }
    let mut buf = [0; 4096];
    // Bounded even for a continuously writing producer; caller polls deadlines.
    for _ in 0..32 {
        match pipe.read(&mut buf) {
            Ok(0) => {
                *eof = true;
                break;
            }
            Ok(n) => {
                capture.append(&buf[..n]);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(io::Error::other("command pipe failed")),
        }
    }
    Ok(())
}
#[cfg(target_os = "macos")]
type WorkspaceState = std::collections::BTreeMap<std::path::PathBuf, crate::tools::FileStamp>;

#[cfg(target_os = "macos")]
pub(super) fn verify_tree(
    root: &std::path::Path,
    cancel: &AtomicBool,
    exclusions: &Exclusions,
) -> io::Result<()> {
    verify_tree_with_git(root, cancel, exclusions, false, None)
}
#[cfg(target_os = "macos")]
fn verify_tree_with_git(
    root: &std::path::Path,
    cancel: &AtomicBool,
    exclusions: &Exclusions,
    git: bool,
    mut manifest: Option<&mut WorkspaceState>,
) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let mut dirs = vec![root.to_path_buf()];
    let mut seen = 0;
    let refusal = |path: &std::path::Path, reason: &str| {
        io::Error::other(format!(
            "command workspace scan refused at {:?}: {reason}",
            path.strip_prefix(root).unwrap_or(path)
        ))
    };
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).map_err(|_| refusal(&dir, "cannot read directory"))? {
            let entry = entry.map_err(|_| refusal(&dir, "cannot read directory entry"))?;
            let path = entry.path();
            seen += 1;
            if cancel.load(Ordering::Relaxed) {
                return Err(refusal(&path, "cancelled"));
            }
            if seen > 20_000 {
                return Err(refusal(
                    &path,
                    "more than 20,000 scanned entries; use --shell-exclude NAME to protect and skip a build/dependency directory",
                ));
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                return Err(refusal(&path, "non-UTF-8 filename"));
            };
            let in_git = git
                && path.strip_prefix(root).is_ok_and(|p| {
                    p.components().next().is_some_and(|c| {
                        c.as_os_str()
                            .to_str()
                            .is_some_and(|s| s.eq_ignore_ascii_case(".git"))
                    })
                });
            // Only dot-prefixed trees are denied at every depth by the OS.
            // A directory ending .pem is not opaque: aliases under it still
            // need scanning. Git metadata is read-only but must be scanned too.
            if name.starts_with('.') && !in_git {
                continue;
            }
            let meta = path
                .symlink_metadata()
                .map_err(|_| refusal(&path, "cannot inspect entry"))?;
            // Inspect the excluded root itself so a symlink with this name can
            // never hide a writable alias. Descendants are OS-protected and do
            // not consume the scan budget (Cargo/pnpm hard links are common).
            if meta.is_dir() && exclusions.contains(name) {
                continue;
            }
            // No hardlinked writable aliases or special devices. Symlinks are
            // denied here too, rather than relying solely on sandbox resolution.
            if meta.is_symlink()
                || meta.is_file() && meta.nlink() != 1
                || !meta.is_file() && !meta.is_dir()
            {
                let reason = if meta.is_symlink() {
                    "symbolic link"
                } else if meta.is_file() {
                    "hard-linked writable file (link count is not 1)"
                } else {
                    "special file"
                };
                return Err(refusal(&path, reason));
            }
            if meta.is_dir() {
                dirs.push(path);
            } else if let Some(manifest) = manifest.as_deref_mut() {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|_| refusal(&path, "outside workspace"))?;
                if relative.components().all(|part| {
                    part.as_os_str()
                        .to_str()
                        .is_some_and(crate::tools::allowed_component)
                }) {
                    manifest.insert(
                        relative.to_path_buf(),
                        crate::tools::FileStamp::from_metadata(&meta),
                    );
                }
            }
        }
    }
    Ok(())
}
#[cfg(target_os = "macos")]
#[repr(C)]
struct Rlimit {
    current: u64,
    maximum: u64,
}
#[cfg(target_os = "macos")]
const _: () = {
    assert!(std::mem::size_of::<Rlimit>() == 16);
    assert!(std::mem::align_of::<Rlimit>() == 8);
    assert!(std::mem::offset_of!(Rlimit, maximum) == 8);
};
#[cfg(target_os = "macos")]
#[repr(C)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
    fn fcntl(fd: i32, command: i32, ...) -> i32;
    fn poll(fds: *mut PollFd, nfds: u32, timeout: i32) -> i32;
    fn setrlimit(resource: i32, limit: *const Rlimit) -> i32;
}
#[cfg(target_os = "macos")]
struct OwnedChild(std::process::Child);
#[cfg(target_os = "macos")]
impl OwnedChild {
    fn kill_group(&mut self) {
        unsafe {
            kill(-(self.0.id() as i32), 9);
        }
        let _ = self.0.kill();
    }
}
#[cfg(target_os = "macos")]
impl Drop for OwnedChild {
    fn drop(&mut self) {
        self.kill_group();
        let _ = self.0.wait();
    }
}
