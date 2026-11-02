//! Every subprocess the CLI starts. [`Exec`] describes one command, the
//! [`Runner`] trait runs it (tests substitute fakes), and [`SystemRunner`] owns
//! the real process group, timeouts, cancellation and retained logs.
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Maximum stdout captured from one command. Probes must split larger data
/// before capture, rather than raise this bound for an unbounded reply.
pub(crate) const MAX_CAPTURE_BYTES: u64 = 16 * 1024 * 1024;

/// A child holds a copy of every descriptor from its spawn to its exec, even
/// with CLOEXEC. A flock that only closes can stay held by that copy. Serialize
/// process-spawning tests with the worker lock test. `state::Lock` unlocks at
/// drop, so such a copy cannot keep it.
#[cfg(test)]
pub(crate) static PROCESS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the process-test lock, recovering from poisoning. The lock guards
/// no data, so a panicking test cannot leave inconsistent state behind it;
/// without recovery its panic would fail every later process test with
/// PoisonError and hide their real results.
#[cfg(test)]
pub(crate) fn process_test_lock() -> std::sync::MutexGuard<'static, ()> {
    PROCESS_TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

/// One subprocess to run: what to start, with which input and environment, and
/// how long it may take. `label` names it in progress lines and errors; argv and
/// the environment are never printed.
#[derive(Clone, Debug)]
pub struct Exec {
    pub label: String,
    pub program: String,
    pub args: Vec<String>,
    pub stdin: Vec<u8>,
    pub env: BTreeMap<String, String>,
    pub unset: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub timeout: Duration,
    /// A durable build job directory; when set, `build_job` runs it instead.
    pub job: Option<PathBuf>,
    /// Whether Ctrl-C stops it; cleanup commands always run to completion.
    pub cancellable: bool,
    /// A readiness probe whose failure is expected while waiting, so its logs are
    /// not retained.
    pub transient_probe: bool,
    /// Overwrite this private file with the latest probe's output instead.
    pub probe_log: Option<PathBuf>,
}

impl Exec {
    /// A cancellable command with no input, the current environment and a 60 s
    /// timeout.
    pub fn new(label: &str, program: &str, args: Vec<String>) -> Self {
        Self {
            label: label.into(),
            program: program.into(),
            args,
            stdin: Vec::new(),
            env: BTreeMap::new(),
            unset: Vec::new(),
            cwd: None,
            timeout: Duration::from_secs(60),
            job: None,
            cancellable: true,
            transient_probe: false,
            probe_log: None,
        }
    }
    /// The [`Command`] with these arguments, environment changes and folder.
    pub(crate) fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args).envs(&self.env);
        for key in &self.unset {
            command.env_remove(key);
        }
        if let Some(cwd) = &self.cwd {
            command.current_dir(cwd);
        }
        command
    }
    /// The program and arguments, shell-quoted, safe to print.
    pub fn shell_line(&self) -> String {
        let args = std::iter::once(&self.program).chain(self.args.iter()).map(|s| quote(s)).collect::<Vec<_>>();
        crate::text::terminal_text(&args.join(" ")).into_owned()
    }
    /// Print the shell-quoted command line that `--dry-run` would run.
    pub fn preview(&self) -> Result<(), String> {
        if let Some(cwd) = &self.cwd {
            crate::output::line(format_args!("# folder: {}", crate::text::terminal_text(&cwd.to_string_lossy())))?;
        }
        crate::output::line(format_args!("$ {}", self.shell_line()))?;
        if !self.stdin.is_empty() {
            crate::output::line(format_args!("# private stdin; environment values and ownership secrets are hidden"))?;
            if let Ok(fields) = crate::text::parse(&String::from_utf8_lossy(&self.stdin)) {
                for (key, value) in fields {
                    if key == "token" || key.starts_with("env.") || key.starts_with("env_unset.") {
                        continue;
                    }
                    crate::output::line(format_args!("# {key}={}", crate::text::terminal_text(&value.replace(['\r', '\n'], " "))))?;
                }
            }
        }
        Ok(())
    }
}

/// Whether a command succeeded, and its standard output.
#[derive(Clone, Debug, Default)]
pub struct Output {
    pub success: bool,
    pub stdout: String,
}

/// Runs every [`Exec`]. [`SystemRunner`] starts real processes (or previews them
/// on `--dry-run`); tests supply scripted runners.
pub trait Runner {
    /// Whether the user pressed Ctrl-C since the lifecycle command started.
    fn interrupted(&self) -> bool {
        interrupted()
    }
    /// Settle build jobs a previous, interrupted command left under `root`.
    fn recover_jobs(&mut self, root: &std::path::Path) -> Result<(), String> {
        crate::build_job::recover(root)
    }
    /// Run a cleanup step, which Ctrl-C does not cancel.
    fn cleanup(&mut self, exec: &Exec) -> Result<Output, String> {
        self.run(exec)
    }
    /// Run to completion and capture standard output.
    fn run(&mut self, exec: &Exec) -> Result<Output, String>;
    /// Run and hand each output line to `line`; an error from `line` stops the
    /// command.
    fn stream(&mut self, exec: &Exec, line: &mut dyn FnMut(&str) -> Result<(), String>) -> Result<Output, String> {
        let output = self.run(exec)?;
        for value in output.stdout.lines() {
            line(value)?;
        }
        Ok(output)
    }
    /// Independent jobs; the default keeps scripted runners deterministic.
    fn parallel(&mut self, first: &Exec, second: &Exec) -> Result<(Output, Output), String> {
        let first = self.run(first);
        let second = self.run(second);
        Ok((first?, second?))
    }
    /// Run attached to this terminal (chat); returns whether it exited successfully.
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String>;
    /// Fail unless `port` on the loopback interface is free.
    fn port_free(&mut self, port: u16) -> Result<(), String>;
    /// Start a detached local process appending output to `log`; returns its PID.
    /// Scripted runners record the command and may answer a PID on stdout.
    fn spawn(&mut self, exec: &Exec, log: &std::path::Path) -> Result<u32, String> {
        let _ = log;
        Ok(self.run(exec)?.stdout.trim().parse().unwrap_or(0))
    }
    /// Wait between polls; a real runner wakes early on Ctrl-C.
    fn pause(&mut self, duration: Duration);
    /// Ask before restarting a running engine on a new build; `false` without a
    /// terminal.
    fn confirm_restart(&mut self) -> Result<bool, String> {
        Ok(false)
    }
    /// An independent runner for one worker thread, or None to keep the work
    /// on this thread. The default keeps scripted runners deterministic.
    fn worker(&self) -> Option<Box<dyn Runner + Send>> {
        None
    }
    /// Whether standard input is a terminal a chat can use. Scripted runners act interactive.
    fn terminal(&self) -> bool {
        true
    }
}

impl<T: Runner + ?Sized> Runner for Box<T> {
    fn interrupted(&self) -> bool { (**self).interrupted() }
    fn recover_jobs(&mut self, root: &std::path::Path) -> Result<(), String> { (**self).recover_jobs(root) }
    fn cleanup(&mut self, exec: &Exec) -> Result<Output, String> { (**self).cleanup(exec) }
    fn run(&mut self, exec: &Exec) -> Result<Output, String> { (**self).run(exec) }
    fn stream(&mut self, exec: &Exec, line: &mut dyn FnMut(&str) -> Result<(), String>) -> Result<Output, String> { (**self).stream(exec, line) }
    fn parallel(&mut self, first: &Exec, second: &Exec) -> Result<(Output, Output), String> { (**self).parallel(first, second) }
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> { (**self).interactive(exec) }
    fn port_free(&mut self, port: u16) -> Result<(), String> { (**self).port_free(port) }
    fn spawn(&mut self, exec: &Exec, log: &std::path::Path) -> Result<u32, String> { (**self).spawn(exec, log) }
    fn pause(&mut self, duration: Duration) { (**self).pause(duration) }
    fn confirm_restart(&mut self) -> Result<bool, String> { (**self).confirm_restart() }
    fn worker(&self) -> Option<Box<dyn Runner + Send>> { (**self).worker() }
    fn terminal(&self) -> bool { (**self).terminal() }
}

/// The real runner. With `dry`, commands are printed instead of started.
pub struct SystemRunner {
    pub dry: bool,
}

/// Report only a bounded operation label, never argv, environment or child output.
pub(crate) struct Progress<'a> {
    label: &'a str,
    next: Duration,
}
impl<'a> Progress<'a> {
    /// Progress for `label`, first reported after 30 s.
    pub(crate) fn new(label: &'a str) -> Self {
        Self { label, next: Duration::from_secs(30) }
    }
    fn line(&mut self, elapsed: Duration) -> Option<String> {
        if elapsed < self.next { return None; }
        self.next = elapsed + Duration::from_secs(30);
        Some(format!("Vade: {} still running ({}s elapsed).", crate::text::argument(self.label), elapsed.as_secs()))
    }
    /// Print a "still running" line on stderr at most every 30 s.
    pub(crate) fn report(&mut self, elapsed: Duration) {
        if let Some(line) = self.line(elapsed) {
            crate::output::stderr_line(format_args!("{line}"));
        }
    }
}

impl Runner for SystemRunner {
    // A dry run previews every command in order on this thread.
    fn worker(&self) -> Option<Box<dyn Runner + Send>> {
        (!self.dry).then(|| Box::new(SystemRunner { dry: false }) as Box<dyn Runner + Send>)
    }
    fn terminal(&self) -> bool {
        use std::io::IsTerminal;
        std::io::stdin().is_terminal()
    }
    fn parallel(&mut self, first: &Exec, second: &Exec) -> Result<(Output, Output), String> {
        if self.dry { return Ok((self.run(first)?, self.run(second)?)); }
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| SystemRunner { dry: false }.run(first));
            let second = self.run(second);
            let first = worker.join().map_err(|_| "Install worker stopped unexpectedly; inspect retained logs.")?;
            Ok((first?, second?))
        })
    }
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        if exec.cancellable && interrupted() {
            return Err(
                "Interrupted. No further command started; retry run or use vade down.".into(),
            );
        }
        if self.dry {
            exec.preview()?;
            return Ok(Output {
                success: true,
                stdout: String::new(),
            });
        }
        if let Some(job) = &exec.job {
            return crate::build_job::run(job);
        }
        // Temporary files keep long compiler output out of RAM and prevent a pipe deadlock.
        let temp_root = std::fs::canonicalize(std::env::temp_dir())
            .map_err(|_| "Cannot open the system temporary directory. Check TMPDIR.")?;
        let scratch = temp_root.join(format!("vade-command-{}", crate::text::random_hex(12)?));
        crate::text::private_dir(&scratch)?;
        let result = (|| {
            let outpath = scratch.join("stdout");
            let errpath = scratch.join("stderr");
            let out = private_file(&outpath)?;
            let err = private_file(&errpath)?;
            let mut command = owned_command(exec);
            let mut child = command
                .stdin(Stdio::piped())
                .stdout(out)
                .stderr(err)
                .spawn()
                .map_err(|_| {
                    format!(
                        "Cannot run {}. Check that its program is installed.",
                        exec.label
                    )
                })?;
            let write = child
                .stdin
                .take()
                .expect("piped stdin")
                .write_all(&exec.stdin);
            if write.is_err() {
                stop_job(&mut child);
                let _ = child.wait();
                return Err(format!(
                    "Cannot send {}. Check the connection and try again.",
                    exec.label
                ));
            }
            let began = Instant::now();
            let until = began + exec.timeout;
            let mut progress = Progress::new(&exec.label);
            let status = loop {
                let status = match child.try_wait() {
                    Ok(status) => status,
                    Err(_) => {
                        stop_job(&mut child);
                        let _ = child.wait();
                        return Err("Cannot inspect the owned command; its job was stopped. Check the retained command logs and run vade status before retrying.".into());
                    }
                };
                match status {
                    Some(status) => break status,
                    None if (exec.cancellable && interrupted()) || Instant::now() >= until => {
                        stop_job(&mut child);
                        let _ = child.wait();
                        return Err(format!(
                            "{} timed out or was interrupted. Its owned job stopped; run vade status before retrying.",
                            exec.label
                        ));
                    }
                    None => {
                        progress.report(began.elapsed());
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            };
            // Never echo arbitrary stderr: remote tools may include inherited credentials.
            let metadata =
                std::fs::metadata(&outpath).map_err(|_| "Cannot read command output. Check the retained log directory permissions and free space.")?;
            if metadata.len() > MAX_CAPTURE_BYTES {
                return Err("Command output is too large. Inspect local source build logs.".into());
            }
            let stdout = std::fs::read_to_string(outpath)
                .map_err(|_| "Command output is not readable text. Inspect the retained stdout log; the command must emit UTF-8 text.")?;
            Ok(Output {
                success: status.success(),
                stdout,
            })
        })();
        // A readiness loop overwrites one private diagnostic instead of
        // discarding every failure or retaining one directory per retry.
        if let Some(path) = &exec.probe_log {
            use std::io::Read;
            let mut diagnostic = format!("operation={}\n", exec.label);
            for name in ["stdout", "stderr"] {
                let mut bytes = Vec::new();
                if let Ok(file) = std::fs::File::open(scratch.join(name)) {
                    file.take(64 * 1024).read_to_end(&mut bytes)
                        .map_err(|_| "Cannot read readiness diagnostic.")?;
                }
                diagnostic.push_str(&format!("\n{name} (up to 64 KiB):\n{}", String::from_utf8_lossy(&bytes)));
            }
            crate::text::atomic_write(path, &diagnostic)?;
        }
        if result.as_ref().is_ok_and(|output| output.success || exec.transient_probe) {
            let _ = std::fs::remove_dir_all(&scratch);
        } else {
            crate::output::stderr_line(format_args!(
                "{} logs retained privately at {}",
                crate::text::terminal_text(&exec.label),
                crate::text::terminal_text(&scratch.to_string_lossy())
            ));
        }
        result
    }
    fn stream(&mut self, exec: &Exec, line: &mut dyn FnMut(&str) -> Result<(), String>) -> Result<Output, String> {
        if interrupted() {
            return Err("Interrupted before streaming request. No request was sent; rerun the command when ready.".into());
        }
        if self.dry {
            return self.run(exec);
        }
        let scratch = std::fs::canonicalize(std::env::temp_dir())
            .map_err(|_| "Cannot open temporary directory. Set TMPDIR to an existing writable private directory.")?
            .join(format!("vade-stream-{}", crate::text::random_hex(12)?));
        crate::text::private_dir(&scratch)?;
        let mut command = owned_command(exec);
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(private_file(&scratch.join("stderr"))?)
            .spawn()
            .map_err(|_| "Cannot start streaming request. Check that curl is installed and executable, then retry.")?;
        if child.stdin.take().unwrap().write_all(&exec.stdin).is_err() {
            stop_job(&mut child);
            let _ = child.wait();
            return Err("Cannot send streaming request. Inspect vade logs and the connection before retrying; no automatic replay was attempted.".into());
        }
        let stdout = child.stdout.take().unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            for value in std::io::BufReader::new(stdout).lines() {
                if send.send(value).is_err() {
                    break;
                }
            }
        });
        let deadline = Instant::now() + exec.timeout;
        let mut output = String::new();
        let mut error = None;
        loop {
            if interrupted() || Instant::now() >= deadline {
                error = Some("Streaming request timed out or was interrupted.".to_string());
                break;
            }
            match receive.recv_timeout(
                deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(100)),
            ) {
                Ok(Ok(value)) if output.len() + value.len() < 16 * 1024 * 1024 => {
                    if let Err(cause) = line(&value) { error = Some(cause); break; }
                    output.push_str(&value);
                    output.push('\n');
                }
                Ok(_) => {
                    error = Some("Streaming output is unreadable or too large.".to_string());
                    break;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
        if let Some(error) = error {
            stop_job(&mut child);
            let _ = child.wait();
            // Closing the receiver lets a still-draining reader leave promptly.
            drop(receive);
            let _ = reader.join();
            return Err(format!("{error} Logs retained at {}", scratch.display()));
        }
        let _ = reader.join();
        // EOF does not prove the process exited; preserve the same outer deadline.
        let status = loop {
            match child.try_wait().map_err(|_| "Cannot inspect stream job. Inspect the retained stream logs and run vade status before retrying.")? {
                Some(status) => break status,
                None if interrupted() || Instant::now() >= deadline => {
                    stop_job(&mut child);
                    let _ = child.wait();
                    return Err("Streaming request timed out after EOF. Inspect the retained stream logs and vade status before retrying.".into());
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        };
        if status.success() {
            let _ = std::fs::remove_dir_all(&scratch);
        } else {
            crate::output::stderr_line(format_args!("Stream logs retained at {}", crate::text::terminal_text(&scratch.to_string_lossy())));
        }
        Ok(Output {
            success: status.success(),
            stdout: output,
        })
    }

    fn cleanup(&mut self, exec: &Exec) -> Result<Output, String> {
        let mut exec = exec.clone();
        exec.cancellable = false;
        self.run(&exec)
    }

    fn interactive(&mut self, exec: &Exec) -> Result<bool, String> {
        if self.dry {
            exec.preview()?;
            return Ok(true);
        }
        exec.command()
            .status()
            .map(|s| s.success())
            .map_err(|_| "Cannot open the harness. Run vade install to build it.".into())
    }
    fn spawn(&mut self, exec: &Exec, log: &std::path::Path) -> Result<u32, String> {
        if self.dry {
            exec.preview()?;
            crate::output::line(format_args!("# detached; output appended to {}", crate::text::terminal_text(&log.to_string_lossy())))?;
            return Ok(0);
        }
        if let Some(dir) = log.parent() {
            crate::text::private_dir(dir)?;
        }
        use std::os::unix::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new().create(true).append(true).mode(0o600).open(log)
            .map_err(|_| format!("Cannot open the {} log. Check the state folder.", exec.label))?;
        let error = file.try_clone().map_err(|_| format!("Cannot open the {} log.", exec.label))?;
        // Its own process group: a later Ctrl-C in this terminal does not reach it.
        let child = exec.command().stdin(Stdio::null()).stdout(file).stderr(error).process_group(0).spawn()
            .map_err(|_| format!("Cannot start {}. Run vade install.", exec.label))?;
        Ok(child.id())
    }
    fn port_free(&mut self, port: u16) -> Result<(), String> {
        if self.dry {
            crate::output::line(format_args!("# require free port {port} on this computer; preserve anything already using it"))?;
            return Ok(());
        }
        std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).map(|_| ())
            .map_err(|_| format!("Port {port} on this computer is in use. Choose another local_port in config; nothing was stopped."))
    }
    fn pause(&mut self, duration: Duration) {
        if !self.dry {
            let until = Instant::now() + duration;
            while !interrupted() && Instant::now() < until {
                std::thread::sleep(
                    Duration::from_millis(20).min(until.saturating_duration_since(Instant::now())),
                );
            }
        }
    }
    fn confirm_restart(&mut self) -> Result<bool, String> {
        use std::io::IsTerminal;
        if self.dry || !std::io::stdin().is_terminal() {
            return Ok(false);
        }
        crate::output::text(format_args!("Restart the engine with this build? [y/N] "))?;
        std::io::stdout()
            .flush()
            .map_err(|_| "Cannot display restart choice. Run vade down and vade up when ready.")?;
        // Ctrl-C or SIGTERM ends the question at once; the engine keeps running.
        let answer = crate::launch::read_prompt_line("Restart question interrupted; the running engine was kept. Run vade down and vade up when ready.")?;
        Ok(matches!(
            answer.trim().to_ascii_lowercase().as_str(),
            "y" | "yes"
        ))
    }
}

fn private_file(path: &std::path::Path) -> Result<std::fs::File, String> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)
        .map_err(|_| "Cannot create private command output. Check the temporary directory.".into())
}

/// Quote one word for a POSIX shell; safe words stay bare.
pub fn quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-./:@,=+".contains(&b))
    {
        return value.into();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

/// A shell word for a remote path, expanding a leading `~/` or `{home}/` to
/// `"$HOME"/`.
pub fn remote_path(value: &str) -> String {
    if let Some(relative) = value
        .strip_prefix("~/")
        .or_else(|| value.strip_prefix("{home}/"))
    {
        format!("\"$HOME\"/{}", quote(relative))
    } else {
        quote(value)
    }
}

/// Run `command` on the SSH host `alias` with [`ssh_options`]. When `alias`
/// names this computer ([`this_computer`]), run the same command string as a
/// local child process instead ([`local`]): a fresh Spark has no SSH key for
/// itself.
pub fn ssh(alias: &str, command: String, label: &str) -> Exec {
    if this_computer(alias) {
        return local(command, label);
    }
    let mut args = ssh_options();
    args.extend([alias.into(), command]);
    Exec::new(label, "ssh", args)
}

/// One more connection that runs only `true`, with ssh's own standard error on
/// standard output, so a failed SSH step can show why. No other command shows
/// its standard error ([`SystemRunner::run`] keeps that rule). This one fails
/// only on exit status 255, the ssh client's own failure: after a login, a remote
/// shell can print anything, and that text never counts. `None` for this computer.
pub fn ssh_check(alias: &str) -> Option<Exec> {
    if this_computer(alias) { return None; }
    let mut args = vec!["-c".to_owned(), "ssh \"$@\" 2>&1 >/dev/null; test $? -ne 255".to_owned(), "sh".to_owned()];
    args.extend(ssh_options());
    args.extend([alias.to_owned(), "true".to_owned()]);
    let mut exec = Exec::new(&format!("SSH check of {alias}"), "sh", args);
    // A failure here is the expected case, and this text is all it has.
    exec.transient_probe = true;
    Some(exec)
}

/// The host a failed `ssh` or `rsync` command tried to reach.
pub(crate) fn ssh_target(exec: &Exec) -> Option<&str> {
    match exec.program.as_str() {
        "ssh" => exec.args.iter().rev().nth(1).map(String::as_str),
        "rsync" => exec.args.last()?.strip_suffix(":./"),
        _ => None,
    }
}

/// Why SSH to a machine stopped, and the one step that fixes it.
#[derive(Clone, Debug, PartialEq)]
pub struct SshStop { pub error: String, pub fix: String }
impl SshStop {
    /// The last line that `ssh alias true` printed, and the fix for that cause.
    /// Vade only names the command: it never edits ~/.ssh.
    pub fn from_output(alias: &str, printed: &str) -> Option<Self> {
        let line = printed.lines().rev().map(str::trim).find(|line| !line.is_empty())?;
        let line: String = crate::text::terminal_text(line).chars().take(200).collect();
        let stop = if line.ends_with('.') { "" } else { "." };
        let host = alias.rsplit('@').next().unwrap_or(alias);
        let (note, fix) = if printed.contains("IDENTIFICATION HAS CHANGED") {
            (" Vade never edits ~/.ssh.", format!("Ask who runs {alias} why its key changed. If that is expected, run ssh {alias} true, run the ssh-keygen -R command that it prints, run ssh {alias} true again and accept the new key, then run vade doctor."))
        } else if line.contains("Host key verification failed") {
            // ssh shows the key to check, and it reads ~/.ssh/config. ssh-keyscan does neither.
            (" Vade never edits ~/.ssh.", format!("Run ssh {alias} true and accept the key, or run ssh-keyscan -H {host} >> ~/.ssh/known_hosts if {host} is the real host name, then run vade doctor."))
        } else if line.contains("Permission denied") {
            (" Vade never edits ~/.ssh.", format!("Run ssh-copy-id {alias}, then run vade doctor."))
        } else {
            ("", format!("Run ssh {alias} true to see the full SSH error, then run vade doctor."))
        };
        Some(Self { error: format!("ssh {alias} failed: {line}{stop}{note}"), fix })
    }
}

/// Run `command` through `sh -c` on this computer, as SSH would run it on a
/// remote one. The child gets only the names an SSH session gives a command
/// (home, path, user, shell and locale), so no controller shell setting
/// reaches a rank on this computer that would not reach a remote rank.
pub fn local(command: String, label: &str) -> Exec {
    let mut exec = Exec::new(label, "sh", vec!["-c".into(), command]);
    exec.unset = std::env::vars_os()
        .map(|(name, _)| name.to_string_lossy().into_owned())
        .filter(|name| !(matches!(name.as_str(), "HOME" | "PATH" | "USER" | "LOGNAME" | "SHELL" | "LANG") || name.starts_with("LC_")))
        .collect();
    exec
}

/// Copy `sources` into the existing folder `folder` on the SSH host `alias`
/// with rsync. `options` follow `-a --chmod=go-rwx`. The receiver runs with
/// umask 077 inside `folder`. When `alias` names this computer, the same
/// rsync runs as a local copy inside `folder`, with the same umask.
pub fn copy_to(label: &str, alias: &str, folder: &str, options: Vec<String>, sources: Vec<String>) -> Exec {
    let receiver = format!("umask 077 && cd {} && exec rsync", remote_path(folder));
    let mut args: Vec<String> = vec!["-a".into(), "--chmod=go-rwx".into()];
    args.extend(options);
    if this_computer(alias) {
        let here = std::env::current_dir().map(|dir| dir.to_string_lossy().into_owned()).unwrap_or_default();
        args.extend(sources.into_iter().map(|source| if source.starts_with('/') { source } else { format!("{here}/{source}") }));
        args.push("./".into());
        let mut exec = local(format!("{receiver} \"$@\""), label);
        exec.args.push("sh".into());
        exec.args.extend(args);
        return exec;
    }
    let ssh = std::iter::once("ssh".into()).chain(ssh_options()).map(|s| quote(&s)).collect::<Vec<_>>().join(" ");
    args.extend([format!("--rsync-path={receiver}"), "-e".into(), ssh]);
    args.extend(sources);
    args.push(format!("{alias}:./"));
    Exec::new(label, "rsync", args)
}

mod host;
#[cfg(test)]
pub(crate) use host::{fake as fake_host, names as host_names, Host};
pub(crate) use host::this_computer;

/// Non-interactive SSH options: batch mode, a 10 s connect timeout, a
/// keepalive that drops a dead peer after about 20 s, and no local commands
/// or shared control connections.
pub fn ssh_options() -> Vec<String> {
    [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=2",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

static INTERRUPTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
extern "C" fn interrupt(_number: i32) {
    INTERRUPTED.store(true, std::sync::atomic::Ordering::Relaxed);
}
/// Whether SIGINT or SIGTERM arrived while [`Interruptions`] was installed.
pub(crate) fn interrupted() -> bool {
    INTERRUPTED.load(std::sync::atomic::Ordering::Relaxed)
}
/// Records SIGINT and SIGTERM instead of exiting, so owned work can unwind;
/// dropping it restores the previous handlers.
pub struct Interruptions(Vec<(i32, usize)>);
impl Interruptions {
    /// Clear the interrupted flag and install the handlers.
    pub fn install() -> Self {
        INTERRUPTED.store(false, std::sync::atomic::Ordering::Relaxed);
        Self(
            [2, 15]
                .into_iter()
                .map(|number| {
                    (number, unsafe {
                        signal(number, interrupt as *const () as usize)
                    })
                })
                .collect(),
        )
    }
}
impl Drop for Interruptions {
    fn drop(&mut self) {
        for (number, previous) in &self.0 {
            unsafe {
                signal(*number, *previous);
            }
        }
    }
}
#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn prctl(option: i32, ...) -> i32;
    fn getppid() -> i32;
}

unsafe extern "C" {
    fn setsid() -> i32;
    fn kill(pid: i32, signal: i32) -> i32;
    fn signal(number: i32, handler: usize) -> usize;
}
pub(crate) fn owned_command(exec: &Exec) -> Command {
    let mut command = exec.command();
    #[cfg(target_os = "linux")]
    let parent = std::process::id() as i32;
    unsafe {
        command.pre_exec(move || {
            #[cfg(target_os = "linux")]
            if prctl(1, 9usize, 0usize, 0usize, 0usize) != 0 || getppid() != parent {
                return Err(std::io::Error::last_os_error());
            }
            for number in [2, 15] {
                signal(number, 0);
            }
            if setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    command
}
pub(crate) fn stop_job(child: &mut std::process::Child) {
    // Called before wait/reap: this group ID belongs to our isolated child.
    unsafe {
        kill(-(child.id() as i32), 9);
    }
}

#[cfg(test)]
#[rustfmt::skip]
mod tests;
