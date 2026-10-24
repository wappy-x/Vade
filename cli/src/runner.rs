use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// A pre_exec fork can briefly inherit another test's flock descriptor even
// with CLOEXEC. Serialize process-spawning tests with the worker lock test.
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
    pub job: Option<PathBuf>,
    pub cancellable: bool,
    pub transient_probe: bool,
    pub probe_log: Option<PathBuf>,
}

impl Exec {
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
    pub fn preview(&self) -> Result<(), String> {
        if let Some(cwd) = &self.cwd {
            crate::output::line(format_args!("# folder: {}", crate::text::terminal_text(&cwd.to_string_lossy())))?;
        }
        let args = std::iter::once(&self.program)
            .chain(self.args.iter())
            .map(|s| quote(s))
            .collect::<Vec<_>>();
        crate::output::line(format_args!("$ {}", crate::text::terminal_text(&args.join(" "))))?;
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

#[derive(Clone, Debug, Default)]
pub struct Output {
    pub success: bool,
    pub stdout: String,
}

pub trait Runner {
    fn interrupted(&self) -> bool {
        interrupted()
    }
    fn recover_jobs(&mut self, root: &std::path::Path) -> Result<(), String> {
        crate::build_job::recover(root)
    }
    fn cleanup(&mut self, exec: &Exec) -> Result<Output, String> {
        self.run(exec)
    }
    fn run(&mut self, exec: &Exec) -> Result<Output, String>;
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
    fn interactive(&mut self, exec: &Exec) -> Result<bool, String>;
    fn port_free(&mut self, port: u16) -> Result<(), String>;
    /// Start a detached local process appending output to `log`; returns its PID.
    /// Scripted runners record the command and may answer a PID on stdout.
    fn spawn(&mut self, exec: &Exec, log: &std::path::Path) -> Result<u32, String> {
        let _ = log;
        Ok(self.run(exec)?.stdout.trim().parse().unwrap_or(0))
    }
    fn pause(&mut self, duration: Duration);
    fn confirm_restart(&mut self) -> Result<bool, String> {
        Ok(false)
    }
}

pub struct SystemRunner {
    pub dry: bool,
}

/// Report only a bounded operation label, never argv, environment or child output.
pub(crate) struct Progress<'a> {
    label: &'a str,
    next: Duration,
}
impl<'a> Progress<'a> {
    pub(crate) fn new(label: &'a str) -> Self {
        Self { label, next: Duration::from_secs(30) }
    }
    fn line(&mut self, elapsed: Duration) -> Option<String> {
        if elapsed < self.next { return None; }
        self.next = elapsed + Duration::from_secs(30);
        Some(format!("Vade: {} still running ({}s elapsed).", crate::text::argument(self.label), elapsed.as_secs()))
    }
    pub(crate) fn report(&mut self, elapsed: Duration) {
        if let Some(line) = self.line(elapsed) {
            crate::output::stderr_line(format_args!("{line}"));
        }
    }
}

impl Runner for SystemRunner {
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
            if metadata.len() > 16 * 1024 * 1024 {
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
        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .map_err(|_| "Cannot read restart choice. Run vade down and vade up when ready.")?;
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

pub fn ssh(alias: &str, command: String, label: &str) -> Exec {
    let mut args = ssh_options();
    args.extend([alias.into(), command]);
    Exec::new(label, "ssh", args)
}

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
pub(crate) fn interrupted() -> bool {
    INTERRUPTED.load(std::sync::atomic::Ordering::Relaxed)
}
pub struct Interruptions(Vec<(i32, usize)>);
impl Interruptions {
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
fn owned_command(exec: &Exec) -> Command {
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
fn stop_job(child: &mut std::process::Child) {
    // Called before wait/reap: this group ID belongs to our isolated child.
    unsafe {
        kill(-(child.id() as i32), 9);
    }
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;
    #[test]
    fn only_transient_probe_failures_suppress_the_retained_log_warning() {
        let _guard = process_test_lock();
        if std::env::var_os("VADE_TEST_TRANSIENT_PROBE").is_none() {
            let output = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "runner::tests::only_transient_probe_failures_suppress_the_retained_log_warning"])
                .env("VADE_TEST_TRANSIENT_PROBE", "1").output().unwrap();
            assert!(output.status.success());
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(!stderr.contains("expected probe"));
            assert!(stderr.contains("ordinary failure logs retained privately at"));
            return;
        }
        for transient in [true, false] {
            let mut exec = Exec::new(if transient { "expected probe" } else { "ordinary failure" }, "sh", vec!["-c".into(), "exit 7".into()]);
            exec.transient_probe = transient;
            assert!(!SystemRunner { dry: false }.run(&exec).unwrap().success);
        }
    }
    #[test]
    fn readiness_keeps_only_the_latest_private_diagnostic() {
        let _guard = process_test_lock();
        let root = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-probe-{}", crate::text::random_hex(8).unwrap()));
        crate::text::private_dir(&root).unwrap();
        let log = root.join("last.log");
        for reason in ["connection refused", "wrong model"] {
            let mut exec = Exec::new("readiness", "sh", vec!["-c".into(), format!("printf '%s' '{}'; printf '%s' '{}' >&2; exit 7", reason, reason)]);
            exec.transient_probe = true;
            exec.probe_log = Some(log.clone());
            assert!(!SystemRunner { dry: false }.run(&exec).unwrap().success);
            let diagnostic = std::fs::read_to_string(&log).unwrap();
            assert!(diagnostic.contains(reason));
            if reason == "wrong model" { assert!(!diagnostic.contains("connection refused")); }
        }
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&log).unwrap().permissions().mode() & 0o777, 0o600);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn progress_is_periodic_terminal_safe_and_does_not_touch_command_data() {
        let mut progress = Progress::new("build\n\x1b[31m");
        assert!(progress.line(Duration::from_secs(29)).is_none());
        let first = progress.line(Duration::from_secs(30)).unwrap();
        assert!(first.contains("30s elapsed") && !first.contains('\n') && !first.contains('\x1b'));
        assert!(progress.line(Duration::from_secs(59)).is_none());
        assert!(progress.line(Duration::from_secs(60)).unwrap().contains("60s elapsed"));
        assert!(progress.line(Duration::from_secs(61)).is_none());
        let label = "x".repeat(5000);
        let mut long = Progress::new(&label);
        assert!(long.line(Duration::from_secs(30)).unwrap().len() < 1100);
    }
    #[test]
    fn real_interrupt_is_prompt_and_recorded_cleanup_can_still_run() {
        let _process_guard = process_test_lock();
        if std::env::var_os("VADE_TEST_INTERRUPT_CONTROL").is_none() {
            let result = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "runner::tests::real_interrupt_is_prompt_and_recorded_cleanup_can_still_run"])
                .env("VADE_TEST_INTERRUPT_CONTROL", "1").output().unwrap();
            assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stdout)); return;
        }
        let _guard = Interruptions::install();
        unsafe extern "C" { fn raise(number: i32) -> i32; }
        unsafe { raise(2); }
        let mut runner = SystemRunner { dry: false };
        let exec = Exec::new("cleanup control", "sh", vec!["-c".into(), "printf cleaned".into()]);
        assert!(runner.run(&exec).unwrap_err().contains("Interrupted"));
        let before = Instant::now(); runner.pause(Duration::from_secs(180));
        assert!(before.elapsed() < Duration::from_millis(100));
        assert_eq!(runner.cleanup(&exec).unwrap().stdout, "cleaned");
    }
    #[test]
    fn timeout_stops_owned_descendants_and_preserves_foreign_child() {
        let _process_guard = process_test_lock();
        let root = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-job-test-{}", crate::text::random_hex(8).unwrap()));
        crate::text::private_dir(&root).unwrap();
        let mut foreign = Command::new("sleep").arg("30").spawn().unwrap();
        let marker = root.join("child");
        let mut exec = Exec::new("owned timeout control", "sh", vec!["-c".into(), "sleep 30 & printf '%s\\n' \"$!\" > \"$1\"; wait".into(), "vade-test".into(), marker.to_string_lossy().into_owned()]);
        exec.timeout = Duration::from_millis(250);
        let result = SystemRunner { dry: false }.run(&exec);
        let foreign_alive = foreign.try_wait().unwrap().is_none();
        let _ = foreign.kill(); let _ = foreign.wait();
        let pid = std::fs::read_to_string(&marker).unwrap().trim().to_string();
        let until = Instant::now() + Duration::from_secs(3); let mut dead = false;
        while Instant::now() < until {
            let status = Command::new("ps").args(["-o", "stat=", "-p", &pid]).output().unwrap();
            let state = String::from_utf8_lossy(&status.stdout);
            if state.trim().is_empty() || state.trim().starts_with('Z') { dead = true; break; }
            std::thread::sleep(Duration::from_millis(20));
        }
        std::fs::remove_dir_all(root).unwrap();
        assert!(result.unwrap_err().contains("timed out") && foreign_alive && dead);
    }
    #[test]
    fn stream_delivers_lines_and_bounds_inherited_pipe_lifetime() {
        let _process_guard = process_test_lock();
        let mut runner = SystemRunner { dry: false }; let mut lines = Vec::new();
        let exec = Exec::new("stream control", "sh", vec!["-c".into(), "printf 'one\\ntwo\\n'".into()]);
        let output = runner.stream(&exec, &mut |line| { lines.push(line.to_string()); Ok(()) }).unwrap();
        assert!(output.success); assert_eq!(lines, ["one", "two"]);
        let mut exec = Exec::new("stream timeout control", "sh", vec!["-c".into(), "sleep 30 & exit 0".into()]); exec.timeout = Duration::from_millis(250);
        assert!(runner.stream(&exec, &mut |_| Ok(())).unwrap_err().contains("timed out"));
    }
    #[test]
    fn previews_propagate_output_errors_before_running_commands() {
        use crate::output::{self, testing::Capture};
        for kind in [std::io::ErrorKind::BrokenPipe, std::io::ErrorKind::PermissionDenied] {
            output::with_writer(Capture { fail_on: Some(("$ ", kind)), ..Capture::default() }, || {
                let mut runner = SystemRunner { dry: true };
                let command = Exec::new("preview only", "missing-preview-program", vec![]);
                assert!(runner.run(&command).unwrap_err().contains("standard output"));
                assert_eq!(output::failure(), Some(kind));
            });
        }
    }
    #[test]
    fn stream_consumer_error_stops_and_reaps_the_owned_process_promptly() {
        let _guard = process_test_lock();
        let root = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-consumer-{}", crate::text::random_hex(8).unwrap()));
        crate::text::private_dir(&root).unwrap();
        let fifo = root.join("release");
        assert!(Command::new("mkfifo").arg(&fifo).status().unwrap().success());
        let mut release = std::fs::OpenOptions::new().read(true).write(true).open(&fifo).unwrap();
        // The owned child and its pipe-holding descendant cannot finish naturally
        // until the test releases them.
        // Readiness excludes process startup from the bounded cancellation wait.
        let mut exec = Exec::new("consumer cancellation", "sh", vec!["-c".into(),
            "read -r release < \"$1\" & child=$!; printf '%s\\n' \"$$\"; wait \"$child\"".into(), "vade-test".into(), fifo.to_string_lossy().into_owned()]);
        exec.timeout = Duration::from_secs(120);
        let (ready_send, ready) = std::sync::mpsc::channel();
        let (done_send, done) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut delivered = 0;
            let result = SystemRunner { dry: false }.stream(&exec, &mut |line| {
                delivered += 1; let _ = ready_send.send(line.to_string());
                Err("consumer output failed".into())
            });
            let _ = done_send.send((result, delivered));
        });
        let timeout = Duration::from_secs(60);
        let pid = ready.recv_timeout(timeout);
        let completed = done.recv_timeout(timeout);
        // Inspect before releasing the gate: cleanup must not make a premature
        // stream return look like successful cancellation and reaping.
        let status = pid.as_ref().ok().filter(|_| completed.is_ok()).map(|pid|
            Command::new("ps").args(["-o", "stat=", "-p", pid]).output());
        release.write_all(b"release\n").unwrap();
        if completed.is_ok() || done.recv_timeout(timeout).is_ok() { worker.join().unwrap(); }
        drop(release);
        std::fs::remove_dir_all(root).unwrap();
        assert!(pid.expect("owned child did not become ready").parse::<u32>().unwrap() > 0);
        let (result, delivered) = completed.expect("consumer cancellation did not finish while the owned child was gated");
        let error = result.unwrap_err();
        assert!(error.contains("consumer output failed"));
        assert_eq!(delivered, 1);
        let status = status.expect("missing owned child inspection").unwrap();
        assert!(status.stderr.is_empty(), "cannot inspect owned child: {}", String::from_utf8_lossy(&status.stderr));
        assert!(String::from_utf8_lossy(&status.stdout).trim().is_empty(), "owned child was not reaped");
    }
    #[test]
    fn process_test_lock_recovers_from_poisoning() {
        // Poison the shared lock in a disposable thread, then prove the next
        // acquisition still succeeds instead of failing with PoisonError.
        let panicking = std::thread::spawn(|| {
            let _guard = process_test_lock();
            panic!("deliberate test poisoning (expected)");
        });
        assert!(panicking.join().is_err());
        assert!(PROCESS_TEST_LOCK.is_poisoned());
        let _guard = process_test_lock();
    }

}

#[cfg(test)]
#[test]
fn parallel_jobs_both_start_before_either_can_finish() {
    let _guard = process_test_lock();
    let dir = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-parallel-{}", crate::text::random_hex(8).unwrap()));
    crate::text::private_dir(&dir).unwrap();
    let job = |own: &str, peer: &str| {
        let mut exec = Exec::new("parallel test", "/bin/sh", vec!["-c".into(),
            "touch \"$1\"; n=0; while ! test -e \"$2\"; do n=$((n+1)); test $n -lt 200 || exit 7; sleep 0.01; done; printf ready".into(),
            "parallel-test".into(), dir.join(own).to_string_lossy().into_owned(), dir.join(peer).to_string_lossy().into_owned()]);
        exec.timeout = Duration::from_secs(5); exec
    };
    let (a, b) = SystemRunner { dry: false }.parallel(&job("a", "b"), &job("b", "a")).unwrap();
    assert!(a.success && b.success); assert_eq!((&*a.stdout, &*b.stdout), ("ready", "ready"));
    std::fs::remove_dir_all(dir).unwrap();
}
