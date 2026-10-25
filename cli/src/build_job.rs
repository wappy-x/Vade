//! Durable local build jobs. A supervisor retains an unreaped session anchor,
//! so a compiler leader's exit never releases the process-group identity.
use crate::runner::{Exec, Output};
use crate::text::{self, Fields, required};
use crate::unix_file::{AnchoredPath, OpenMode};
use crate::unix_flags::{F_GETFL, F_SETFL, LOCK_EX, LOCK_NB, O_NONBLOCK};
use std::fs::{self, File};
use std::io::Read;
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

unsafe extern "C" {
    fn setsid() -> i32;
    fn getpgid(pid: i32) -> i32;
    fn getsid(pid: i32) -> i32;
    fn getppid() -> i32;
    fn kill(pid: i32, number: i32) -> i32;
    fn flock(fd: i32, operation: i32) -> i32;
    fn fcntl(fd: i32, command: i32, ...) -> i32;
}

fn save(job: &Path, name: &str, fields: &Fields) -> Result<(), String> {
    text::atomic_write(&job.join(name), &text::encode(fields))
}
fn read(job: &Path, name: &str) -> Result<Option<Fields>, String> {
    let path = job.join(name);
    text::reject_symlinks(&path)?;
    // Recovery happens in the launcher before any command timeout. Anchored
    // opens use O_NONBLOCK; validate the opened descriptor before reading it.
    let file = match AnchoredPath::new(&path).and_then(|path| path.open(OpenMode::Read)) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("Cannot read build ownership evidence; keep the job directory.".into()),
    };
    if !file.metadata().is_ok_and(|metadata| metadata.is_file()) {
        return Err("Build ownership evidence must be a regular file; keep the job directory.".into());
    }
    let mut value = String::new();
    file.take(1024 * 1024 + 1).read_to_string(&mut value)
        .map_err(|_| "Cannot read build ownership evidence; keep the job directory.")?;
    if value.len() > 1024 * 1024 {
        return Err("Build ownership evidence exceeds 1 MiB; keep the job directory.".into());
    }
    text::parse(&value).map(Some)
}

/// Record `exec` as a durable job under `root/build-jobs` and point `exec.job`
/// at it, so a later command can find and settle an interrupted build.
pub fn prepare(root: &Path, exec: &mut Exec) -> Result<(), String> {
    let job = root.join("build-jobs").join(text::random_hex(16)?);
    text::private_dir(&job)?;
    if !exec.stdin.is_empty() {
        return Err("Build commands cannot consume private stdin.".into());
    }
    let mut intent = Fields::from([
        ("token".into(), text::random_hex(32)?),
        ("program".into(), exec.program.clone()),
        ("label".into(), exec.label.clone()),
        ("timeout_ms".into(), exec.timeout.as_millis().to_string()),
    ]);
    if let Some(cwd) = &exec.cwd {
        intent.insert("cwd".into(), cwd.to_string_lossy().into_owned());
    }
    for (i, arg) in exec.args.iter().enumerate() {
        intent.insert(format!("arg.{i}"), arg.clone());
    }
    for (key, value) in &exec.env {
        intent.insert(format!("env.{key}"), value.clone());
    }
    for (i, key) in exec.unset.iter().enumerate() {
        intent.insert(format!("unset.{i}"), key.clone());
    }
    // Both execution instructions and ownership intent are synced before spawn.
    save(&job, "intent.conf", &intent)?;
    exec.job = Some(job);
    Ok(())
}

fn helper_command(mode: &str, job: &Path) -> Result<Command, String> {
    let mut command =
        Command::new(std::env::current_exe().map_err(|_| "Cannot find build helper.")?);
    #[cfg(not(test))]
    command.arg(mode).arg(job);
    #[cfg(test)]
    command
        .args(["--exact", "build_job::tests::helper_process", "--nocapture"])
        .env("VADE_TEST_BUILD_MODE", mode)
        .env("VADE_TEST_BUILD_JOB", job);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    unsafe {
        command.pre_exec(|| {
            if setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    Ok(command)
}

/// Run a prepared job under its own supervisor process and wait for it,
/// reporting progress.
pub fn run(job: &Path) -> Result<Output, String> {
    let began = Instant::now();
    let mut progress = crate::runner::Progress::new("local source build");
    let mut child = helper_command("__build-supervisor", job)?
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|_| "Cannot start recorded build supervisor.")?;
    // The sole writer stays in this CLI; its loss is EOF in the supervisor.
    let lifetime = child
        .stdin
        .take()
        .ok_or("Build lifetime pipe is missing.")?;
    loop {
        if crate::runner::interrupted() {
            cancel(job)?;
        }
        if child
            .try_wait()
            .map_err(|_| "Cannot inspect build supervisor; run vade down.")?
            .is_some()
        {
            break;
        }
        progress.report(began.elapsed());
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(lifetime);
    let stopped = read(job, "stopped.conf")?.ok_or_else(|| {
        format!(
            "Build cleanup is unresolved. Keep {}; run vade down.",
            job.display()
        )
    })?;
    if let Some(error) = stopped.get("error") {
        return Err(format!("{error} Build logs retained at {}", job.display()));
    }
    let path = job.join("stdout");
    if fs::metadata(&path)
        .map_err(|_| "Cannot read build output.")?
        .len()
        > 16 * 1024 * 1024
    {
        return Err("Build output is too large; inspect the private job logs.".into());
    }
    Ok(Output {
        success: stopped.get("success").map(String::as_str) == Some("true"),
        stdout: fs::read_to_string(path).map_err(|_| "Build output is not readable text.")?,
    })
}

fn cancel(job: &Path) -> Result<(), String> {
    save(
        job,
        "cancel.conf",
        &Fields::from([("cancel".into(), "true".into())]),
    )
}
fn lock(job: &Path) -> Result<Option<File>, String> {
    let path = job.join("supervisor.lock");
    text::reject_symlinks(&path)?;
    let file = AnchoredPath::new(&path).and_then(|path| path.open(OpenMode::Create))
        .map_err(|_| "Cannot open build supervisor lock.")?;
    if !file.metadata().is_ok_and(|metadata| metadata.is_file()) {
        return Err("Build supervisor lock must be a regular file; keep the job directory.".into());
    }
    if unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) } == 0 {
        Ok(Some(file))
    } else if std::io::Error::last_os_error().kind() == std::io::ErrorKind::WouldBlock {
        Ok(None)
    } else {
        Err("Cannot lock build supervisor evidence.".into())
    }
}

/// Cancel and wait for any job under `root` that an earlier command did not
/// finish.
pub fn recover(root: &Path) -> Result<(), String> {
    let jobs = root.join("build-jobs");
    text::reject_symlinks(&jobs)?;
    if !jobs.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(jobs).map_err(|_| "Cannot list recorded build jobs.")? {
        let job = entry
            .map_err(|_| "Cannot read build job directory.")?
            .path();
        text::reject_symlinks(&job)?;
        let intent =
            read(&job, "intent.conf")?.ok_or("Build job intent is missing; keep its evidence.")?;
        if !text::hex(required(&intent, "token")?, 64) {
            return Err("Invalid build ownership token.".into());
        }
        if read(&job, "stopped.conf")?.is_some() {
            continue;
        }
        cancel(&job)?;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if read(&job, "stopped.conf")?.is_some() {
                break;
            }
            if let Some(_lock) = lock(&job)? {
                if read(&job, "stopped.conf")?.is_some() {
                    break;
                }
                if let Some(supervisor) = read(&job, "supervisor.conf")? {
                    if !lost_and_stopped(&job, &supervisor)? {
                        return Err(format!(
                            "Build supervisor was lost; cleanup is uncertain. Nothing signalled. Keep {}.",
                            job.display()
                        ));
                    }
                    // A tombstone, like below: the evidence stays and a late helper refuses.
                    save(
                        &job,
                        "stopped.conf",
                        &Fields::from([("error".into(), "Build supervisor was lost after its processes stopped.".into())]),
                    )?;
                    crate::output::stderr_line(format_args!(
                        "Vade: cleared a stale build record; its supervisor and build processes are gone ({}).",
                        text::terminal_text(&job.to_string_lossy())
                    ));
                    break;
                }
                // No registered supervisor can have spawned an anchor. Retain a
                // cancellation tombstone so a late helper cannot start this job.
                save(
                    &job,
                    "stopped.conf",
                    &Fields::from([("error".into(), "Build cancelled before execution.".into())]),
                )?;
                break;
            }
            if Instant::now() >= deadline {
                return Err(
                    "Build cleanup is still pending; keep the record and retry vade down.".into(),
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    Ok(())
}

/// Called with the supervisor lock held, which the supervisor keeps until it
/// exits. The record is stale only when no recorded process can still run: the
/// supervisor identity is not alive, and the anchor's process group (the anchor
/// leads it) has no live member or belongs to an earlier boot. Nothing is
/// signalled here.
fn lost_and_stopped(job: &Path, supervisor: &Fields) -> Result<bool, String> {
    let alive = supervisor.get("pid").and_then(|pid| pid.parse::<u32>().ok()).is_none_or(|pid| {
        snapshot(pid).is_ok_and(|now| ["boot", "birth"].iter().all(|k| supervisor.get(*k) == now.get(*k)))
    });
    if alive {
        return Ok(false);
    }
    // Without anchor.conf no compiler ran: the anchor records itself before it
    // checks cancel.conf, which recovery wrote before this read.
    let Some(anchor) = read(job, "anchor.conf")? else { return Ok(true) };
    if anchor.get("boot") != snapshot(std::process::id())?.get("boot") {
        return Ok(true);
    }
    Ok(!group_active(required(&anchor, "pid")?.parse().map_err(|_| "Invalid build anchor identity.")?)?)
}

/// The private `__build-supervisor` and `__build-anchor` processes.
pub fn helper(mode: &str, job: &Path) -> Result<(), String> {
    text::private_dir(job)?;
    let intent = read(job, "intent.conf")?.ok_or("No durable build intent; nothing executed.")?;
    if !text::hex(required(&intent, "token")?, 64) {
        return Err("Invalid build token.".into());
    }
    if mode == "__build-anchor" {
        return anchor(job, &intent);
    }
    if mode != "__build-supervisor" {
        return Err("Invalid build helper mode.".into());
    }
    let _lock = lock(job)?.ok_or("Build already has a supervisor.")?;
    if read(job, "cancel.conf")?.is_some() || read(job, "stopped.conf")?.is_some() {
        save(
            job,
            "stopped.conf",
            &Fields::from([("error".into(), "Build cancelled before execution.".into())]),
        )?;
        return Ok(());
    }
    let mut supervisor = snapshot(std::process::id())?;
    supervisor.insert("token".into(), intent["token"].clone());
    save(job, "supervisor.conf", &supervisor)?;
    let timeout = required(&intent, "timeout_ms")?
        .parse::<u64>()
        .map_err(|_| "Invalid build timeout.")?;
    let deadline = Instant::now() + Duration::from_millis(timeout);
    let flags = unsafe { fcntl(0, F_GETFL) };
    if flags < 0 || unsafe { fcntl(0, F_SETFL, flags | O_NONBLOCK) } < 0 {
        return Err("Cannot watch CLI lifetime; build evidence retained.".into());
    }
    let mut anchor = helper_command("__build-anchor", job)?
        .spawn()
        .map_err(|_| "Cannot start build session anchor; ownership evidence retained.")?;
    let outcome = (|| -> Result<Fields, String> { loop {
        let mut byte = [0];
        let lost = match std::io::stdin().read(&mut byte) {
            Ok(0) => true,
            Ok(_) => false,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => false,
            Err(_) => true,
        };
        if lost || read(job, "cancel.conf")?.is_some() || Instant::now() >= deadline {
            break Ok(Fields::from([("error".into(), "Build timed out, was interrupted, or lost its CLI; its recorded group stopped.".into())]));
        }
        if let Some(result) = read(job, "result.conf")? { break Ok(result); }
        std::thread::sleep(Duration::from_millis(20));
    } })().unwrap_or_else(|error| Fields::from([("error".into(), error)]));
    // Do not try_wait/reap the anchor before the final group signal. Even if it
    // exited unexpectedly, its reserved PID prevents process-group ID reuse.
    let until = Instant::now() + Duration::from_secs(5);
    let identity = loop {
        if let Some(identity) = read(job, "anchor.conf")? {
            break identity;
        }
        if Instant::now() >= until {
            return Err("Unrecorded build anchor; nothing signalled, evidence retained.".into());
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    stop_group(job, &intent, &identity, &mut anchor)?;
    save(job, "stopped.conf", &outcome)
}

fn anchor(job: &Path, intent: &Fields) -> Result<(), String> {
    let supervisor =
        read(job, "supervisor.conf")?.ok_or("Build supervisor identity is missing.")?;
    let parent = unsafe { getppid() } as u32;
    let now = snapshot(parent)?;
    if supervisor.get("token") != intent.get("token")
        || supervisor.get("pid") != Some(&parent.to_string())
        || now
            .iter()
            .any(|(key, value)| supervisor.get(key) != Some(value))
        || read(job, "anchor.conf")?.is_some()
    {
        return Err("Build supervisor identity changed; no compiler executed.".into());
    }
    let mut identity = snapshot(std::process::id())?;
    identity.insert("token".into(), intent["token"].clone());
    save(job, "anchor.conf", &identity)?;
    let result = (|| {
        if read(job, "cancel.conf")?.is_some() {
            return Err("Build cancelled before compiler spawn.".into());
        }
        let args = (0..)
            .map_while(|i| intent.get(&format!("arg.{i}")).cloned())
            .collect();
        let mut exec = Exec::new(&intent["label"], &intent["program"], args);
        exec.cwd = intent.get("cwd").map(PathBuf::from);
        for (key, value) in intent {
            if let Some(key) = key.strip_prefix("env.") {
                exec.env.insert(key.into(), value.clone());
            }
            if key.starts_with("unset.") {
                exec.unset.push(value.clone());
            }
        }
        let mut child = exec
            .command()
            .stdin(Stdio::null())
            .stdout(log(job, "stdout")?)
            .stderr(log(job, "stderr")?)
            .spawn()
            .map_err(|_| "Cannot spawn compiler in recorded group.".to_string())?;
        let status = child
            .wait()
            .map_err(|_| "Cannot inspect compiler exit.".to_string())?;
        Ok(Fields::from([(
            "success".into(),
            status.success().to_string(),
        )]))
    })();
    let result = result.unwrap_or_else(|error| Fields::from([("error".into(), error)]));
    save(job, "result.conf", &result)?;
    // Compiler completion (including failure) leaves the group anchor alive for
    // its supervisor's signal. A lost supervisor cannot signal, so the anchor
    // exits instead of holding the record open forever.
    while std::os::unix::process::parent_id() == parent {
        std::thread::sleep(Duration::from_millis(250));
    }
    Ok(())
}
fn log(job: &Path, name: &str) -> Result<File, String> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(job.join(name))
        .map_err(|_| "Cannot create private build log.".into())
}

fn check_anchor(
    intent: &Fields,
    saved: &Fields,
    now: &Fields,
    child_pid: u32,
) -> Result<(), String> {
    if saved.get("token") != intent.get("token")
        || saved.get("pid") != Some(&child_pid.to_string())
        || saved.get("pgid") != saved.get("pid")
        || saved.get("sid") != saved.get("pid")
        || ["pid", "pgid", "sid", "boot", "birth"]
            .iter()
            .any(|k| !saved.contains_key(*k) || saved.get(*k) != now.get(*k))
    {
        return Err(
            "Build anchor identity changed. Nothing signalled; retain ownership evidence.".into(),
        );
    }
    Ok(())
}
fn stop_group(
    job: &Path,
    intent: &Fields,
    saved: &Fields,
    anchor: &mut Child,
) -> Result<(), String> {
    // Only the supervisor holding this unreaped Child may signal the group.
    // Recovery requests cancellation instead of sending numeric PID signals.
    let now = snapshot(anchor.id())?;
    check_anchor(intent, saved, &now, anchor.id())?;
    if unsafe { kill(-(anchor.id() as i32), 9) } != 0 {
        return Err("Cannot stop recorded build group; retain its evidence.".into());
    }
    let group = anchor.id();
    anchor
        .wait()
        .map_err(|_| "Cannot reap build anchor; retain its evidence.")?;
    let deadline = Instant::now() + Duration::from_secs(5);
    while group_active(group)? {
        if Instant::now() >= deadline {
            return Err(format!(
                "Build group is not proved stopped; retain {}.",
                job.display()
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}
fn group_active(group: u32) -> Result<bool, String> {
    // Enumeration is only a stop-proof, never a selection source for signals.
    let output = Command::new("/bin/ps")
        .args(["-axo", "pgid=,stat="])
        .output()
        .map_err(|_| "Cannot prove build group stopped.")?;
    if !output.status.success() {
        return Err("Cannot prove build group stopped.".into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).lines().any(|line| {
        let mut fields = line.split_whitespace();
        fields.next().and_then(|s| s.parse::<u32>().ok()) == Some(group)
            && fields.next().is_some_and(|state| !state.starts_with('Z'))
    }))
}

fn snapshot(pid: u32) -> Result<Fields, String> {
    let (boot, birth) = birth(pid)?;
    let pgid = unsafe { getpgid(pid as i32) };
    let sid = unsafe { getsid(pid as i32) };
    if pgid < 0 || sid < 0 {
        return Err("Cannot re-check build session identity; nothing signalled.".into());
    }
    Ok(Fields::from([
        ("pid".into(), pid.to_string()),
        ("pgid".into(), pgid.to_string()),
        ("sid".into(), sid.to_string()),
        ("boot".into(), boot),
        ("birth".into(), birth),
    ]))
}
#[cfg(target_os = "linux")]
fn birth(pid: u32) -> Result<(String, String), String> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))
        .map_err(|_| "Cannot read build birth identity.")?;
    let ticks = stat
        .rsplit_once(')')
        .and_then(|(_, rest)| rest.split_whitespace().nth(19))
        .ok_or("Invalid build birth identity.")?;
    let boot = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map_err(|_| "Cannot read build boot identity.")?;
    Ok((boot.trim().into(), ticks.into()))
}
#[cfg(target_os = "macos")]
fn birth(pid: u32) -> Result<(String, String), String> {
    #[repr(C)]
    #[derive(Default)]
    struct BsdInfo {
        header: [u32; 12],
        comm: [u8; 16],
        name: [u8; 32],
        tail: [u32; 6],
        sec: u64,
        usec: u64,
    }
    unsafe extern "C" {
        fn proc_pidinfo(pid: i32, flavor: i32, arg: u64, buffer: *mut BsdInfo, size: i32) -> i32;
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            old: *mut [i64; 2],
            size: *mut usize,
            new: *const u8,
            len: usize,
        ) -> i32;
    }
    let mut info = BsdInfo::default();
    let size = std::mem::size_of::<BsdInfo>() as i32;
    let mut boot = [0i64; 2];
    let mut boot_size = std::mem::size_of_val(&boot);
    if unsafe { proc_pidinfo(pid as i32, 3, 0, &mut info, size) } != size
        || unsafe {
            sysctlbyname(
                c"kern.boottime".as_ptr(),
                &mut boot,
                &mut boot_size,
                std::ptr::null(),
                0,
            )
        } != 0
    {
        return Err("Cannot read build boot/birth identity; nothing signalled.".into());
    }
    Ok((
        format!("{}:{}", boot[0], boot[1]),
        format!("{}:{}", info.sec, info.usec),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn helper_process() {
        if let (Ok(mode), Some(job)) = (
            std::env::var("VADE_TEST_BUILD_MODE"),
            std::env::var_os("VADE_TEST_BUILD_JOB"),
        ) {
            if mode == "__build-driver" {
                let _ = run(Path::new(&job));
            } else if mode == "__build-recover-fifo" {
                assert!(recover(Path::new(&job)).unwrap_err().contains("regular file"));
            } else {
                helper(&mode, Path::new(&job)).unwrap();
            }
        }
    }
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let root = fs::canonicalize(std::env::temp_dir())
                .unwrap()
                .join(format!("vade-build-test-{}", text::random_hex(8).unwrap()));
            text::private_dir(&root).unwrap();
            Self(root)
        }
        fn job(&self, exit: bool, timeout: Duration) -> PathBuf {
            // A shell is the mock compiler, and sleep is its mock descendant.
            let script = format!(
                "sleep 30 & printf '%s\\n' \"$!\" > \"$1\"; {}",
                if exit { "exit 7" } else { "wait" }
            );
            let mut exec = Exec::new(
                "mock build",
                "sh",
                vec![
                    "-c".into(),
                    script,
                    "control".into(),
                    self.0.join("descendant").to_string_lossy().into_owned(),
                ],
            );
            exec.timeout = timeout;
            prepare(&self.0, &mut exec).unwrap();
            let job = exec.job.unwrap();
            assert!(read(&job, "intent.conf").unwrap().is_some());
            assert!(read(&job, "anchor.conf").unwrap().is_none());
            assert!(!self.0.join("descendant").exists());
            job
        }
        fn assert_stopped(&self, job: &Path) {
            assert!(read(job, "stopped.conf").unwrap().is_some());
            let anchor = read(job, "anchor.conf").unwrap().unwrap();
            assert_eq!(anchor["pid"], anchor["pgid"]);
            assert_eq!(anchor["pid"], anchor["sid"]);
            assert!(!group_active(anchor["pgid"].parse().unwrap()).unwrap());
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    struct Foreign(Child);
    impl Drop for Foreign {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    #[test]
    fn mock_compiler_failure_timeout_and_cli_loss_stop_whole_recorded_group() {
        let _process_guard = crate::runner::process_test_lock();
        let mut foreign = Foreign(Command::new("sleep").arg("30").spawn().unwrap());
        let failure = Fixture::new();
        let job = failure.job(true, Duration::from_secs(5));
        assert!(!run(&job).unwrap().success);
        failure.assert_stopped(&job);
        let timeout = Fixture::new();
        let job = timeout.job(false, Duration::from_millis(300));
        assert!(run(&job).unwrap_err().contains("timed out"));
        timeout.assert_stopped(&job);
        let lost = Fixture::new();
        let job = lost.job(false, Duration::from_secs(30));
        let mut cli = helper_command("__build-driver", &job)
            .unwrap()
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !lost.0.join("descendant").exists() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(20));
        }
        // SIGKILL the disposable CLI driver while mock compiler descendants run.
        // The supervisor must clean on EOF without a recovery/cancel request.
        cli.kill().unwrap();
        cli.wait().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while read(&job, "stopped.conf").unwrap().is_none() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(20));
        }
        recover(&lost.0).unwrap();
        lost.assert_stopped(&job);
        assert!(foreign.0.try_wait().unwrap().is_none());
    }
    #[test]
    fn recovery_cancels_active_and_unspawned_jobs_without_unrecorded_signals() {
        let _process_guard = crate::runner::process_test_lock();
        let f = Fixture::new();
        let job = f.job(false, Duration::from_secs(30));
        let mut supervisor = helper_command("__build-supervisor", &job)
            .unwrap()
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !f.0.join("descendant").exists() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(20));
        }
        recover(&f.0).unwrap();
        supervisor.wait().unwrap();
        f.assert_stopped(&job);
        let pending = Fixture::new();
        let job = pending.job(false, Duration::from_secs(30));
        recover(&pending.0).unwrap();
        assert!(!pending.0.join("descendant").exists());
        // A late helper respects the durable cancellation tombstone.
        assert!(
            run(&job)
                .unwrap_err()
                .contains("cancelled before execution")
        );
        assert!(read(&job, "anchor.conf").unwrap().is_none());
    }
    #[test]
    fn recovery_clears_a_lost_supervisor_record_only_when_its_processes_are_gone() {
        let _process_guard = crate::runner::process_test_lock();
        // A supervisor that failed or was killed after registering, with no anchor.
        let f = Fixture::new();
        let job = f.job(false, Duration::from_secs(30));
        let intent = read(&job, "intent.conf").unwrap().unwrap();
        let mut gone = Command::new("sleep").arg("30").spawn().unwrap();
        let mut record = snapshot(gone.id()).unwrap();
        record.insert("token".into(), intent["token"].clone());
        save(&job, "supervisor.conf", &record).unwrap();
        gone.kill().unwrap();
        gone.wait().unwrap();
        recover(&f.0).unwrap();
        assert!(read(&job, "stopped.conf").unwrap().unwrap()["error"].contains("supervisor was lost"));
        recover(&f.0).unwrap();
        assert!(run(&job).unwrap_err().contains("cancelled before execution"));
        assert!(!f.0.join("descendant").exists());
        // A free lock with a live recorded identity is not proof; keep the record.
        let live = Fixture::new();
        let job = live.job(false, Duration::from_secs(30));
        let mut record = snapshot(std::process::id()).unwrap();
        record.insert("token".into(), intent["token"].clone());
        save(&job, "supervisor.conf", &record).unwrap();
        assert!(recover(&live.0).unwrap_err().contains("supervisor was lost"));
        assert!(read(&job, "stopped.conf").unwrap().is_none());
        // A recorded anchor from an earlier boot cannot still run.
        save(&job, "anchor.conf", &Fields::from([
            ("pid".into(), std::process::id().to_string()),
            ("boot".into(), "an-earlier-boot".into()),
            ("birth".into(), "0".into()),
        ])).unwrap();
        record.insert("boot".into(), "an-earlier-boot".into());
        save(&job, "supervisor.conf", &record).unwrap();
        recover(&live.0).unwrap();
        assert!(read(&job, "stopped.conf").unwrap().is_some());
    }
    #[test]
    fn recovery_waits_for_a_killed_supervisors_build_then_clears_it() {
        let _process_guard = crate::runner::process_test_lock();
        let f = Fixture::new();
        // The mock compiler waits for its `sleep 30` descendant.
        let job = f.job(false, Duration::from_secs(30));
        let mut supervisor = Foreign(helper_command("__build-supervisor", &job)
            .unwrap()
            .stdin(Stdio::piped())
            .spawn()
            .unwrap());
        let deadline = Instant::now() + Duration::from_secs(5);
        while !f.0.join("descendant").exists() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(20));
        }
        supervisor.0.kill().unwrap();
        supervisor.0.wait().unwrap();
        // The compiler still runs under its anchor: keep the record.
        let anchor: u32 = read(&job, "anchor.conf").unwrap().unwrap()["pid"].parse().unwrap();
        assert!(recover(&f.0).unwrap_err().contains("supervisor was lost"));
        assert!(read(&job, "stopped.conf").unwrap().is_none());
        // Once the build ends, the orphaned anchor exits and recovery clears the record.
        let descendant = fs::read_to_string(f.0.join("descendant")).unwrap();
        assert!(Command::new("kill").args(["-9", descendant.trim()]).status().unwrap().success());
        let deadline = Instant::now() + Duration::from_secs(5);
        while group_active(anchor).unwrap() {
            assert!(Instant::now() < deadline, "orphaned anchor kept running");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(read(&job, "result.conf").unwrap().is_some());
        recover(&f.0).unwrap();
        assert!(read(&job, "stopped.conf").unwrap().unwrap()["error"].contains("supervisor was lost"));
    }
    #[test]
    fn recovery_rejects_fifo_records_and_lock_within_bounded_subprocesses() {
        let _process_guard = crate::runner::process_test_lock();
        for name in ["intent.conf", "stopped.conf", "supervisor.conf", "supervisor.lock"] {
            let f = Fixture::new();
            let job = f.job(false, Duration::from_secs(30));
            let path = job.join(name);
            if path.exists() { fs::remove_file(&path).unwrap(); }
            assert!(Command::new("mkfifo").arg(&path).status().unwrap().success());
            let mut child = Foreign(helper_command("__build-recover-fifo", &f.0).unwrap().spawn().unwrap());
            let deadline = Instant::now() + Duration::from_secs(15);
            loop {
                if let Some(status) = child.0.try_wait().unwrap() {
                    assert!(status.success(), "recovery should reject {name}");
                    break;
                }
                assert!(Instant::now() < deadline, "recovery hung on {name}");
                std::thread::sleep(Duration::from_millis(20));
            }
            assert!(!f.0.join("descendant").exists());
        }
    }

    #[test]
    fn ownership_record_reads_are_bounded_and_keep_invalid_evidence() {
        let f = Fixture::new();
        let path = f.0.join("intent.conf");
        fs::write(&path, vec![b'x'; 1024 * 1024 + 1]).unwrap();
        assert!(read(&f.0, "intent.conf").unwrap_err().contains("exceeds 1 MiB"));
        assert_eq!(fs::metadata(&path).unwrap().len(), 1024 * 1024 + 1);
        fs::write(&path, [0xff]).unwrap();
        assert!(read(&f.0, "intent.conf").unwrap_err().contains("Cannot read"));
        assert_eq!(fs::read(path).unwrap(), [0xff]);
    }

    #[test]
    fn mock_anchor_identity_recheck_refuses_reuse_reboot_and_incomplete_records() {
        let intent = Fields::from([("token".into(), "a".repeat(64))]);
        let mut saved = Fields::from([
            ("pid".into(), "42".into()),
            ("pgid".into(), "42".into()),
            ("sid".into(), "42".into()),
            ("birth".into(), "100".into()),
            ("boot".into(), "boot-a".into()),
            ("token".into(), intent["token"].clone()),
        ]);
        let now = saved.clone();
        check_anchor(&intent, &saved, &now, 42).unwrap();
        for key in ["birth", "boot", "pid", "pgid", "sid"] {
            let mut changed = saved.clone();
            changed.insert(key.into(), "changed".into());
            assert!(check_anchor(&intent, &saved, &changed, 42).is_err());
        }
        let mut wrong_token = saved.clone();
        wrong_token.insert("token".into(), "wrong-token".into());
        assert!(check_anchor(&intent, &wrong_token, &now, 42).is_err());
        for key in ["birth", "boot", "pid", "pgid", "sid", "token"] {
            let value = saved.remove(key).unwrap();
            assert!(check_anchor(&intent, &saved, &now, 42).is_err());
            saved.insert(key.into(), value);
        }
    }
}
