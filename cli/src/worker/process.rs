//! Process identity, safe signalling and failed-launch rollback.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Identity {
    pub(super) pid: u32,
    pub(super) start: String,
    pub(super) boot: String,
    pub(super) exe: PathBuf,
    pub(super) command: Vec<u8>,
}

impl Identity {
    pub(super) fn write_to(&self, record: &mut Fields) {
        record.insert("pid".into(), self.pid.to_string());
        record.insert("start_ticks".into(), self.start.clone());
        record.insert("boot_id".into(), self.boot.clone());
        record.insert(
            "exe_hex".into(),
            to_hex(self.exe.as_os_str().as_encoded_bytes()),
        );
        record.insert("cmdline_hex".into(), to_hex(&self.command));
    }

    pub(super) fn from_record(record: &Fields) -> Result<Option<Self>, String> {
        let Some(pid) = record.get("pid") else {
            return Ok(None);
        };
        let pid = pid
            .parse::<u32>()
            .ok()
            .filter(|pid| *pid > 1 && *pid <= i32::MAX as u32)
            .ok_or_else(|| {
                "The saved process number is invalid. Leave these files in place; see Recovery in cli/README.md.".to_string()
            })?;
        let start = required(record, "start_ticks")?.to_owned();
        if start.is_empty() || !start.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err("The saved process start time is invalid. Leave these files in place; see Recovery in cli/README.md.".into());
        }
        let exe = unhex(required(record, "exe_hex")?)?;
        use std::os::unix::ffi::OsStringExt;
        Ok(Some(Self {
            pid,
            start,
            boot: required(record, "boot_id")?.to_owned(),
            exe: PathBuf::from(std::ffi::OsString::from_vec(exe)),
            command: unhex(required(record, "cmdline_hex")?)?,
        }))
    }
}

pub(super) fn snapshot(pid: u32) -> Result<Option<Identity>, String> {
    let root = PathBuf::from("/proc").join(pid.to_string());
    let read = |path: &Path| -> Result<Option<Vec<u8>>, String> {
        match fs::read(path) {
            Ok(value) => Ok(Some(value)),
            Err(error) if matches!(error.kind(), std::io::ErrorKind::NotFound) => Ok(None),
            Err(_) => {
                Err("Cannot verify the saved engine process. Leave these files in place; see Recovery in cli/README.md.".into())
            }
        }
    };
    let Some(stat) = read(&root.join("stat"))? else {
        return Ok(None);
    };
    let stat = String::from_utf8(stat).map_err(|_| {
        "Cannot read engine identity. Leave these files in place; see Recovery in cli/README.md.".to_string()
    })?;
    let Some(start) = stat_start(&stat)? else {
        return Ok(None);
    };
    let boot = boot_id()?;
    let exe = match fs::read_link(root.join("exe")) {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => {
            return Err("Cannot verify the engine executable. Leave these files in place; see Recovery in cli/README.md.".into());
        }
    };
    let Some(command) = read(&root.join("cmdline"))? else {
        return Ok(None);
    };
    // Read stat twice so a PID replacement during the snapshot never produces
    // a mixture of the old and new processes' identity fields.
    let Some(after) = read(&root.join("stat"))? else {
        return Ok(None);
    };
    let after =
        String::from_utf8(after).map_err(|_| "Cannot parse engine identity.".to_string())?;
    if stat_start(&after)?.as_ref() != Some(&start) {
        return Err(
            "The engine process changed while checking it. Retry vade status; if it still fails, see Recovery in cli/README.md."
                .into(),
        );
    }
    Ok(Some(Identity {
        pid,
        start,
        boot,
        exe,
        command,
    }))
}

pub(super) fn stat_start(stat: &str) -> Result<Option<String>, String> {
    let tail = stat
        .rsplit_once(')')
        .map(|(_, tail)| tail)
        .ok_or_else(|| "Cannot parse Linux process identity.".to_string())?;
    let fields: Vec<_> = tail.split_whitespace().collect();
    if fields
        .first()
        .is_some_and(|state| matches!(*state, "Z" | "X"))
    {
        return Ok(None);
    }
    fields
        .get(19)
        .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .map(|value| Some((*value).to_owned()))
        .ok_or_else(|| "Cannot parse Linux process birth time.".into())
}

fn boot_id() -> Result<String, String> {
    fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map(|value| value.trim().to_owned())
        .map_err(|_| "Cannot read this machine's boot identity. Run vade doctor.".into())
}

pub(super) fn verify(identity: &Identity) -> Result<bool, String> {
    // A reboot makes stale records uncertain even if this PID is absent now.
    let boot = boot_id()?;
    if boot != identity.boot {
        return verify_snapshot(identity, &boot, None);
    }
    let now = snapshot(identity.pid)?;
    let result = verify_snapshot(identity, &boot, now.as_ref());
    if result.is_err() {
        std::thread::sleep(Duration::from_millis(20));
        if snapshot(identity.pid)?.is_none() {
            return Ok(false);
        }
    }
    result
}

pub(super) fn verify_snapshot(
    identity: &Identity,
    boot: &str,
    now: Option<&Identity>,
) -> Result<bool, String> {
    if boot != identity.boot {
        return Err(
            "This machine rebooted after launch. Leave these files in place; see Recovery in cli/README.md."
                .into(),
        );
    }
    match now {
        None => Ok(false),
        Some(now) if now == identity => Ok(true),
        Some(_) => Err("The saved process number now belongs to a different process. Nothing was signaled. Leave these files in place; see Recovery in cli/README.md.".into()),
    }
}

fn pidfd(pid: u32) -> Result<Option<File>, String> {
    if !cfg!(all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )) {
        return Err("Safe process handles need 64-bit Linux with pidfd support. Update the machine's kernel.".into());
    }
    // Linux shares these new syscall numbers on both supported architectures.
    let fd = unsafe { syscall(434, pid as i32, 0u32) };
    if fd == -1 {
        if std::io::Error::last_os_error().raw_os_error() == Some(3) {
            return Ok(None);
        }
        return Err(
            "Cannot open a safe process handle. Check Linux pidfd support with vade doctor.".into(),
        );
    }
    Ok(Some(unsafe { File::from_raw_fd(fd as i32) }))
}

pub(super) fn check_pidfd() -> Result<(), String> {
    let handle = pidfd(unsafe { getpid() } as u32)?
        .ok_or_else(|| "Cannot test a safe process handle. Run vade doctor.".to_string())?;
    send_pidfd(&handle, 0)
}

fn send_pidfd(handle: &File, number: i32) -> Result<(), String> {
    let result = unsafe {
        syscall(
            424,
            handle.as_raw_fd(),
            number,
            std::ptr::null::<u8>(),
            0u32,
        )
    };
    if result == -1 && std::io::Error::last_os_error().raw_os_error() != Some(3) {
        return Err("Could not safely signal the engine. Retry vade down; if it still fails, see Recovery in cli/README.md.".into());
    }
    Ok(())
}

pub(super) fn stop_identity(identity: &Identity) -> Result<(), String> {
    // Open before verifying: a reused numeric PID can never redirect this handle.
    let handle = pidfd(identity.pid)?;
    if !verify(identity)? {
        return Ok(());
    }
    let handle = handle
        .ok_or_else(|| "The engine process changed before stop. Leave these files in place; see Recovery in cli/README.md.".to_string())?;
    send_pidfd(&handle, TERM)?;
    if wait_stopped(identity, Duration::from_secs(30))? {
        return Ok(());
    }
    if verify(identity)? {
        send_pidfd(&handle, KILL)?;
    }
    if wait_stopped(identity, Duration::from_secs(5))? {
        return Ok(());
    }
    Err("The owned engine did not stop. Its saved files remain; retry vade down or see Recovery in cli/README.md.".into())
}

fn wait_stopped(identity: &Identity, timeout: Duration) -> Result<bool, String> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !verify(identity)? {
            return Ok(true);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Ok(false)
}

pub(super) fn rollback_child(child: &mut Child) -> Result<(), String> {
    if child
        .try_wait()
        .map_err(|_| "Cannot check the newly started engine. Leave these files in place; see Recovery in cli/README.md.".to_string())?
        .is_some()
    {
        return Ok(());
    }
    unsafe {
        kill(child.id() as i32, TERM);
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if child
            .try_wait()
            .map_err(|_| {
                "Cannot check the newly started engine. Leave these files in place; see Recovery in cli/README.md.".to_string()
            })?
            .is_some()
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    child.kill().map_err(|_| {
        "Could not stop the failed launch. Leave these files in place; see Recovery in cli/README.md."
            .to_string()
    })?;
    child
        .wait()
        .map_err(|_| "Could not confirm the failed launch stopped. Leave these files in place; see Recovery in cli/README.md.".to_string())?;
    Ok(())
}
