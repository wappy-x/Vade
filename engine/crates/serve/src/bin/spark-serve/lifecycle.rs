//! Linux process boundary. All CUDA work begins after detaching.
#![allow(unsafe_code)]
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, Write},
    os::{fd::AsRawFd, unix::{fs::OpenOptionsExt, net::UnixStream}},
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);
unsafe extern "C" {
    fn fork() -> i32;
    fn setsid() -> i32;
    fn dup2(old: i32, new: i32) -> i32;
    fn flock(fd: i32, operation: i32) -> i32;
    fn signal(number: i32, handler: usize) -> usize;
}
extern "C" fn stop(_: i32) {
    // No allocation, I/O, locks or CUDA calls in the signal handler.
    SHUTDOWN.store(true, Ordering::Relaxed);
    spark_gpu_probe::teardown::arm();
}

pub struct PidFile { path: PathBuf, _lock: File }
impl Drop for PidFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.path); }
}

/// None is the successful launcher parent; Some is the resident child.
/// Called on the single main thread, before tokenizer threads or CUDA exist.
pub fn detach() -> Result<Option<PidFile>, String> {
    let state = std::env::var_os("XDG_STATE_HOME").filter(|p| !p.is_empty())
        .map(PathBuf::from).filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".local/state")))
        .ok_or("set XDG_STATE_HOME or HOME for --daemon")?.join("vade");
    std::fs::create_dir_all(&state).map_err(|e| e.to_string())?;
    let path = state.join("spark-serve.pid");
    let mut lock = OpenOptions::new().read(true).write(true).create(true)
        .truncate(false).mode(0o600).open(&path).map_err(|e| e.to_string())?;
    // Lock the inode, not the recorded PID: stale files never block a restart.
    if unsafe { flock(lock.as_raw_fd(), 2 | 4) } != 0 {
        return Err(format!("daemon already running or PID file locked: {}", path.display()));
    }
    let log_path = state.join("spark-serve.log");
    let log = OpenOptions::new().create(true).append(true).mode(0o600)
        .open(&log_path).map_err(|e| e.to_string())?;
    let null = File::open("/dev/null").map_err(|e| e.to_string())?;
    let (mut parent, mut child) = UnixStream::pair().map_err(|e| e.to_string())?;
    parent.set_read_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
    // No other threads exist here; the child owns its inherited Rust state.
    let pid = unsafe { fork() };
    if pid < 0 { return Err(std::io::Error::last_os_error().to_string()); }
    if pid > 0 {
        drop(child);
        let mut ready = [0];
        parent.read_exact(&mut ready).map_err(|e| format!("daemon detach failed: {e}"))?;
        println!("daemon pid={pid} pid_file={} log={}; loading model, wait for ready bind= in log",
            path.display(), log_path.display());
        return Ok(None);
    }
    drop(parent);
    let guard = PidFile { path, _lock: lock.try_clone().map_err(|e| e.to_string())? };
    if unsafe { setsid() } < 0 { return Err(std::io::Error::last_os_error().to_string()); }
    for (source, dest) in [(null.as_raw_fd(), 0), (log.as_raw_fd(), 1), (log.as_raw_fd(), 2)] {
        if unsafe { dup2(source, dest) } < 0 { return Err(std::io::Error::last_os_error().to_string()); }
    }
    lock.set_len(0).and_then(|_| lock.rewind())
        .and_then(|_| writeln!(lock, "{}", std::process::id())).map_err(|e| e.to_string())?;
    child.write_all(&[1]).map_err(|e| e.to_string())?;
    Ok(Some(guard))
}

pub fn install_signals() -> Result<(), String> {
    for number in [2, 15] {
        // Linux signal(2) takes a function pointer and returns SIG_ERR on failure.
        if unsafe { signal(number, stop as *const () as usize) } == usize::MAX {
            return Err(std::io::Error::last_os_error().to_string());
        }
    }
    spark_gpu_probe::teardown::install()?;
    Ok(())
}
