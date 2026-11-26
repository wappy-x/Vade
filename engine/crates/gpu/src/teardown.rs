//! Serving processes bound fatal cleanup even when the driver stops responding.
use std::{cell::Cell, sync::{Arc, Mutex, OnceLock, Weak, atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering}}, time::{Duration, Instant}};

const TEARDOWN_TIMEOUT: Duration = Duration::from_millis(4500);
static PROGRESS_TIMEOUT: OnceLock<Result<Duration, String>> = OnceLock::new();
const EXPIRED: u64 = u64::MAX;
static ARMED: AtomicBool = AtomicBool::new(false);
static SUPERVISED: AtomicBool = AtomicBool::new(false);
static INSTALLED: OnceLock<Result<(), String>> = OnceLock::new();
static EPOCH: OnceLock<Instant> = OnceLock::new();
static WAITS: Mutex<Vec<Weak<WaitState>>> = Mutex::new(Vec::new());

fn progress_timeout() -> Result<Duration, String> {
    PROGRESS_TIMEOUT.get_or_init(|| {
        let seconds = match std::env::var("VADE_GPU_WAIT_TIMEOUT_SECS") {
            Ok(value) => value.parse::<u64>().ok().filter(|seconds| *seconds <= u64::MAX / 1000)
                .ok_or("VADE_GPU_WAIT_TIMEOUT_SECS must be a nonnegative whole number of seconds (0 disables the GPU wait bound)")?,
            Err(std::env::VarError::NotPresent) => 120,
            Err(_) => return Err("VADE_GPU_WAIT_TIMEOUT_SECS must contain a whole number of seconds".into()),
        };
        Ok(Duration::from_secs(seconds))
    }).clone()
}

/// Install after any daemon fork and before creating GPU resources.
/// GPU probes only retire on a stalled wait, not on an ordinary returned error.
pub fn install() -> Result<(), String> {
    SUPERVISED.store(true, Ordering::Relaxed);
    INSTALLED.get_or_init(|| start(TEARDOWN_TIMEOUT)).clone()
}

/// Signal-safe: no allocation, I/O, locks, or driver calls.
/// Repeated errors never extend the deadline.
pub fn arm() { ARMED.store(true, Ordering::Relaxed); }

#[derive(Clone, Copy)]
pub(crate) enum WaitKind { Context, Stream, Event, Cleanup, Readback, Initialize }
impl WaitKind {
    fn name(value: u8) -> &'static str {
        match value { 0 => "CUDA context synchronization", 1 => "CUDA stream synchronization",
            2 => "CUDA event synchronization", 3 => "CUDA resource cleanup",
            4 => "CUDA readback", _ => "CUDA context initialization" }
    }
}
struct WaitState { deadline: AtomicU64, kind: AtomicU8 }

/// One registration per CUDA context, with no allocation or lock per wait.
/// Nested cleanup retains the outer deadline rather than extending it.
pub(crate) struct Progress { state: Arc<WaitState>, depth: Cell<usize>, timeout: Duration }
pub(crate) struct Wait<'a> { progress: &'a Progress }
impl Progress {
    pub(crate) fn new() -> Result<Self, String> {
        INSTALLED.get_or_init(|| start(TEARDOWN_TIMEOUT)).clone()?;
        let state = Arc::new(WaitState { deadline: AtomicU64::new(0), kind: AtomicU8::new(0) });
        WAITS.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&state));
        Ok(Self { state, depth: Cell::new(0), timeout: progress_timeout()? })
    }
    pub(crate) fn wait(&self, kind: WaitKind) -> Wait<'_> {
        self.wait_for(kind, self.timeout)
    }
    fn wait_for(&self, kind: WaitKind, timeout: Duration) -> Wait<'_> {
        if self.depth.get() == 0 && !timeout.is_zero() {
            self.state.kind.store(kind as u8, Ordering::Relaxed);
            // Zero denotes idle; round up so a sub-millisecond wait is nonzero.
            self.state.deadline.store(ticks().saturating_add(timeout.as_millis() as u64).saturating_add(1), Ordering::Release);
        }
        self.depth.set(self.depth.get() + 1);
        let wait = Wait { progress: self };
        wait.poll();
        wait
    }
}
impl Wait<'_> {
    pub(crate) fn poll(&self) {
        if self.progress.state.deadline.load(Ordering::Acquire) == EXPIRED { retain_until_exit(); }
    }
}
impl Drop for Wait<'_> {
    fn drop(&mut self) {
        let depth = self.progress.depth.get() - 1;
        self.progress.depth.set(depth);
        if depth == 0 {
            let deadline = self.progress.state.deadline.load(Ordering::Acquire);
            // The supervisor claims an expired wait before the owner can leave
            // this scope and drop/reuse GPU buffers or registered NIC memory.
            if deadline == EXPIRED || self.progress.state.deadline.compare_exchange(
                deadline, 0, Ordering::AcqRel, Ordering::Acquire).is_err() { retain_until_exit(); }
        } else { self.poll(); }
    }
}
fn ticks() -> u64 { EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64 }
fn retain_until_exit() -> ! {
    // No unwinding: in-flight host/GPU/NIC owners remain on this thread's stack.
    loop { std::thread::park(); }
}
fn retire_process() -> ! {
    // C atexit hooks can re-enter a stalled driver. Fatal retirement must
    // bypass those hooks as well as Rust drops and preserve in-flight owners.
    #[cfg(unix)]
    unsafe {
        unsafe extern "C" { fn _exit(status: std::ffi::c_int) -> !; }
        _exit(1)
    }
    #[cfg(not(unix))]
    std::process::exit(1)
}

fn start(timeout: Duration) -> Result<(), String> {
    let progress_timeout = progress_timeout()?;
    std::thread::Builder::new().name("teardown-deadline".into()).spawn(move || {
        let mut retire_at = None;
        let mut reported = false;
        loop {
            let now = Instant::now();
            if SUPERVISED.load(Ordering::Relaxed) && ARMED.load(Ordering::Relaxed) {
                retire_at.get_or_insert(now + timeout);
            }
            let stalled = {
                let mut waits = WAITS.lock().unwrap_or_else(|e| e.into_inner());
                waits.retain(|wait| wait.strong_count() != 0);
                waits.iter().filter_map(Weak::upgrade).find_map(|wait| {
                    let deadline = wait.deadline.load(Ordering::Acquire);
                    if deadline == 0 || deadline != EXPIRED && ticks() < deadline { return None; }
                    if deadline != EXPIRED && wait.deadline.compare_exchange(deadline, EXPIRED,
                        Ordering::AcqRel, Ordering::Acquire).is_err() { return None; }
                    Some(WaitKind::name(wait.kind.load(Ordering::Relaxed)))
                })
            };
            if let Some(operation) = stalled {
                retire_at.get_or_insert(now + timeout);
                if !reported {
                    reported = true;
                    // Stderr may be locked or its pipe full. Logging must never
                    // block the independent retirement deadline.
                    let _ = std::thread::Builder::new().name("gpu-stall-error".into()).spawn(move || {
                        eprintln!("fatal GPU progress timeout: {operation} exceeded {} seconds; retiring process with in-flight allocations retained", progress_timeout.as_secs());
                    });
                }
            }
            if retire_at.is_some_and(|deadline| now >= deadline) { retire_process(); }
            std::thread::sleep(Duration::from_millis(10));
        }
    }).map(|_| ()).map_err(|e| e.to_string())
}

#[cfg(test)]
#[test]
fn fatal_exit_without_signal() {
    if std::env::var_os("VADE_TEARDOWN_TEST_CHILD").is_some() { SUPERVISED.store(true, Ordering::Relaxed); start(Duration::from_millis(20)).unwrap(); arm(); loop { std::thread::park(); } }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "teardown::fatal_exit_without_signal"])
        .env("VADE_TEARDOWN_TEST_CHILD", "1").status().unwrap();
    assert_eq!(status.code(), Some(1));
}
