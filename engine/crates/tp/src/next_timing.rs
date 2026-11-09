//! Opt-in world-2/4 step timer (VADE_NEXT_STEP_TIMING=1, read once per
//! process). Each Next step's host wall time is split into rank exchanges
//! (data-plane gathers through the shared transport, plus the command
//! broadcast/verdict: moving bytes to and waiting on the peers) and the rest
//! (this rank's kernel launches, kernels, read-backs and host work). Timers
//! only: commands, transfers and arithmetic are unchanged. Off by default.
use std::{
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::Instant,
};

const WINDOW: u64 = 64;
static STEPS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static STEP_NS: AtomicU64 = AtomicU64::new(0);
static DATA_NS: AtomicU64 = AtomicU64::new(0);
static DATA_CALLS: AtomicU64 = AtomicU64::new(0);
static COMMAND_NS: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_NEXT_STEP_TIMING").is_ok_and(|v| v == "1"))
}
pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}
fn ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
/// One data-plane gather (rank-major all-gather of floats or bytes).
pub fn data(start: Option<Instant>) {
    if let Some(t) = start {
        DATA_NS.fetch_add(ns(t), Relaxed);
        DATA_CALLS.fetch_add(1, Relaxed);
    }
}
/// Command-plane time outside the step itself (broadcast and verdict).
pub fn command(start: Option<Instant>) {
    if let Some(t) = start {
        COMMAND_NS.fetch_add(ns(t), Relaxed);
    }
}
/// One finished step of `rows` tokens; prints and restarts every WINDOW steps.
pub fn step(start: Option<Instant>, rank: usize, rows: usize) {
    let Some(t) = start else { return };
    STEP_NS.fetch_add(ns(t), Relaxed);
    ROWS.fetch_add(rows as u64, Relaxed);
    if STEPS.fetch_add(1, Relaxed) + 1 < WINDOW {
        return;
    }
    let steps = STEPS.swap(0, Relaxed).max(1) as f64;
    let ms = |v: &AtomicU64| v.swap(0, Relaxed) as f64 / 1e6 / steps;
    let (rows, step, data, calls, command) = (
        ROWS.swap(0, Relaxed),
        ms(&STEP_NS),
        ms(&DATA_NS),
        DATA_CALLS.swap(0, Relaxed) as f64 / steps,
        ms(&COMMAND_NS),
    );
    eprintln!(
        "next_step_timing rank={rank} steps={steps} rows={rows} step_ms={step:.3} data_exchange_ms={data:.3} data_exchanges_per_step={calls:.1} local_ms={:.3} command_ms={command:.3}",
        step - data
    );
}
