//! A GPU is busy only with a compute app or power draw at or above 30 W.
//! Utilization alone never blocks: after a server exits, the GB10 counter
//! can stay at its last value with no apps and low power.
use super::query;
use std::time::{Duration, Instant};

const BUSY_POWER_WATTS: f64 = 30.0;

#[derive(Debug, PartialEq)]
enum GpuState { Idle, Stale(u32), BusyPower, Apps }

/// Judge one recorded sample set: compute apps, then power draw, then
/// utilization only to name a stale counter.
fn gpu_state(apps: &str, power: &str, utilization: &str) -> Result<GpuState, String> {
    if !apps.trim().is_empty() { return Ok(GpuState::Apps); }
    let unavailable = "GPU power draw is unavailable. Run vade doctor; keep any ownership record.";
    let powers = power.lines().map(|v| v.trim().parse::<f64>()).collect::<Result<Vec<_>, _>>()
        .map_err(|_| unavailable)?;
    if powers.is_empty() || powers.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(unavailable.into());
    }
    if powers.iter().any(|v| *v >= BUSY_POWER_WATTS) { return Ok(GpuState::BusyPower); }
    // Utilization cannot block, so an unreadable value names no stale counter.
    match utilization.lines().filter_map(|v| v.trim().parse::<u32>().ok()).filter(|v| *v <= 100).max() {
        Some(v) if v > 0 => Ok(GpuState::Stale(v)),
        _ => Ok(GpuState::Idle),
    }
}

pub(crate) fn stale_line(utilization: u32) -> String {
    format!("Stale GPU utilization counter: {utilization} % with no compute apps and power draw under 30 W; continuing.")
}

enum Sample { Idle(Option<u32>), Busy }

/// `departing`: after a stop, the driver may list the stopped server for a
/// moment, so a listed compute process is a busy sample, not someone else's job.
fn idle(timeout: Duration, departing: bool) -> Result<Sample, String> {
    let began = Instant::now();
    let remaining = |began: Instant| timeout.saturating_sub(began.elapsed());
    let apps = query("nvidia-smi", &["--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], timeout)?;
    if !apps.trim().is_empty() {
        if departing { return Ok(Sample::Busy); }
        return Err("This machine's GPU is in use. Stop the compute application with its original launcher, then retry; run vade doctor to check this machine.".into());
    }
    if remaining(began).is_zero() { return Err("GPU inspection timed out. Keep any ownership record and run vade doctor before retrying.".into()); }
    let power = query("nvidia-smi", &["--query-gpu=power.draw", "--format=csv,noheader,nounits"], remaining(began))?;
    // Utilization is advisory only; a failed read names no stale counter.
    let utilization = query("nvidia-smi", &["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], remaining(began)).unwrap_or_default();
    Ok(match gpu_state(&apps, &power, &utilization)? {
        GpuState::Apps if departing => Sample::Busy,
        GpuState::Apps => return Err("This machine's GPU is in use. Stop the compute application with its original launcher, then retry; run vade doctor to check this machine.".into()),
        GpuState::BusyPower => Sample::Busy,
        GpuState::Stale(v) => Sample::Idle(Some(v)),
        GpuState::Idle => Sample::Idle(None),
    })
}

/// The stale utilization value to report, if the GPU is idle with one.
pub(super) fn gpu_empty() -> Result<Option<u32>, String> {
    if let Some(stale) = wait_idle(Duration::from_secs(3), |budget| idle(budget, false))? { return Ok(stale); }
    Err("This machine's GPU is busy or unavailable. Run vade doctor before starting.".into())
}

pub(super) fn gpu_settled() -> Result<Option<u32>, String> {
    settle(Duration::from_secs(10), |budget| idle(budget, true))
}

fn settle(budget: Duration, sample: impl FnMut(Duration) -> Result<Sample, String>) -> Result<Option<u32>, String> {
    if let Some(stale) = wait_idle(budget, sample)? { return Ok(stale); }
    Err("The owned server stopped, but the GPU still lists a compute process or shows activity. Keep the ownership record, wait a moment and retry vade down; vade doctor shows what is running.".into())
}

fn wait_idle(budget: Duration, mut sample: impl FnMut(Duration) -> Result<Sample, String>) -> Result<Option<Option<u32>>, String> {
    let deadline = Instant::now() + budget;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { break; }
        // A foreign compute app or an inspection error is never retried.
        if let Sample::Idle(stale) = sample(remaining.min(Duration::from_secs(2)))? { return Ok(Some(stale)); }
        std::thread::sleep(Duration::from_millis(100).min(deadline.saturating_duration_since(Instant::now())));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shutdown_retries_only_busy_samples_and_never_an_inspection_failure() {
        let mut samples = 0;
        settle(Duration::from_secs(1), |_| { samples += 1; Ok(if samples == 3 { Sample::Idle(None) } else { Sample::Busy }) }).unwrap();
        assert_eq!(samples, 3);
        let mut failed = 0;
        let error = settle(Duration::from_secs(1), |_| { failed += 1; Err("foreign compute app".into()) }).unwrap_err();
        assert_eq!(error, "foreign compute app");
        assert_eq!(failed, 1);
        assert!(settle(Duration::from_millis(2), |_| Ok(Sample::Busy)).unwrap_err().contains("retry vade down"));
        assert_eq!(settle(Duration::from_secs(1), |_| Ok(Sample::Idle(Some(96)))).unwrap(), Some(96));
    }
    #[test]
    fn recorded_samples_judge_idle_stale_apps_and_power() {
        // Idle: no apps, low power, zero utilization.
        assert_eq!(gpu_state("", "18.50\n", "0\n").unwrap(), GpuState::Idle);
        // Stale counter: no apps and 19 W, so 96 % cannot block.
        assert_eq!(gpu_state("", "19.00\n", "96\n").unwrap(), GpuState::Stale(96));
        // Real work: a listed compute app blocks at any power or utilization.
        assert_eq!(gpu_state("1234, flash-serve, 1024 MiB\n", "19.00\n", "0\n").unwrap(), GpuState::Apps);
        // Power only: no app is listed, but draw at or above 30 W is busy.
        assert_eq!(gpu_state("", "45.20\n", "0\n").unwrap(), GpuState::BusyPower);
        assert_eq!(gpu_state("", "30.00\n", "0\n").unwrap(), GpuState::BusyPower);
        assert_eq!(gpu_state("", "29.99\n", "5\n").unwrap(), GpuState::Stale(5));
        // Every GPU counts for power and for the stale value.
        assert_eq!(gpu_state("", "18.00\n31.00\n", "0\n").unwrap(), GpuState::BusyPower);
        assert_eq!(gpu_state("", "18.00\n19.00\n", "0\n96\n").unwrap(), GpuState::Stale(96));
        // Unreadable utilization alone never blocks; unreadable power does.
        assert_eq!(gpu_state("", "19.00\n", "N/A\n").unwrap(), GpuState::Idle);
        for invalid in ["", "N/A\n", "-1.00\n", "19.00\n\n"] {
            assert!(gpu_state("", invalid, "0\n").is_err(), "{invalid:?}");
        }
        assert!(stale_line(96).contains("96 %") && stale_line(96).contains("no compute apps"));
    }
}
