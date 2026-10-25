//! Startup permits light display activity without compute apps; shutdown requires zero.
use super::query;
use std::time::{Duration, Instant};

/// `departing`: after a stop, the driver may list the stopped server for a
/// moment, so a listed compute process is a busy sample, not someone else's job.
fn idle(timeout: Duration, maximum_utilization: u32, departing: bool) -> Result<bool, String> {
    let began = Instant::now();
    let apps = query("nvidia-smi", &["--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], timeout)?;
    if !apps.trim().is_empty() {
        if departing { return Ok(false); }
        return Err("This machine's GPU is in use. Stop the compute application with its original launcher, then retry; run vade doctor to check this machine.".into());
    }
    let remaining = timeout.saturating_sub(began.elapsed());
    if remaining.is_zero() { return Err("GPU inspection timed out. Keep any ownership record and run vade doctor before retrying.".into()); }
    let utilization = query("nvidia-smi", &["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], remaining)?;
    utilization_idle(&utilization, maximum_utilization)
}

fn utilization_idle(value: &str, maximum: u32) -> Result<bool, String> {
    let values = value.lines().map(|v| v.trim().parse::<u32>()).collect::<Result<Vec<_>, _>>()
        .map_err(|_| "GPU utilization is unavailable. Run vade doctor; keep any ownership record.")?;
    if values.is_empty() || values.iter().any(|v| *v > 100) {
        return Err("GPU utilization is unavailable. Run vade doctor; keep any ownership record.".into());
    }
    Ok(values.iter().all(|v| *v <= maximum))
}

pub(super) fn gpu_empty() -> Result<(), String> {
    if wait_idle(Duration::from_secs(3), |budget| idle(budget, 5, false))? { return Ok(()); }
    Err("This machine's GPU is busy or unavailable. Run vade doctor before starting.".into())
}

pub(super) fn gpu_settled() -> Result<(), String> {
    settle(Duration::from_secs(10), |budget| idle(budget, 0, true))
}

fn settle(budget: Duration, sample: impl FnMut(Duration) -> Result<bool, String>) -> Result<(), String> {
    if wait_idle(budget, sample)? { return Ok(()); }
    Err("The owned server stopped, but the GPU still lists a compute process or shows activity. Keep the ownership record, wait a moment and retry vade down; vade doctor shows what is running.".into())
}

fn wait_idle(budget: Duration, mut sample: impl FnMut(Duration) -> Result<bool, String>) -> Result<bool, String> {
    let deadline = Instant::now() + budget;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { break; }
        // A remaining compute process or an inspection error is never retried.
        if sample(remaining.min(Duration::from_secs(2)))? { return Ok(true); }
        std::thread::sleep(Duration::from_millis(100).min(deadline.saturating_duration_since(Instant::now())));
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shutdown_retries_only_busy_samples_and_never_an_inspection_failure() {
        let mut samples = 0;
        settle(Duration::from_secs(1), |_| { samples += 1; Ok(samples == 3) }).unwrap();
        assert_eq!(samples, 3);
        let mut failed = 0;
        let error = settle(Duration::from_secs(1), |_| { failed += 1; Err("foreign compute app".into()) }).unwrap_err();
        assert_eq!(error, "foreign compute app");
        assert_eq!(failed, 1);
        assert!(settle(Duration::from_millis(2), |_| Ok(false)).unwrap_err().contains("retry vade down"));
    }
    #[test]
    fn utilization_requires_valid_zero_samples_from_every_gpu() {
        assert!(utilization_idle("0\n0\n", 0).unwrap());
        assert!(!utilization_idle("0\n7\n", 0).unwrap());
        assert!(utilization_idle("0\n5\n", 5).unwrap());
        assert!(!utilization_idle("0\n6\n", 5).unwrap());
        assert!(!utilization_idle("0\n1\n", 0).unwrap());
        for invalid in ["", "N/A\n", "-1\n", "101\n", "0\n\n"] {
            assert!(utilization_idle(invalid, 5).is_err(), "{invalid:?}");
        }
    }
}
