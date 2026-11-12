//! Sequence-parallel prefill: replicated model, contiguous positions, ring K/V exchange.
pub type Result<T> = std::result::Result<T, String>;
pub use engine_fabric::discovery;
pub mod partition;
pub mod transport;
pub use partition::Partition;
pub use transport::{Ring, TransportStats};
#[cfg(target_os = "linux")]
pub mod prefill;
#[cfg(target_os = "linux")]
pub mod coordinator;
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub use engine_fabric::verbs;

/// Aggregate CUDA allocation cap. Leave headroom for host staging and the OS.
#[cfg(target_os = "linux")]
pub fn memory_budget() -> Result<usize> {
    let bytes = match std::env::var("SEQPAR_MEMORY_BYTES") {
        Ok(value) => value.parse::<usize>().map_err(|_| "invalid SEQPAR_MEMORY_BYTES")?,
        Err(std::env::VarError::NotPresent) => {
            let info = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
            let kib = info.lines().find_map(|line| line.strip_prefix("MemAvailable:"))
                .and_then(|line| line.split_whitespace().next()).and_then(|value| value.parse::<usize>().ok())
                .ok_or("missing MemAvailable; set SEQPAR_MEMORY_BYTES")?;
            kib.checked_mul(1024).and_then(|n| n.checked_mul(9)).map(|n| n / 10)
                .ok_or("available memory byte count overflow")?
        }
        Err(error) => return Err(error.to_string()),
    };
    if bytes == 0 { return Err("SEQPAR_MEMORY_BYTES must be positive".into()); }
    Ok(bytes)
}

pub mod control;
