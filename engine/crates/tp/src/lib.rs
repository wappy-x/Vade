//! Head/output-row-sharded tensor parallelism for any number of DGX Sparks.
pub mod transport;
pub use engine_fabric::discovery;
pub mod runtime;
pub mod startup_pool;
pub mod flash_pool;
pub mod next;
#[cfg(any(target_os="linux",test))]
mod service_commands;
#[cfg(target_os = "linux")]
pub mod model;
#[cfg(target_os = "linux")]
pub mod service;
#[cfg(target_os="linux")]
mod rdma;

pub type Result<T> = std::result::Result<T, String>;

/// Contiguous, balanced rows; no head-count or power-of-two restriction.
pub fn rows(total: usize, rank: usize, world: usize) -> std::ops::Range<usize> {
    let base = total / world;
    let extra = total % world;
    let start = base * rank + rank.min(extra);
    start..start + base + usize::from(rank < extra)
}

#[cfg(target_os="linux")]
pub mod next_gpu;
pub mod next_bw;
pub mod next_http;
pub mod next_timing;
