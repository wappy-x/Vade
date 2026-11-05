//! Head/output-row-sharded tensor parallelism for any number of DGX Sparks.
pub mod transport;
pub mod tensor_layout;
pub mod layer_transport;
#[path = "flash/next_tp_proxy.rs"]
#[cfg(target_os="linux")]
pub mod tensor_wire;
pub mod layer_driver;
pub mod layer_operands;
pub mod session_state;
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
pub mod ordered_projection;
pub mod mxfp8_projection;
pub mod dynamic_projection;
pub mod causal_layer;
pub mod bounded_decoder;
pub mod row_split;
pub mod head_path;
pub mod resident_weights;
mod resident_profile;
pub mod expert_path;
