//! Owned GB10 engine: BF16 weights, device-resident FP32 activations and KV.
#[path = "hybrid/arms.rs"]
pub mod arms;
pub mod context;
pub mod draft;
#[cfg(target_os = "linux")]
pub mod flash;
// Compile the production TCP/relay protocol tests on hosts without the CUDA backend.
#[cfg(all(test, not(target_os = "linux")))]
#[allow(dead_code)]
#[path = "flash/next_pp.rs"]
mod flash_next_pp_host;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/candidate.rs"]
pub mod flash_candidate;
#[path = "flash/draft_stack.rs"]
pub mod flash_draft_stack;
#[path = "flash/tp_switch.rs"]
pub mod flash_tp_switch;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/mx8_registry.rs"]
pub(crate) mod flash_mx8_registry;
#[cfg(any(target_os = "linux", test))]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[path = "flash/split_ranges.rs"]
pub(crate) mod flash_split_ranges;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/next_input.rs"]
mod flash_next_input;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/next_workspace.rs"]
mod flash_next_workspace;
#[path = "flash/head_options.rs"]
pub mod flash_head_options;
#[path = "flash/prompt_options.rs"]
pub mod flash_prompt_options;
#[path = "flash/next_options.rs"]
pub mod flash_next_options;
#[path = "flash/next_lookup.rs"]
pub mod flash_next_lookup;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/next_prefill.rs"]
mod flash_next_prefill;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/next_read_plan.rs"]
mod flash_next_read_plan;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/next_projection.rs"]
mod flash_next_projection;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/next_stage_plan.rs"]
pub(crate) mod flash_next_stage_plan;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/next_split_plan.rs"]
pub(crate) mod flash_next_split_plan;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/next_xsplit_plan.rs"]
pub(crate) mod flash_next_xsplit_plan;
#[path = "flash/next_rows.rs"]
pub mod flash_next_rows;
#[cfg(any(target_os = "linux", test))]
#[path = "flash/next_storage.rs"]
mod flash_next_storage;
#[path = "flash/verification.rs"]
pub mod flash_verification;
#[cfg(target_os = "linux")]
pub mod hybrid;
#[path = "hybrid/draft_control.rs"]
pub mod hybrid_draft;
#[cfg(target_os = "linux")]
pub mod layer;
#[cfg(target_os = "linux")]
pub mod qwen;
pub mod sample;
pub use engine_fabric::profile as collective;

/// Greedy policy shared by runners: finite logits, lowest token ID wins a tie.
pub fn greedy_token(logits: &[f32]) -> Result<u32, String> {
    sample::Sampler::new(sample::Sampling::greedy())?.sample(logits, &[], 0, 0, |_| true)
}

/// Projections load in packed8 layout; raw matrix probes retain Vec8 support.
#[cfg(target_os = "linux")]
pub fn configured_matvec_mode() -> Result<spark_gpu_probe::driver::MatvecMode, String> {
    Ok(spark_gpu_probe::driver::MatvecMode::Vec8)
}

#[cfg(target_os = "linux")]
pub mod gemma4;
