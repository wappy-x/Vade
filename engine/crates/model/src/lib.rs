//! Bounded model-file inspection and dense Qwen3 checkpoint validation.
pub mod flash;
pub mod csf;
pub mod host_ops;
pub mod norm_cap;
pub mod expert_widen;
pub mod capped_moe;
pub mod capped_head;
pub mod dsa;
pub mod mhc;
pub mod mla;
pub mod placement;
pub mod sampling_policy;
pub mod plan;
pub mod cartridge;
pub mod gemma4;
pub mod gemma4_assistant;
pub mod split;
pub use engine_base::{Error, Result, events, json};
pub mod qwen3;
pub mod qwen3_5;
pub mod phi4;
pub mod llama;
pub mod kda;
pub mod block_mix;
pub mod visit_loop;
pub mod safetensors;
pub mod layout;
pub mod loop_host;
pub mod shapes;
pub mod short_conv;
pub mod sink_attention;
pub mod swiglu;
/// Fixed FP32 math shared with the GPU crate and its kernels: one source, included by path.
/// A host form that must give the device's bytes takes `exp`, `sigmoid`, `tanh`, `ln` and `rn` from here.
#[path = "../../gpu/src/fixed_math.rs"]
pub mod fixed_math;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error(message.into())
}
