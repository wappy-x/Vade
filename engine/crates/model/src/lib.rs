//! Bounded model-file inspection and dense Qwen3 checkpoint validation.
pub mod flash;
pub mod csf;
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
pub mod safetensors;
pub mod shapes;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error(message.into())
}
