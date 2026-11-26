//! Shared checkpoint admission and compatibility geometry for CLI and HTTP.
#[allow(unused_imports)] // Host selection tests include this compatibility module directly.
pub use engine_model::qwen3_5::plan::{config, inspect, uses_typed_loader};

pub fn require_world(world: usize) -> Result<(), String> {
    if world != 1 { Err("hybrid currently requires world size 1".into()) } else { Ok(()) }
}
