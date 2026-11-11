//! Flash Next text engine. Official FP8/BF16 weights, FP32 activations.
//! Resident rank weights and the embedding, HC and PLE execution blocks.
pub mod weights;
pub mod nvfp4;
pub mod rows;
pub mod blocks;
pub mod gdn;
pub mod qsa;
pub use weights::{TensorSlice, Weights};

pub mod moe;
pub mod head;
pub mod next;
pub mod next_batched;
mod next_load;
pub mod next_native;
pub mod session;
#[cfg(test)]
mod router_tests;

/// NVFP4 uses explicit W4A16 routed experts; FP8 remains the default.
pub fn require_inference(format: engine_model::flash::WeightFormat) -> Result<(), String> {
    if format == engine_model::flash::WeightFormat::Nvfp4 {
        format.require_opt_in(engine_model::flash::WeightFormat::opt_in().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Larger Flash deployments use an open chain; pairs keep the ring default.
pub fn path_topology(world: usize) -> Result<bool, String> {
    match std::env::var("TP_TOPOLOGY").as_deref() {
        Ok("path") => Ok(true), Ok("ring") => Ok(false), Err(_) => Ok(world > 2),
        _ => Err("TP_TOPOLOGY must be ring or path".into()),
    }
}

