//! Draft ownership is shape-driven; only a verified consecutive target prefix commits.
use engine_model::plan::ModelPlan;
pub fn validate(p: &ModelPlan) -> Result<(), String> {
    engine_model::qwen3_5::plan::validate_typed_mtp(p)
}
pub fn block_prefix(layers: usize, layer: usize) -> String {
    if layer < layers { format!("{}.layers.{layer}",engine_model::qwen3_5::PREFIX) } else { format!("mtp.layers.{}", layer-layers) }
}
pub fn depth(configured: usize, limit: usize) -> usize { limit.min(if configured == 0 { 3 } else { configured }) }
pub fn verified_prefix(entry: usize, rows: usize, position: usize) -> Result<usize, String> {
    position.checked_sub(entry).filter(|&n| n > 0 && n <= rows).ok_or("MTP requires an owned verified consecutive prefix".into())
}
