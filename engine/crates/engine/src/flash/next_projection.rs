//! MXFP8 selection for the text-only Flash-Next decode path.
use engine_model::flash::mixed::{Mixed, Storage};
pub(crate) fn text_mxfp8(q: &Mixed, module: &str) -> bool {
    // NextModel executes text decode only; neither vision nor MTP is enabled.
    module.starts_with("model.language_model.") && q.storage(module) == Storage::Mxfp8 }

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    #[test]
    fn text_mxfp8_projection_count_excludes_vision_and_mtp() {
        let q = Mixed { ignore: Vec::new(), layers: BTreeMap::from([
            ("model".into(), Storage::Mxfp8), ("mtp".into(), Storage::Mxfp8)]) };
        let mut names = Vec::new();
        for li in 0..48 {
            let base = format!("model.language_model.layers.{li}");
            let (attn, projections) = if li % 4 == 3 {
                ("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj"])
            } else { ("linear_attn", ["in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"]) };
            names.extend(projections.map(|p| format!("{base}.{attn}.{p}.weight")));
            names.extend(["gate_proj", "up_proj", "down_proj"].map(|p|
                format!("{base}.mlp.shared_expert.{p}.weight")));
        }
        for li in 0..27 {
            names.extend(["qkv", "proj"].map(|p| format!("model.visual.blocks.{li}.attn.{p}.weight")));
        }
        assert_eq!(names.len(), 438);
        names.extend(["q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj"].map(|p|
            format!("mtp.layers.0.self_attn.{p}.weight")));
        assert_eq!(names.iter().filter_map(|n| n.strip_suffix(".weight"))
            .filter(|m| text_mxfp8(&q, m)).count(), 384);
    }
}
