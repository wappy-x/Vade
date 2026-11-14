//! Compatibility view for existing shared bindings; config is never parsed a second time.
use crate::{
    flash::LayerKind,
    qwen3_5::{Checkpoint, Config},
};
use std::path::Path;
use crate::{plan::{Format, ModelPlan}, shapes::{Ffn, Mixer, Part, Place, Store}, safetensors::Dtype};
type Result<T> = std::result::Result<T, String>;

/// Resident per-expert FP8 belongs to the typed loader, after header reconciliation.
/// Flash stream residuals retain their own loader and execution path.
pub fn uses_typed_loader(p: &ModelPlan) -> bool {
    matches!(p.format, Format::Fp8 { .. })
        && p.shape.residual == crate::shapes::Residual::Single
        && !p.shape.layers.is_empty()
        && p.shape.layers.iter().all(|l| matches!(l.ffn, Ffn::Experts(e)
            if e.storage == crate::shapes::ExpertStorage::PerExpert))
}

pub fn inspect(root: &Path, mtp: bool) -> Result<(ModelPlan, Checkpoint)> {
    let (plan, mut index, shards) = ModelPlan::inspect_parts(root).map_err(|e| e.to_string())?;
    let config = config(&plan)?;
    if !uses_typed_loader(&plan) {
        if config.experts != 0 && config.fp8 { return Err("unsupported expert storage: FP8 requires PerExpert; BF16 fallback is disabled".into()); }
        if plan.shape.layers.iter().any(|l| matches!(l.ffn, Ffn::Experts(e) if e.storage != crate::shapes::ExpertStorage::Stacked)) { return Err("unsupported expert storage: PerExpert requires FP8".into()); }
    }
    if mtp { mtp_weight_bytes(&plan)?; }
    let text_bytes = plan.shape.tensors.iter().filter(|t| matches!(t.role.place, Place::Model | Place::Layer(_)))
        .try_fold(0u64, |sum, t| { let bytes = t.shape.iter().try_fold(t.dtype.bytes(), |n, v| n.checked_mul(*v))?; sum.checked_add(bytes) }).ok_or("hybrid weight byte overflow")?;
    if index.total_size == 0 { index.total_size = shards.values().flat_map(|s| s.tensors.values()).map(crate::safetensors::Tensor::byte_len).sum(); }
    Ok((plan, Checkpoint { config, index, shards, text_bytes }))
}
/// Native typed MTP contract, also checked before HTTP memory admission.
pub fn validate_typed_mtp(p: &ModelPlan) -> Result<()> {
    if p.shape.draft.len() != 1 || !matches!(p.shape.draft[0].mixer, Mixer::Attention(a) if a.index.is_none()) {
        return Err("MTP requires one non-indexed attention block".into());
    }
    if p.shape.tensors.iter().any(|t| matches!(t.role.place, Place::Draft { .. }) && matches!(t.role.part, Part::Embedding | Part::Head)) {
        return Err("MTP must share the target embedding and head IDs".into());
    }
    let fc = p.shape.tensors.iter().find(|t| t.name == "mtp.fc.weight").ok_or("MTP concat projection absent")?;
    if fc.dtype != Dtype::Bf16 || fc.shape != [p.shape.hidden, p.shape.hidden.checked_mul(2).ok_or("MTP concat overflow")?]
        || fc.role.place != (Place::Draft { layer: None }) || fc.role.part != Part::Whole || fc.role.store != Store::Weight {
        return Err("MTP concat requires BF16 [H,2H]".into());
    }
    Ok(())
}
/// Stored draft payload, including FP8 block scales, from the reconciled headers.
/// Legacy MoE stays refused; its dense shape/byte formula cannot describe experts.
pub fn mtp_weight_bytes(p: &ModelPlan) -> Result<u64> {
    if uses_typed_loader(p) { validate_typed_mtp(p)?; }
    else { config(p)?.mtp_shapes().map_err(|e| e.to_string())?; }
    p.shape.tensors.iter().filter(|t| matches!(t.role.place, Place::Draft { .. }))
        .try_fold(0u64, |sum,t| t.shape.iter().try_fold(t.dtype.bytes(), |n,v| n.checked_mul(*v))
            .and_then(|bytes| sum.checked_add(bytes))).ok_or("MTP weight byte overflow".into())
}
/// Temporary geometry view, derived solely from the reconciled plan. No checkpoint names.
pub fn config(p: &ModelPlan) -> Result<Config> {
    p.hybrid_projections().map_err(|e| e.to_string())?;
    let s = &p.shape;
    let linear = s.layers.iter().find_map(|l| if let Mixer::Linear(v) = l.mixer { Some(v) } else { None }).ok_or("hybrid requires linear layers")?;
    let attention = s.layers.iter().find_map(|l| if let Mixer::Attention(v) = l.mixer { Some(v) } else { None }).ok_or("hybrid requires attention layers")?;
    let ffn = s.layers.first().ok_or("hybrid requires layers")?.ffn;
    if s.layers.iter().chain(&s.draft).any(|l| l.ffn != ffn || match l.mixer { Mixer::Linear(v) => v != linear, Mixer::Attention(v) => v != attention }) {
        return Err("existing hybrid bindings require uniform mixer and FFN geometry".into());
    }
    let (intermediate, experts, selected_experts, shared_intermediate) = match ffn {
        Ffn::Dense { width } => (width, 0, 0, 0), Ffn::Experts(e) => (e.width, e.count, e.selected, e.shared_width),
    };
    let tied_embeddings = !s.tensors.iter().any(|t| t.role.part == crate::shapes::Part::Head && t.role.store == Store::Weight);
    Ok(Config { hidden: s.hidden, intermediate, experts, selected_experts, shared_intermediate,
        layers: s.layers.iter().map(|l| match l.mixer { Mixer::Linear(_) => LayerKind::DeltaNet, Mixer::Attention(_) => LayerKind::SparseAttention }).collect(),
        query_heads: attention.query_heads, kv_heads: attention.kv_heads, head_dim: attention.head_dim,
        linear_key_heads: linear.key_heads, linear_value_heads: linear.value_heads, linear_key_dim: linear.key_dim,
        linear_value_dim: linear.value_dim, linear_conv: linear.conv, vocab: s.vocab, max_positions: s.max_positions,
        bos_token: p.bos_token, eos_token: p.eos_token, epsilon: p.epsilon, rope_theta: p.rope.theta,
        rotary_dim: p.rope.rotary_dim, mrope_sections: p.rope.sections.clone(), mtp_layers: s.draft.len() as u64,
        mtp_dedicated_embeddings: false, tied_embeddings, fp8: matches!(p.format, Format::Fp8 { .. }),
        unquantized: match &p.format { Format::Bf16 => Vec::new(), Format::Fp8 { unquantized, .. } => unquantized.clone(),
            Format::Mixed(_) => return Err("mixed storage requires the Flash loader".into()) } })
}
