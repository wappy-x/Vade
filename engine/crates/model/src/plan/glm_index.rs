//! Bind index names to roles. The index cannot establish dtypes or shapes.
use super::{description::{TextDescription, TokenMixer}, storage::{WeightStorage, ScaleEncoding}};
use crate::{Result, invalid, safetensors::{Index, Limits}, shapes::*};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdaTensor { Query, Key, Value, Output, QueryConv, KeyConv, ValueConv,
    ForgetDown, ForgetUp, GateDown, GateUp, Beta, Decay, TimeBias, OutputNorm }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentTensor { QueryDown, QueryNorm, QueryUp, KvDown, KvNorm, KvUp, Output }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexTensor { Query, Key, KeyNorm, KeyNormBias, HeadWeights, PoolPosition, PoolGate }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Branch { Attention, Ffn }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MhcTensor { Projection, Base, Scale }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftTensor { EmbeddingHiddenProjection, EmbeddingNorm, HiddenNorm, HeadNorm }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    Embedding, Head, FinalNorm, InputNorm, PostAttentionNorm, Router, RouterBias,
    Dense(Proj), Expert { id: u32, projection: Proj }, Shared(Proj),
    Kda(KdaTensor), Latent(LatentTensor), Index(IndexTensor),
    Mhc { branch: Branch, parameter: MhcTensor }, Draft(DraftTensor), Vision,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsfComponent { Bases, Codes, PageBases, Offsets, Exceptions }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyCsfComponent { Fixed, Exceptions }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorStorage { Weight, BlockScale, GlobalScale, InputScale, Csf(LegacyCsfComponent), Csf32(CsfComponent) }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorBinding {
    pub operation: TensorRole,
    pub role: Role,
    pub component: TensorStorage,
    pub storage: WeightStorage,
}
#[derive(Debug)]
pub struct MappedIndex { pub index: Index, pub tensors: BTreeMap<String, TensorBinding> }
fn require(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(invalid(why)) } }
fn projection(s: &str) -> Option<Proj> { match s { "gate_proj" => Some(Proj::Gate), "up_proj" => Some(Proj::Up),
    "down_proj" => Some(Proj::Down), _ => None } }
fn suffix(name: &str) -> Result<(&str, TensorStorage)> {
    if let Some((module,part)) = name.rsplit_once(".weight_scale.") {
        let legacy = match part { "nvfp4_csf_fixed" => Some(LegacyCsfComponent::Fixed),
            "nvfp4_csf_exceptions" => Some(LegacyCsfComponent::Exceptions), _ => None };
        if let Some(component) = legacy { return Ok((module,TensorStorage::Csf(component))); }
        let component = match part { "csf32_bases" => CsfComponent::Bases, "csf32_codes" => CsfComponent::Codes,
            "csf32_page_bases" => CsfComponent::PageBases, "csf32_offsets" => CsfComponent::Offsets,
            "csf32_exceptions" => CsfComponent::Exceptions, _ => return Err(invalid("unknown scale component")) };
        return Ok((module,TensorStorage::Csf32(component)));
    }
    for (suffix,component) in [(".weight_scale_2",TensorStorage::GlobalScale),(".input_scale",TensorStorage::InputScale),
        (".weight_scale",TensorStorage::BlockScale),(".weight",TensorStorage::Weight)] {
        if let Some(module) = name.strip_suffix(suffix) { return Ok((module,component)); }
    }
    Ok((name,TensorStorage::Weight))
}
impl TextDescription {
    /// Read only index JSON. Do not open a shard, invent a shape, or admit execution.
    pub fn map_index(&self, bytes: &[u8]) -> Result<MappedIndex> {
        let index = Index::parse_mixed(bytes, Limits { header_bytes: HEADER_BYTES,
            tensors: super::super::flash::mixed::MAX_MIXED_TENSORS, rank: MAX_TENSOR_RANK, shards: MAX_SHARDS })?;
        let tensors = index.weight_map.keys().map(|name| self.bind_tensor(name).map(|b| (name.clone(),b))).collect::<Result<_>>()?;
        Ok(MappedIndex { index,tensors })
    }
    pub fn bind_tensor(&self, name: &str) -> Result<TensorBinding> {
        let (module,component) = suffix(name)?;
        if name.starts_with("model.visual.") {
            let store = match component { TensorStorage::Weight => Store::Weight,
                TensorStorage::GlobalScale | TensorStorage::InputScale => Store::Scalar, _ => Store::BlockScale };
            return Ok(TensorBinding { operation: TensorRole::Vision, role: Role { place: Place::Unused,part: Part::Whole,store },
                component,storage: WeightStorage::Plain });
        }
        let (place,operation) = if module == "model.language_model.embed_tokens" { (Place::Model,TensorRole::Embedding) }
            else if module == "lm_head" { (Place::Model,TensorRole::Head) }
            else if module == "model.language_model.norm" { (Place::Model,TensorRole::FinalNorm) }
            else {
                let tail = module.strip_prefix("model.language_model.layers.").ok_or_else(|| invalid(format!("unknown text tensor {name}")))?;
                let (digits,tail) = tail.split_once('.').ok_or_else(|| invalid("missing layer tensor suffix"))?;
                let id = digits.parse::<u32>().map_err(|_| invalid("invalid layer index"))? as usize;
                require(id.to_string() == digits,"noncanonical layer index")?;
                let decoder = self.layers.get(id);
                let draft_id = id.checked_sub(self.layers.len());
                require(decoder.is_some() || draft_id.is_some_and(|i| (i as u64) < self.declared_draft_layers),"layer index outside config")?;
                let place = if decoder.is_some() { Place::Layer(id as u32) } else { Place::Draft { layer: Some(draft_id.unwrap() as u32) } };
                let operation = self.layer_role(tail,decoder.copied(),decoder.is_none())?;
                (place,operation)
            };
        let storage = self.storage.storage(module);
        let valid = match component {
            TensorStorage::Weight => true,
            TensorStorage::BlockScale => matches!(storage,WeightStorage::Mxfp8 | WeightStorage::Fp8Block { .. }
                | WeightStorage::Nvfp4 { encoding: ScaleEncoding::Direct,.. }),
            TensorStorage::GlobalScale => matches!(storage,WeightStorage::Nvfp4 { .. }),
            TensorStorage::InputScale => matches!(storage,WeightStorage::Nvfp4 { activation_scale: true,.. }),
            TensorStorage::Csf(_) => matches!(storage,WeightStorage::Nvfp4 { encoding: ScaleEncoding::Csf,.. }),
            TensorStorage::Csf32(_) => matches!(storage,WeightStorage::Nvfp4 { encoding: ScaleEncoding::Csf32,.. }),
        };
        require(valid,"tensor scale component differs from declared storage")?;
        // Names without .weight occur for stored coefficients. Bias is also
        // explicit. Reject a new spelling instead of silently making an alias.
        if component == TensorStorage::Weight {
            let bare = matches!(operation,TensorRole::Mhc { .. } | TensorRole::Kda(KdaTensor::Decay | KdaTensor::TimeBias)
                | TensorRole::Index(IndexTensor::PoolPosition | IndexTensor::PoolGate) | TensorRole::RouterBias);
            let bias = operation == TensorRole::Index(IndexTensor::KeyNormBias);
            require(if bare { name == module } else if bias { name.ends_with(".bias") } else { name.ends_with(".weight") },"invalid tensor weight suffix")?;
        }
        let part = match operation {
            TensorRole::Embedding => Part::Embedding, TensorRole::Head => Part::Head,
            TensorRole::Dense(p) => Part::Dense(p),TensorRole::Shared(p) => Part::Shared(p),
            TensorRole::Expert { id,projection } => Part::Expert { id: Some(id),proj: projection },
            TensorRole::Index(_) => Part::Index,
            _ => Part::Whole,
        };
        let store = match component { TensorStorage::Weight => Store::Weight,
            TensorStorage::GlobalScale | TensorStorage::InputScale => Store::Scalar, _ => Store::BlockScale };
        Ok(TensorBinding { operation,role: Role { place,part,store },component,storage })
    }
    fn layer_role(&self, tail: &str, layer: Option<super::description::DecoderDescription>, draft: bool) -> Result<TensorRole> {
        let simple = match tail {
            "input_layernorm" => Some(TensorRole::InputNorm), "post_attention_layernorm" => Some(TensorRole::PostAttentionNorm),
            "mlp.gate" => Some(TensorRole::Router), "mlp.gate.e_score_correction_bias" => Some(TensorRole::RouterBias),
            "eh_proj" if draft => Some(TensorRole::Draft(DraftTensor::EmbeddingHiddenProjection)),
            "enorm" if draft => Some(TensorRole::Draft(DraftTensor::EmbeddingNorm)),
            "hnorm" if draft => Some(TensorRole::Draft(DraftTensor::HiddenNorm)),
            "shared_head.norm" if draft => Some(TensorRole::Draft(DraftTensor::HeadNorm)), _ => None,
        };
        if let Some(role) = simple {
            if matches!(role,TensorRole::Router | TensorRole::RouterBias) {
                require(layer.is_none_or(|l| matches!(l.ffn,Ffn::Experts(_))),"router outside an expert layer")?;
            }
            return Ok(role);
        }
        for (prefix,branch) in [("hc_attn_",Branch::Attention),("hc_ffn_",Branch::Ffn)] {
            if let Some(s) = tail.strip_prefix(prefix) {
                require(!draft,"draft residual semantics are unverified")?;
                let parameter = match s { "fn" => MhcTensor::Projection,"base" => MhcTensor::Base,"scale" => MhcTensor::Scale,
                    _ => return Err(invalid("unknown mHC parameter")) };
                return Ok(TensorRole::Mhc { branch,parameter });
            }
        }
        if let Some(t) = tail.strip_prefix("mlp.") {
            if let Some(proj) = projection(t) {
                require(layer.is_some_and(|l| matches!(l.ffn,Ffn::Dense { .. })),"dense tensor outside a dense layer")?;
                return Ok(TensorRole::Dense(proj));
            }
            if let Some(t) = t.strip_prefix("shared_experts.") {
                require(layer.is_none_or(|l| matches!(l.ffn,Ffn::Experts(e) if e.shared_width > 0)),"shared tensor outside a shared expert layer")?;
                return projection(t).map(TensorRole::Shared).ok_or_else(|| invalid("unknown shared projection"));
            }
            if let Some(t) = t.strip_prefix("experts.") {
                let (id,p) = t.split_once('.').ok_or_else(|| invalid("missing expert projection"))?;
                let parsed = id.parse::<u32>().map_err(|_| invalid("invalid expert index"))?;
                require(parsed.to_string() == id,"noncanonical expert index")?;
                let count = layer.and_then(|l| if let Ffn::Experts(e) = l.ffn { Some(e.count) } else { None })
                    .or_else(|| self.layers.iter().find_map(|l| if let Ffn::Experts(e) = l.ffn { Some(e.count) } else { None }));
                require((draft || layer.is_some_and(|l| matches!(l.ffn,Ffn::Experts(_))))
                    && count.is_some_and(|n| (parsed as u64) < n),"expert index outside config")?;
                return projection(p).map(|projection| TensorRole::Expert { id: parsed,projection }).ok_or_else(|| invalid("unknown expert projection"));
            }
        }
        let t = tail.strip_prefix("self_attn.").ok_or_else(|| invalid(format!("unknown layer tensor {tail}")))?;
        let kda = layer.is_some_and(|l| matches!(l.mixer,TokenMixer::Kda(_)));
        if kda {
            let role = match t { "q_proj" => KdaTensor::Query,"k_proj" => KdaTensor::Key,"v_proj" => KdaTensor::Value,
                "o_proj" => KdaTensor::Output,"q_conv1d" => KdaTensor::QueryConv,"k_conv1d" => KdaTensor::KeyConv,"v_conv1d" => KdaTensor::ValueConv,
                "f_a_proj" => KdaTensor::ForgetDown,"f_b_proj" => KdaTensor::ForgetUp,"g_a_proj" => KdaTensor::GateDown,"g_b_proj" => KdaTensor::GateUp,
                "b_proj" => KdaTensor::Beta,"A_log" => KdaTensor::Decay,"dt_bias" => KdaTensor::TimeBias,"o_norm" => KdaTensor::OutputNorm,
                _ => return Err(invalid(format!("unknown KDA tensor {t}"))) };
            return Ok(TensorRole::Kda(role));
        }
        // Appended attention names are inventoried only. This does not derive
        // the draft's forward pass or reuse the Qwen draft wrapper.
        let role = match t { "q_a_proj" => Some(LatentTensor::QueryDown),"q_a_layernorm" => Some(LatentTensor::QueryNorm),
            "q_b_proj" => Some(LatentTensor::QueryUp),"kv_a_proj_with_mqa" => Some(LatentTensor::KvDown),"kv_a_layernorm" => Some(LatentTensor::KvNorm),
            "kv_b_proj" => Some(LatentTensor::KvUp),"o_proj" => Some(LatentTensor::Output), _ => None };
        if let Some(role) = role { return Ok(TensorRole::Latent(role)); }
        let role = match t { "indexer.wq_b" => IndexTensor::Query,"indexer.wk" => IndexTensor::Key,
            "indexer.k_norm" => IndexTensor::KeyNorm,"indexer.k_norm.bias" => IndexTensor::KeyNormBias,
            "indexer.weights_proj" => IndexTensor::HeadWeights,"indexer.index_kpool_compress_ape" => IndexTensor::PoolPosition,
            "indexer.index_kpool_compress_gate" => IndexTensor::PoolGate, _ => return Err(invalid(format!("unknown latent attention tensor {t}"))) };
        Ok(TensorRole::Index(role))
    }
}
