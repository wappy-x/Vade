//! Translate the GLM text schema into operation data. This is host metadata only.
use super::{description::*, storage::StorageMap, PlanError, Result, invalid, mul};
use crate::{json::Value, shapes::*};
use crate::host_ops::router::{Router, RouterKind};
use std::collections::{BTreeMap, BTreeSet};
type Object = BTreeMap<String, Value>;
fn get<'a>(o: &'a Object, key: &'static str) -> Result<&'a Value> { o.get(key).ok_or(PlanError::Missing { field: key }) }
fn object<'a>(v: &'a Value, key: &'static str) -> Result<&'a Object> {
    v.object().map_err(|_| PlanError::WrongType { field: key, expected: "object" })
}
fn array<'a>(o: &'a Object, key: &'static str) -> Result<&'a [Value]> {
    get(o,key)?.array().map_err(|_| PlanError::WrongType { field: key, expected: "array" })
}
fn string<'a>(v: &'a Value, key: &'static str) -> Result<&'a str> {
    v.string().map_err(|_| PlanError::WrongType { field: key, expected: "string" })
}
fn is(o: &Object, key: &'static str, value: &str) -> Result<()> {
    check(string(get(o,key)?,key)? == value, key, "unsupported value")
}
fn integer(v: &Value, key: &'static str) -> Result<u64> {
    v.u64().map_err(|_| PlanError::WrongType { field: key, expected: "unsigned integer" })
}
fn n(o: &Object, key: &'static str, max: u64, zero: bool) -> Result<u64> {
    let n = integer(get(o,key)?,key)?;
    check(n <= max && (zero || n > 0), key, "dimension is outside its limit")?; Ok(n)
}
fn number(o: &Object, key: &'static str) -> Result<f64> {
    get(o,key)?.f64().map_err(|_| PlanError::WrongType { field: key, expected: "finite number" })
}
fn boolean(o: &Object, key: &'static str) -> Result<bool> {
    get(o,key)?.boolean().map_err(|_| PlanError::WrongType { field: key, expected: "boolean" })
}
fn check(ok: bool, field: &'static str, reason: &'static str) -> Result<()> { if ok { Ok(()) } else { Err(invalid(field,reason)) } }
fn flags(o: &Object, values: &[(&'static str,bool)]) -> Result<()> {
    for &(key,value) in values { check(boolean(o,key)? == value,key,"unsupported value")?; } Ok(())
}
fn indices(o: &Object, key: &'static str, expected: &[usize]) -> Result<()> {
    let a = array(o,key)?.iter().map(|v| integer(v,key)).collect::<Result<Vec<_>>>()?;
    let set = a.iter().copied().collect::<BTreeSet<_>>();
    check(set.len() == a.len() && set == expected.iter().map(|&i| i as u64).collect(),key,"does not match layer_types")
}
#[allow(clippy::manual_is_multiple_of)] // A zero divisor must still panic in `%`.
pub(super) fn parse(doc: &Value) -> Result<TextDescription> {
    let root = object(doc,"config")?;
    let kind = string(get(root,"model_type")?,"model_type")?;
    let o = if kind == "glm5_next_text" { root } else { object(get(root,"text_config")?,"text_config")? };
    is(o,"model_type","glm5_next_text")?;
    is(o,"dtype","bfloat16")?; is(o,"hidden_act","silu")?;
    is(o,"moe_router_dtype","float32")?;
    let router_kind = RouterKind::from_selectors(string(get(o,"scoring_func")?,"scoring_func")?,
        string(get(o,"topk_method")?,"topk_method")?).ok_or_else(|| invalid("scoring_func","unsupported routing selectors"))?;
    flags(o,&[("attention_bias",false),("use_cache",true),("mhc",true),("mla_use_nope",true)])?;
    check(number(o,"attention_dropout")? == 0.0,"attention_dropout","must be zero for inference")?;
    let tied_embeddings = boolean(o,"tie_word_embeddings")?;
    if !std::ptr::eq(root,o) && root.contains_key("tie_word_embeddings") {
        check(boolean(root,"tie_word_embeddings")? == tied_embeddings,"tie_word_embeddings","root and text values differ")?;
    }
    let hidden = n(o,"hidden_size",MAX_HIDDEN,false)?;
    let vocab = n(o,"vocab_size",MAX_VOCAB,false)?;
    let count = n(o,"num_hidden_layers",MAX_LAYERS,false)?;
    let epsilon = number(o,"rms_norm_eps")?;
    check(epsilon > 0.0,"rms_norm_eps","must be positive")?;
    let eos = get(o,"eos_token_id")?;
    let eos_tokens = match eos { Value::Array(a) => a.iter().map(|v| integer(v,"eos_token_id")).collect::<Result<Vec<_>>>()?,
        _ => vec![integer(eos,"eos_token_id")?] };
    check(!eos_tokens.is_empty() && eos_tokens.iter().all(|&v| v < vocab)
        && eos_tokens.iter().collect::<BTreeSet<_>>().len() == eos_tokens.len(),"eos_token_id","empty, repeated or outside vocabulary")?;
    let pad_token = n(o,"pad_token_id",vocab-1,true)?;
    let linear = object(get(o,"linear_attn_config")?,"linear_attn_config")?;
    let kda = KdaDescription { heads: n(linear,"num_heads",MAX_HEADS,false)?,
        head_dim: n(linear,"head_dim",MAX_HEAD_DIM,false)?, conv: n(linear,"short_conv_kernel_size",MAX_HEADS,false)?,
        gate_lower_bound: number(linear,"gate_lower_bound")? };
    check(kda.gate_lower_bound < 0.0,"gate_lower_bound","must be negative")?;
    check(mul(kda.heads,kda.head_dim)? <= MAX_WIDTH,"linear_attn_config","channel width exceeds limit")?;
    let indexer = PoolIndexerDescription { heads: n(o,"index_n_heads",MAX_HEADS,false)?, dim: n(o,"index_head_dim",MAX_HEAD_DIM,false)?,
        token_budget: n(o,"index_topk",MAX_POSITIONS,false)?, pool: n(o,"index_kpool",MAX_POSITIONS,false)?,
        learned_pool: boolean(o,"index_kpool_compress")?, select_tail: boolean(o,"index_kpool_always_select_tail")?,
        share_for_draft_iteration: boolean(o,"index_share_for_mtp_iteration")? };
    check(indexer.learned_pool && indexer.token_budget % indexer.pool == 0,"index_kpool","unsupported pool or indivisible token budget")?;
    let latent = LatentDescription { heads: n(o,"num_attention_heads",MAX_HEADS,false)?, query_rank: n(o,"q_lora_rank",MAX_WIDTH,false)?,
        kv_rank: n(o,"kv_lora_rank",MAX_WIDTH,false)?, key_dim: n(o,"qk_nope_head_dim",MAX_HEAD_DIM,false)?,
        value_dim: n(o,"v_head_dim",MAX_HEAD_DIM,false)?, rotary_dim: n(o,"qk_rope_head_dim",MAX_HEAD_DIM,true)?, indexer };
    check(latent.rotary_dim == 0 && !o.contains_key("rope_parameters"),"qk_rope_head_dim","NoPE requires no rotary dimensions or parameters")?;
    check(n(o,"qk_head_dim",MAX_HEAD_DIM,false)? == latent.key_dim,"qk_head_dim","differs from nonrotary head dimension")?;
    // indexer_rope_interleave has no effect with no rotary channels.
    boolean(o,"indexer_rope_interleave")?;
    let experts = n(o,"n_routed_experts",MAX_EXPERTS,false)?;
    let width = n(o,"moe_intermediate_size",MAX_WIDTH,false)?;
    let shared_width = mul(n(o,"n_shared_experts",MAX_EXPERTS,true)?,width)?;
    check(shared_width <= MAX_WIDTH,"n_shared_experts","combined shared width exceeds limit")?;
    let expert = Ffn::Experts(ExpertShape { count: experts, selected: n(o,"num_experts_per_tok",experts,false)?,
        width, shared_width, storage: ExpertStorage::PerExpert });
    let dense_width = n(o,"intermediate_size",MAX_WIDTH,false)?;
    let types = array(o,"layer_types")?;
    let ffns = array(o,"mlp_layer_types")?;
    let indexers = array(o,"indexer_types")?;
    check(types.len() as u64 == count && ffns.len() == types.len() && indexers.len() == types.len(),"layer_types","schedule length differs from layer count")?;
    let (mut kda_ids,mut full_ids,mut layers) = (Vec::new(),Vec::new(),Vec::new());
    for (i,kind) in types.iter().enumerate() {
        check(string(&indexers[i],"indexer_types")? == "full","indexer_types","shared indexer schedule is unsupported")?;
        let mixer = match string(kind,"layer_types")? {
            "linear_attention" => { kda_ids.push(i); TokenMixer::Kda(kda) },
            "deepseek_sparse_attention" => { full_ids.push(i); TokenMixer::LatentAttention(latent) },
            _ => return Err(invalid("layer_types","unknown attention operation")),
        };
        let ffn = match string(&ffns[i],"mlp_layer_types")? { "dense" => Ffn::Dense { width: dense_width }, "sparse" => expert,
            _ => return Err(invalid("mlp_layer_types","unknown feed-forward operation")) };
        layers.push(DecoderDescription { mixer,ffn });
    }
    // The explicit FFN schedule is authoritative. first_k_dense_replace is a
    // source default and does not override a supplied mlp_layer_types list.
    if o.contains_key("first_k_dense_replace") { n(o,"first_k_dense_replace",count,true)?; }
    indices(linear,"kda_layers",&kda_ids)?; indices(linear,"full_attn_layers",&full_ids)?;
    let residual = SinkhornDescription { streams: n(o,"hc_mult",MAX_HEADS,false)?,
        iterations: n(o,"hc_sinkhorn_iters",MAX_SINKHORN_ITERS,false)?, epsilon: number(o,"hc_eps")? };
    check(residual.epsilon > 0.0,"hc_eps","must be positive")?;
    mul(residual.streams,hidden)?;
    let groups = n(o,"n_group",experts,false)?;
    check(experts % groups == 0,"n_group","expert count does not divide into groups")?;
    let routing = RoutingDescription { kind: router_kind,
        normalize_selected: boolean(o,"norm_topk_prob")?, scale: number(o,"routed_scaling_factor")?,
        groups, selected_groups: n(o,"topk_group",groups,false)?, shared_gate: false };
    check(routing.scale > 0.0,"routed_scaling_factor","must be positive")?;
    let Ffn::Experts(expert_shape) = expert else { unreachable!() };
    Router::from_description(&routing,expert_shape).map_err(|e| super::read_error("routing",e))?;
    let swiglu_limit = number(o,"swiglu_limit")?;
    crate::swiglu::SwiGlu::new(Some(swiglu_limit))
        .map_err(|e| super::read_error("swiglu_limit",e))?;
    let q = root.get("quantization_config").or_else(|| o.get("quantization_config"))
        .ok_or(PlanError::Missing { field: "quantization_config" })?;
    let storage = StorageMap::parse(q).map_err(|e| super::read_error("quantization_config",e))?;
    Ok(TextDescription { hidden,vocab,max_positions: n(o,"max_position_embeddings",MAX_POSITIONS,false)?,epsilon,
        eos_tokens,pad_token,tied_embeddings,rope: None,layers,residual,routing,swiglu_limit,storage,
        declared_draft_layers: n(o,"num_nextn_predict_layers",MAX_LAYERS,true)? })
}
