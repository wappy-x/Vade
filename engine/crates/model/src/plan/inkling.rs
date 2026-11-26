//! Translate the looped text schema into operation data. This is host metadata only.
//! The schema is the released `config.json` of `model_type` `inkling2_loop`. An unknown
//! field or value is refused, because this file cannot know what it changes in the math.
//!
//! Reference: transformers v5.24.0, `models/inkling2_loop`. This file follows these lines
//! and holds its own code.
//! - `configuration_inkling2_loop.py`: 116, 120 and 136 (fields the released config leaves
//!   out), 144-149 (`loop_config`), 155-169 (loop range rules).
//! - `modeling_inkling2_loop.py`: 546-572 and 639-665 (the two attention kinds), 828-853
//!   (block mix), 861-862 and 1005-1006 (block boundary, no attention mix in layer 0),
//!   1293-1298 (layer kinds), 1350-1440 (visit order), 1405 (state slot of a later pass).
use super::{description::*, PlanError, Result, invalid};
use crate::{json::Value, shapes::*};
use crate::host_ops::router::{Router, RouterKind};
use std::collections::BTreeMap;
type Object = BTreeMap<String, Value>;

const ROOT_TYPE: &str = "inkling2_loop";
const TEXT_TYPE: &str = "inkling2_loop_text";
/// The released config has no `scoring_func`. The reference fills this value.
/// Reference: `configuration_inkling2_loop.py:136`.
const SCORE: &str = "sigmoid";
/// The released config has no `hidden_act`. Reference: `configuration_inkling2_loop.py:120`.
const ACTIVATION: &str = "silu";
/// Fixed taps: the last stored tap meets the current token. Not yet checked by a run.
/// Reference: `modeling_inkling2_loop.py:293-299` and `314-322`.
pub const FIXED_TAP_ORDER: TapOrder = TapOrder::OldestFirst;
/// Token-made taps: stored row 0 meets the current token. Not yet checked by a run.
/// Reference: `modeling_inkling2_loop.py:410`.
pub const TOKEN_TAP_ORDER: TapOrder = TapOrder::NewestFirst;
/// Token-made taps: index = head * kernel + lag. Not yet checked by a run.
/// Reference: `modeling_inkling2_loop.py:431`.
pub const TOKEN_TAP_INDEX: TapIndex = TapIndex::HeadThenLag;
/// A bound for the parser. It is not a size of any checkpoint.
const MAX_PASSES: u64 = 64;
const TEXT_KEYS: &[&str] = &["attn_res_normalize_values","d_rel","eos_token_id","final_logit_softcapping","head_dim",
    "hidden_act","hidden_size","intermediate_size","layer_types","local_layer_ids","log_scaling_alpha","log_scaling_n_floor",
    "loop_config","mlp_gate_softcap","mlp_up_softcap","model_max_length","model_type","n_layers_per_attn_res_block",
    "n_routed_experts","n_shared_experts","num_attention_heads","num_experts_per_tok","num_hidden_layers",
    "num_key_value_heads","o_bias","pad_token_id","q_bias","qk_l2_clip_threshold","rel_extent","rel_l2_clip_threshold",
    "rms_norm_eps","route_scale","sconv_kernel_size","scoring_func","sliding_window_size","swa_head_dim",
    "swa_num_attention_heads","swa_num_key_value_heads","torch_dtype","unpadded_vocab_size","use_gate_bias",
    "use_value_norm_no_gamma","vocab_size"];
const LOOP_KEYS: &[&str] = &["core_end","core_start","gate_rank","is_varloop","num_passes"];
const STORAGE_KEYS: &[&str] = &["quant_algo","quant_method"];

fn get<'a>(o: &'a Object, key: &'static str) -> Result<&'a Value> { o.get(key).ok_or(PlanError::Missing { field: key }) }
fn object<'a>(v: &'a Value, key: &'static str) -> Result<&'a Object> {
    v.object().map_err(|_| PlanError::WrongType { field: key, expected: "object" })
}
fn string<'a>(v: &'a Value, key: &'static str) -> Result<&'a str> {
    v.string().map_err(|_| PlanError::WrongType { field: key, expected: "string" })
}
fn integer(v: &Value, key: &'static str) -> Result<u64> {
    v.u64().map_err(|_| PlanError::WrongType { field: key, expected: "unsigned integer" })
}
fn check(ok: bool, field: &'static str, reason: &'static str) -> Result<()> { if ok { Ok(()) } else { Err(invalid(field,reason)) } }
fn n(o: &Object, key: &'static str, max: u64, zero: bool) -> Result<u64> {
    let n = integer(get(o,key)?,key)?;
    check(n <= max && (zero || n > 0), key, "dimension is outside its limit")?; Ok(n)
}
fn positive(o: &Object, key: &'static str) -> Result<f64> {
    let v = get(o,key)?.f64().map_err(|_| PlanError::WrongType { field: key, expected: "finite number" })?;
    check(v > 0.0, key, "must be positive")?; Ok(v)
}
fn is(o: &Object, key: &'static str, value: &str) -> Result<()> {
    let actual = string(get(o,key)?,key)?;
    if actual == value { Ok(()) } else { Err(PlanError::Unsupported { field: key, value: actual.into() }) }
}
/// A field the reference fills when the config leaves it out. A stated value must be that value.
fn is_or_absent(o: &Object, key: &'static str, value: &str) -> Result<()> { if o.contains_key(key) { is(o,key,value) } else { Ok(()) } }
fn flags(o: &Object, values: &[(&'static str,bool)]) -> Result<()> {
    for &(key,value) in values {
        let actual = get(o,key)?.boolean().map_err(|_| PlanError::WrongType { field: key, expected: "boolean" })?;
        if actual != value { return Err(PlanError::Unsupported { field: key, value: actual.to_string() }); }
    }
    Ok(())
}
fn known(o: &Object, field: &'static str, keys: &[&str]) -> Result<()> {
    match o.keys().find(|k| !keys.contains(&k.as_str())) {
        Some(key) => Err(PlanError::Unsupported { field, value: format!("unknown field {key}") }),
        None => Ok(()),
    }
}
fn rising(o: &Object, key: &'static str, limit: u64) -> Result<Vec<u64>> {
    let values = get(o,key)?.array().map_err(|_| PlanError::WrongType { field: key, expected: "array" })?
        .iter().map(|v| integer(v,key)).collect::<Result<Vec<_>>>()?;
    check(values.windows(2).all(|w| w[0] < w[1]) && values.iter().all(|&v| v < limit), key, "not rising or outside its limit")?;
    Ok(values)
}
fn attention(a: SinkAttentionDescription, field: &'static str) -> Result<SinkAttentionDescription> {
    check(a.query_heads.is_multiple_of(a.kv_heads), field, "key heads do not divide query heads")?;
    check(super::mul(a.query_heads,a.head_dim)? <= MAX_WIDTH, field, "channel width exceeds limit")?;
    Ok(a)
}
/// Run order for one token and the state slot of each visit.
fn visits(layers: u64, core: LoopRange) -> Vec<Visit> {
    let once = |layer| Visit { layer, pass: 0, slot: layer, messages: false };
    let span = core.end - core.start;
    let mut out = (0..core.start).map(once).collect::<Vec<_>>();
    for pass in 1..=core.passes {
        // The first pass uses the slot of the layer. Each later pass has its own slots after the layers.
        out.extend((core.start..core.end).map(|layer| Visit { layer, pass, messages: pass > 1,
            slot: if pass == 1 { layer } else { layers + (pass - 2) * span + (layer - core.start) } }));
    }
    out.extend((core.end..layers).map(once));
    out
}

pub(super) fn parse(doc: &Value) -> Result<LoopTextDescription> {
    let root = object(doc,"config")?;
    let nested = string(get(root,"model_type")?,"model_type")? == ROOT_TYPE;
    let o = if nested { object(get(root,"text_config")?,"text_config")? } else { root };
    is(o,"model_type",TEXT_TYPE)?;
    known(o,"text_config",TEXT_KEYS)?;
    is(o,"torch_dtype","bfloat16")?;
    if nested { is_or_absent(root,"dtype","bfloat16")?; }
    is_or_absent(o,"scoring_func",SCORE)?; is_or_absent(o,"hidden_act",ACTIVATION)?;
    flags(o,&[("q_bias",false),("o_bias",false),("use_value_norm_no_gamma",true),("attn_res_normalize_values",true),("use_gate_bias",true)])?;
    let hidden = n(o,"hidden_size",MAX_HIDDEN,false)?;
    let vocab = n(o,"vocab_size",MAX_VOCAB,false)?;
    let head_rows = n(o,"unpadded_vocab_size",vocab,false)?;
    let count = n(o,"num_hidden_layers",MAX_LAYERS,false)?;
    let epsilon = positive(o,"rms_norm_eps")?;
    let eos = match (o.get("eos_token_id"), nested.then(|| root.get("eos_token_id")).flatten()) {
        (Some(a),Some(b)) => { check(a == b,"eos_token_id","root and text values differ")?; a },
        (Some(v),None) | (None,Some(v)) => v,
        (None,None) => return Err(PlanError::Missing { field: "eos_token_id" }),
    };
    let eos_tokens = match eos { Value::Array(a) => a.iter().map(|v| integer(v,"eos_token_id")).collect::<Result<Vec<_>>>()?,
        _ => vec![integer(eos,"eos_token_id")?] };
    // A stop token must be a row that the head can write.
    check(!eos_tokens.is_empty() && eos_tokens.iter().all(|&v| v < head_rows)
        && eos_tokens.iter().enumerate().all(|(i,v)| !eos_tokens[..i].contains(v)),"eos_token_id","empty, repeated or outside the head rows")?;
    let pad_token = match o.get("pad_token_id") { None | Some(Value::Null) => None, Some(_) => Some(n(o,"pad_token_id",vocab-1,true)?) };

    let kernel = n(o,"sconv_kernel_size",MAX_HEADS,false)?;
    let window = n(o,"sliding_window_size",MAX_POSITIONS,false)?;
    let (dim,clip,query_key_clip) = (n(o,"d_rel",MAX_HEAD_DIM,false)?,positive(o,"rel_l2_clip_threshold")?,positive(o,"qk_l2_clip_threshold")?);
    let scale = PositionScale { floor: n(o,"log_scaling_n_floor",MAX_POSITIONS,false)?, alpha: positive(o,"log_scaling_alpha")? };
    let full = attention(SinkAttentionDescription { query_heads: n(o,"num_attention_heads",MAX_HEADS,false)?,
        kv_heads: n(o,"num_key_value_heads",MAX_HEADS,false)?, head_dim: n(o,"head_dim",MAX_HEAD_DIM,false)?, window: None,
        relative: RelativeBiasDescription { dim, extent: n(o,"rel_extent",MAX_POSITIONS,false)?, clip }, query_key_clip, position_scale: Some(scale),
        key_value_conv: ShortConvDescription { kernel, taps: ConvTaps::FixedAndToken { order: TOKEN_TAP_ORDER, index: TOKEN_TAP_INDEX } },
        separate_inputs: true },"num_attention_heads")?;
    // In a window layer the relative term reaches the whole window.
    let local = attention(SinkAttentionDescription { query_heads: n(o,"swa_num_attention_heads",MAX_HEADS,false)?,
        kv_heads: n(o,"swa_num_key_value_heads",MAX_HEADS,false)?, head_dim: n(o,"swa_head_dim",MAX_HEAD_DIM,false)?, window: Some(window),
        relative: RelativeBiasDescription { dim, extent: window, clip }, query_key_clip, position_scale: None,
        key_value_conv: ShortConvDescription { kernel, taps: ConvTaps::Fixed { order: FIXED_TAP_ORDER } },
        separate_inputs: false },"swa_num_attention_heads")?;

    let experts = n(o,"n_routed_experts",MAX_EXPERTS,false)?;
    check(n(o,"n_shared_experts",MAX_EXPERTS,true)? == 0,"n_shared_experts","this schema has no shared expert")?;
    let expert = ExpertShape { count: experts, selected: n(o,"num_experts_per_tok",experts,false)?,
        width: n(o,"intermediate_size",MAX_WIDTH,false)?, shared_width: 0, storage: ExpertStorage::Stacked };
    // Sigmoid scores, the bias takes part in the pick only, the picked scores are divided by their sum.
    let routing = RoutingDescription { kind: RouterKind::CorrectedSigmoid, normalize_selected: true,
        scale: positive(o,"route_scale")?, groups: 1, selected_groups: 1, shared_gate: false };
    Router::from_description(&routing,expert).map_err(|e| super::read_error("routing",e))?;
    let caps = SoftCaps { gate: positive(o,"mlp_gate_softcap")?, up: positive(o,"mlp_up_softcap")?, logits: positive(o,"final_logit_softcapping")? };

    let block_layers = n(o,"n_layers_per_attn_res_block",count,false)?;
    let l = object(get(o,"loop_config")?,"loop_config")?;
    known(l,"loop_config",LOOP_KEYS)?;
    flags(l,&[("is_varloop",false)])?;
    let core = LoopRange { start: n(l,"core_start",count,false)?, end: n(l,"core_end",count,false)?,
        passes: n(l,"num_passes",MAX_PASSES,false)?, gate_rank: n(l,"gate_rank",MAX_WIDTH,false)? };
    check(core.start < core.end,"loop_config","the loop range is empty")?;
    check(core.start.is_multiple_of(block_layers) && core.end.is_multiple_of(block_layers),"loop_config","the loop range does not start and end on a block")?;

    let window_layers = rising(o,"local_layer_ids",count)?;
    let layers = (0..count).map(|i| {
        let attention = if window_layers.contains(&i) { local } else { full };
        LoopLayerDescription { attention, ffn: Ffn::Experts(expert), boundary: i.is_multiple_of(block_layers),
            attention_queries: if i == 0 { 0 } else if attention.separate_inputs { 3 } else { 1 },
            message_gates: core.start <= i && i < core.end }
    }).collect::<Vec<_>>();
    if let Some(types) = o.get("layer_types") {
        let types = types.array().map_err(|_| PlanError::WrongType { field: "layer_types", expected: "array" })?;
        check(types.len() == layers.len(),"layer_types","length differs from num_hidden_layers")?;
        for (kind,layer) in types.iter().zip(&layers) {
            let stated = match string(kind,"layer_types")? { "hybrid" => false, "hybrid_sliding" => true,
                other => return Err(PlanError::Unsupported { field: "layer_types", value: other.into() }) };
            check(stated == layer.attention.window.is_some(),"layer_types","differs from local_layer_ids")?;
        }
    }
    let visits = visits(count,core);

    let declared_storage = match nested.then(|| root.get("quantization_config")).flatten() {
        None | Some(Value::Null) => None,
        Some(v) => { let q = object(v,"quantization_config")?; known(q,"quantization_config",STORAGE_KEYS)?;
            Some(DeclaredStorage { method: string(get(q,"quant_method")?,"quant_method")?.into(), algo: string(get(q,"quant_algo")?,"quant_algo")?.into() }) },
    };
    let declared_draft = match nested.then(|| root.get("mtp_config")).flatten() {
        None | Some(Value::Null) => None,
        Some(v) => { let m = object(v,"mtp_config")?;
            let draft = DraftDeclaration { layers: n(m,"num_hidden_layers",MAX_LAYERS,false)?, taps: rising(m,"target_layer_ids",visits.len() as u64 + 1)?,
                hidden: n(m,"hidden_size",MAX_HIDDEN,false)?, query_heads: n(m,"num_attention_heads",MAX_HEADS,false)?,
                kv_heads: n(m,"num_key_value_heads",MAX_HEADS,false)?, head_dim: n(m,"head_dim",MAX_HEAD_DIM,false)?,
                width: n(m,"intermediate_size",MAX_WIDTH,false)?, markov_rank: n(m,"markov_rank",MAX_WIDTH,false)? };
            check(draft.hidden == hidden,"mtp_config","hidden size differs from the text model")?;
            Some(draft) },
    };
    let declares_vision = nested && !matches!(root.get("vision_config"),None | Some(Value::Null));
    Ok(LoopTextDescription { hidden, vocab, head_rows, max_positions: n(o,"model_max_length",MAX_POSITIONS,false)?, epsilon,
        eos_tokens, pad_token, layers, visits, residual: BlockMixDescription { block_layers, normalize: true }, core, routing, caps,
        conv: ShortConvDescription { kernel, taps: ConvTaps::Fixed { order: FIXED_TAP_ORDER } },
        declared_storage, declared_draft, declares_vision })
}
