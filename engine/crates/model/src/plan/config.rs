//! Config adapters for checkpoint-derived geometry and supported semantics.
use super::*;
use crate::json::{self, Value};
type Object = BTreeMap<String, Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Family { Flash, Hybrid }
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Schema {
    pub family: Family,
    pub tied: bool,
    /// Explicit config intent; omission stays unspecified until headers arrive.
    pub declared_tied: Option<bool>,
    pub ple: Option<Ple>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Ple { pub size: u64, pub heads: u64, pub embed: u64, pub conv: u64 }

fn get<'a>(o: &'a Object, field: &'static str) -> Result<&'a Value> {
    o.get(field).ok_or(PlanError::Missing { field })
}
fn wrong(field: &'static str, expected: &'static str) -> PlanError {
    PlanError::WrongType { field, expected }
}
fn object<'a>(v: &'a Value, field: &'static str) -> Result<&'a Object> {
    v.object().map_err(|_| wrong(field, "object"))
}
fn array<'a>(v: &'a Value, field: &'static str) -> Result<&'a [Value]> {
    v.array().map_err(|_| wrong(field, "array"))
}
fn text<'a>(v: &'a Value, field: &'static str) -> Result<&'a str> {
    v.string().map_err(|_| wrong(field, "string"))
}
fn integer(v: &Value, field: &'static str) -> Result<u64> {
    v.u64().map_err(|_| wrong(field, "unsigned integer"))
}
fn boolean(o: &Object, field: &'static str) -> Result<bool> {
    get(o, field)?.boolean().map_err(|_| wrong(field, "boolean"))
}
fn number(o: &Object, field: &'static str) -> Result<f64> {
    get(o, field)?.f64().map_err(|_| wrong(field, "finite number"))
}
fn bounded(value: u64, field: &'static str, cap: u64) -> Result<u64> {
    if value == 0 || value > cap { return Err(invalid(field, "dimension is zero or above its limit")); }
    Ok(value)
}
fn n(o: &Object, field: &'static str, cap: u64) -> Result<u64> {
    bounded(integer(get(o, field)?, field)?, field, cap)
}
fn optional_bool(o: &Object, field: &'static str, default: bool) -> Result<bool> {
    if o.contains_key(field) { boolean(o, field) } else { Ok(default) }
}
fn require(ok: bool, o: &Object, field: &'static str) -> Result<()> {
    if ok { Ok(()) } else { Err(PlanError::Unsupported { field, value: format!("{:?}", get(o, field)?) }) }
}
fn is(o: &Object, field: &'static str, expected: &str) -> Result<()> {
    require(text(get(o, field)?, field)? == expected, o, field)
}
fn flag(o: &Object, field: &'static str, expected: bool) -> Result<()> {
    require(boolean(o, field)? == expected, o, field)
}
fn flash(o: &Object) -> Result<Schema> {
    is(o, "output_gate_type", "sigmoid")?;
    flag(o, "mtp_use_dedicated_embeddings", false)?;
    Ok(Schema { family: Family::Flash, tied: false, declared_tied: None, ple: Some(Ple {
        size: n(o, "ngram_size", MAX_HEADS)?, heads: n(o, "heads_per_ngram", MAX_HEADS)?,
        embed: n(o, "ple_embed_dim", MAX_WIDTH)?, conv: n(o, "ple_conv_kernel_size", MAX_HEADS)?,
    }) })
}
fn hybrid(root: &Object, o: &Object, moe: bool) -> Result<Schema> {
    if root.contains_key("text_config") {
        let expected = if moe { "Qwen3_5MoeForConditionalGeneration" } else { "Qwen3_5ForConditionalGeneration" };
        let a = array(get(root, "architectures")?, "architectures")?;
        require(a == [Value::String(expected.into())], root, "architectures")?;
    }
    is(o, "dtype", "bfloat16")?;
    if o.contains_key("output_gate_type") { is(o, "output_gate_type", "swish")?; }
    flag(o, "attn_output_gate", true)?;
    require(number(o, "attention_dropout")? == 0.0, o, "attention_dropout")?;
    flag(o, "use_cache", true)?;
    Ok(Schema { family: Family::Hybrid, tied: false, declared_tied: None, ple: None })
}
fn format(root: &Object, o: &Object, family: Family) -> Result<Format> {
    let q = root.get("quantization_config").or_else(|| o.get("quantization_config"));
    let Some(q) = q else {
        if family == Family::Flash { return Err(PlanError::Missing { field: "quantization_config" }); }
        return Ok(Format::Bf16);
    };
    let q = object(q, "quantization_config")?;
    if text(get(q, "quant_method")?, "quant_method")? == "modelopt" {
        let mixed = crate::flash::mixed::Mixed::parse(q).map_err(|e| read_error("quantization_config", e))?;
        if mixed.layers.values().any(|s| matches!(s, crate::flash::mixed::Storage::Fp8Block { .. })) {
            return Err(PlanError::MissingOps { ops: vec!["FP8_PB_WO weight_scale execution binding".into()] });
        }
        return Ok(Format::Mixed(mixed));
    }
    is(q, "quant_method", "fp8")?;
    is(q, "activation_scheme", "dynamic")?;
    for field in ["weight_per_tensor", "act_per_tensor"] {
        if family == Family::Flash || !q.contains_key("fmt") || q.contains_key(field) { flag(q, field, false)?; }
    }
    if family == Family::Hybrid && q.contains_key("fmt") { is(q, "fmt", "e4m3")?; }
    let block = array(get(q, "weight_block_size")?, "weight_block_size")?.iter()
        .map(|v| integer(v, "weight_block_size")).collect::<Result<Vec<_>>>()?;
    require(block == [FP8_BLOCK, FP8_BLOCK], q, "weight_block_size")?;
    let unquantized = array(get(q, "modules_to_not_convert")?, "modules_to_not_convert")?.iter()
        .map(|v| Ok(text(v, "modules_to_not_convert")?.to_owned())).collect::<Result<Vec<_>>>()?;
    Ok(Format::Fp8 { block: [FP8_BLOCK, FP8_BLOCK], unquantized })
}
fn layers(o: &Object, count: u64, linear: LinearShape, attention: AttentionShape, ffn: Ffn) -> Result<Vec<LayerShape>> {
    let types = array(get(o, "layer_types")?, "layer_types")?;
    if types.len() as u64 != count { return Err(invalid("layer_types", "length differs from num_hidden_layers")); }
    types.iter().map(|v| Ok(LayerShape { ffn, mixer: match text(v, "layer_types")? {
        "linear_attention" => Mixer::Linear(linear), "full_attention" => Mixer::Attention(attention),
        value => return Err(PlanError::Unsupported { field: "layer_types", value: value.into() }),
    } })).collect()
}
pub(super) fn parse(bytes: &[u8]) -> Result<ModelPlan> {
    let doc = json::parse(bytes, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })
        .map_err(|e| read_error("config.json", e))?;
    let root = object(&doc, "config")?;
    let model_type = text(get(root, "model_type")?, "model_type")?;
    let (family, moe, expected) = match model_type {
        "qwen4_exp" | "qwen4_exp_text" => (Family::Flash, true, "qwen4_exp_text"),
        "qwen3_5_moe" | "qwen3_5_moe_text" => (Family::Hybrid, true, "qwen3_5_moe_text"),
        "qwen3_5" | "qwen3_5_text" => (Family::Hybrid, false, "qwen3_5_text"),
        value => return Err(PlanError::Unsupported { field: "model_type", value: value.into() }),
    };
    let o = if model_type == expected { root } else { object(get(root, "text_config")?, "text_config")? };
    is(o, "model_type", expected)?;
    is(o, "hidden_act", "silu")?;
    is(o, "mamba_ssm_dtype", "float32")?;
    flag(o, "attention_bias", false)?;
    let mut schema = match family { Family::Flash => flash(o)?, Family::Hybrid => hybrid(root, o, moe)? };
    for config in [root, o] {
        if config.contains_key("tie_word_embeddings") {
            let tied = boolean(config, "tie_word_embeddings")?;
            require(schema.declared_tied.is_none_or(|v| v == tied), config, "tie_word_embeddings")?;
            schema.declared_tied = Some(tied);
        }
    }
    schema.tied = schema.declared_tied.unwrap_or(false);
    let format = format(root, o, family)?;
    let hidden = n(o, "hidden_size", MAX_HIDDEN)?;
    let vocab = n(o, "vocab_size", MAX_VOCAB)?;
    let head_dim = n(o, "head_dim", MAX_HEAD_DIM)?;
    let rope = object(get(o, "rope_parameters")?, "rope_parameters")?;
    is(rope, "rope_type", "default")?;
    flag(rope, "mrope_interleaved", true)?;
    let partial = number(rope, "partial_rotary_factor")?;
    if o.contains_key("partial_rotary_factor") && number(o, "partial_rotary_factor")? != partial {
        return Err(invalid("partial_rotary_factor", "text and RoPE factors differ"));
    }
    let sections = array(get(rope, "mrope_section")?, "mrope_section")?.iter()
        .map(|v| bounded(integer(v, "mrope_section")?, "mrope_section", MAX_HEAD_DIM)).collect::<Result<Vec<_>>>()?;
    let rotary_dim = mul(sections.iter().try_fold(0, |s, &v| add(s, v))?, 2)?;
    let rotary = head_dim as f64 * partial;
    if partial <= 0.0 || partial > 1.0 || rotary.fract() != 0.0 || rotary != rotary_dim as f64 || rotary_dim == 0 {
        return Err(invalid("partial_rotary_factor", "rotary dimension is not exact or differs from sections"));
    }
    let index_fields = ["indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio"];
    let index = if index_fields.iter().any(|f| o.contains_key(*f)) { Some(IndexShape {
        heads: n(o, "indexer_n_heads", MAX_HEADS)?, kv_heads: n(o, "indexer_kv_heads", MAX_HEADS)?,
        dim: n(o, "indexer_head_dim", MAX_HEAD_DIM)?, budget: n(o, "indexer_budget", MAX_POSITIONS)?,
        compression: n(o, "indexer_compress_ratio", MAX_POSITIONS)?,
    }) } else { None };
    let attention = AttentionShape { query_heads: n(o, "num_attention_heads", MAX_HEADS)?,
        kv_heads: n(o, "num_key_value_heads", MAX_HEADS)?, head_dim, rotary_dim, gated: true, index };
    let linear = LinearShape { key_heads: n(o, "linear_num_key_heads", MAX_HEADS)?,
        value_heads: n(o, "linear_num_value_heads", MAX_HEADS)?, key_dim: n(o, "linear_key_head_dim", MAX_HEAD_DIM)?,
        value_dim: n(o, "linear_value_head_dim", MAX_HEAD_DIM)?, conv: n(o, "linear_conv_kernel_dim", MAX_HEADS)? };
    if attention.query_heads % attention.kv_heads != 0 || linear.value_heads % linear.key_heads != 0 {
        return Err(invalid("heads", "query/KV or linear value/key heads do not divide exactly"));
    }
    let width = n(o, if moe { "moe_intermediate_size" } else { "intermediate_size" }, MAX_WIDTH)?;
    let ffn = if moe {
        let count = n(o, "num_experts", MAX_EXPERTS)?;
        let selected = n(o, "num_experts_per_tok", count)?;
        let shared = n(o, "shared_expert_intermediate_size", MAX_WIDTH)?;
        if matches!(format, Format::Fp8 { .. }) && width % FP8_BLOCK != 0 {
            return Err(invalid("moe_intermediate_size", "FP8 expert width is not a block multiple"));
        }
        Ffn::Experts(ExpertShape { count, selected, width, shared_width: shared,
            storage: if family == Family::Flash { ExpertStorage::PerExpert } else { ExpertStorage::Stacked } })
    } else { Ffn::Dense { width } };
    let count = n(o, "num_hidden_layers", MAX_LAYERS)?;
    let decoder = layers(o, count, linear, attention, ffn)?;
    if o.contains_key("full_attention_interval") {
        let interval = n(o, "full_attention_interval", MAX_LAYERS)?;
        for (i, layer) in decoder.iter().enumerate() {
            if matches!(layer.mixer, Mixer::Attention(_)) != (add(i as u64, 1)? % interval == 0) {
                return Err(invalid("full_attention_interval", "does not match layer_types"));
            }
        }
    }
    let draft_count = integer(get(o, "mtp_num_hidden_layers")?, "mtp_num_hidden_layers")?;
    if draft_count > MAX_LAYERS { return Err(invalid("mtp_num_hidden_layers", "above its limit")); }
    let draft = if draft_count == 0 { Vec::new() } else if family == Family::Hybrid {
        if o.contains_key("mtp_use_dedicated_embeddings") { flag(o, "mtp_use_dedicated_embeddings", false)?; }
        (0..draft_count).map(|_| LayerShape { mixer: Mixer::Attention(attention), ffn }).collect()
    } else {
        let mtp = object(get(o, "mtp")?, "mtp")?;
        if integer(get(mtp, "num_hidden_layers")?, "num_hidden_layers")? != draft_count {
            return Err(invalid("mtp_num_hidden_layers", "differs from mtp.num_hidden_layers"));
        }
        flag(mtp, "hybrid", true)?;
        require(get(mtp, "mtp_use_hidden_state_from_layer")? == &Value::Null, mtp, "mtp_use_hidden_state_from_layer")?;
        require(number(mtp, "rope_theta")? == number(rope, "rope_theta")?, mtp, "rope_theta")?;
        layers(mtp, draft_count, linear, attention, ffn)?
    };
    let residual = if family == Family::Flash { Residual::Streams {
        count: n(o, "hc_count", MAX_HEADS)?, rank: n(o, "hc_lowrank", MAX_WIDTH)?,
    } } else { Residual::Single };
    let lexical = if let Some(ple) = schema.ple {
        let ids = array(get(o, "ple_layer_ids")?, "ple_layer_ids")?;
        let [id] = ids else { return Err(PlanError::Unsupported { field: "ple_layer_ids", value: format!("{ids:?}") }); };
        let layer = integer(id, "ple_layer_ids")?.checked_sub(1).ok_or(invalid("ple_layer_ids", "must be one-based"))?;
        if layer >= count { return Err(invalid("ple_layer_ids", "outside decoder")); }
        let degree = ple.size.checked_sub(1).filter(|&v| v != 0).ok_or(invalid("ngram_size", "must exceed one"))?;
        let channels = mul(ple.heads, degree)?;
        if ple.embed % channels != 0 { return Err(invalid("ple_embed_dim", "ngram rows do not divide exactly")); }
        Some(LexicalShape { layer, parts: n(o, "split_ngram_parts", MAX_SHARDS as u64)?, part_rows: 0,
            row_dim: bounded(ple.embed / channels, "lexical row_dim", MAX_WIDTH)? })
    } else { None };
    let epsilon = number(o, "rms_norm_eps")?;
    let theta = number(rope, "rope_theta")?;
    if epsilon <= 0.0 || theta <= 1.0 { return Err(invalid("rms_norm_eps/rope_theta", "nonpositive norm or RoPE base")); }
    let eos_token = integer(get(o, "eos_token_id")?, "eos_token_id")?;
    let bos_token = o.get("bos_token_id").map(|v| integer(v, "bos_token_id")).transpose()?.unwrap_or(eos_token);
    if eos_token >= vocab || bos_token >= vocab {
        return Err(invalid("eos_token_id/bos_token_id", "outside vocabulary"));
    }
    Ok(ModelPlan { shape: ModelShape { hidden, vocab, max_positions: n(o, "max_position_embeddings", MAX_POSITIONS)?,
        layers: decoder, draft, residual, lexical, tensors: Vec::new() },
        rope: Rope { theta, rotary_dim, sections, interleaved: true }, epsilon, eos_token, bos_token,
        renormalize_top_k: optional_bool(o, "norm_topk_prob", true)?, format, schema })
}
