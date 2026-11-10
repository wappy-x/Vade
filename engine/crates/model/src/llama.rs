//! Official Llama 3.1/3.2 BF16 dense checkpoint contract.
use crate::{Result, invalid, json::{Value, field}, qwen3::Config};

#[derive(Debug, Clone)]
pub struct Rope { pub factor: f32, pub low: f32, pub high: f32, pub original: f32 }

pub(crate) fn parse(value: &Value) -> Result<Config> {
    let o = value.object()?;
    let allowed = ["_name_or_path", "architectures", "attention_bias", "attention_dropout",
        "bos_token_id", "eos_token_id", "head_dim", "hidden_act", "hidden_size",
        "initializer_range", "intermediate_size", "max_position_embeddings", "mlp_bias",
        "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads",
        "pad_token_id", "pretraining_tp", "rms_norm_eps", "rope_scaling", "rope_theta",
        "tie_word_embeddings", "torch_dtype", "transformers_version", "use_cache", "vocab_size"];
    for key in o.keys() {
        if !allowed.contains(&key.as_str()) { return Err(invalid(format!("unsupported Llama config field {key}"))); }
    }
    let a = field(o, "architectures")?.array()?;
    if a.len() != 1 || a[0].string()? != "LlamaForCausalLM"
        || field(o, "model_type")?.string()? != "llama"
        || field(o, "torch_dtype")?.string()? != "bfloat16"
        || field(o, "hidden_act")?.string()? != "silu"
        || field(o, "attention_bias")?.boolean()? || field(o, "mlp_bias")?.boolean()?
        || field(o, "attention_dropout")?.f64()? != 0.0 || !field(o, "use_cache")?.boolean()?
        || field(o, "pretraining_tp")?.u64()? != 1 {
        return Err(invalid("Llama requires official bias-free BF16 SiLU configuration"));
    }
    let scaling = field(o, "rope_scaling")?.object()?;
    if field(scaling, "rope_type")?.string()? != "llama3"
        || scaling.keys().any(|k| !matches!(k.as_str(), "rope_type" | "factor" |
            "low_freq_factor" | "high_freq_factor" | "original_max_position_embeddings")) {
        return Err(invalid("unsupported Llama RoPE scaling"));
    }
    let rope = Rope { factor: field(scaling, "factor")?.f64()? as f32,
        low: field(scaling, "low_freq_factor")?.f64()? as f32,
        high: field(scaling, "high_freq_factor")?.f64()? as f32,
        original: field(scaling, "original_max_position_embeddings")?.u64()? as f32 };
    if !matches!(rope.factor, 8.0 | 32.0) || rope.low != 1.0 || rope.high != 4.0 || rope.original != 8192.0 {
        return Err(invalid("unsupported Llama 3.1/3.2 frequency scaling"));
    }
    let n = |key| field(o, key)?.u64();
    let hidden = n("hidden_size")?; let heads = n("num_attention_heads")?;
    if heads == 0 || hidden % heads != 0 { return Err(invalid("invalid Llama head geometry")); }
    let dim = o.get("head_dim").map(Value::u64).transpose()?.unwrap_or(hidden / heads);
    let eos = field(o, "eos_token_id")?.array()?.iter().map(Value::u64).collect::<Result<Vec<_>>>()?;
    if eos != [128001, 128008, 128009] || n("bos_token_id")? != 128000 || n("vocab_size")? != 128256 {
        return Err(invalid("unsupported Llama special-token contract"));
    }
    let c = Config { qk_norm: false, fused_projections: false, llama_rope: Some(rope),
        hidden_size: hidden, attention_heads: heads, head_dim: dim,
        intermediate_size: n("intermediate_size")?, layers: n("num_hidden_layers")?,
        kv_heads: n("num_key_value_heads")?, vocab_size: n("vocab_size")?,
        max_positions: n("max_position_embeddings")?, bos_token: 128000, eos_token: 128009,
        rms_norm_eps: field(o, "rms_norm_eps")?.f64()?, rope_theta: field(o, "rope_theta")?.f64()?,
        yarn: false, tie_word_embeddings: field(o, "tie_word_embeddings")?.boolean()?,
        moe_intermediate_size: 0, num_experts: 0, num_experts_per_tok: 0, norm_topk_prob: false };
    if !matches!((c.hidden_size, c.intermediate_size, c.layers, c.attention_heads, c.kv_heads, c.head_dim),
        (2048, 8192, 16, 32, 8, 64) | (3072, 8192, 28, 24, 8, 128) | (4096, 14336, 32, 32, 8, 128))
        || c.max_positions != 131072 || c.rms_norm_eps != 1e-5 || c.rope_theta != 500000.0
        || c.llama_rope.as_ref().unwrap().factor != if c.hidden_size == 4096 { 8.0 } else { 32.0 }
        || c.tie_word_embeddings != (c.hidden_size != 4096) {
        return Err(invalid("unsupported Llama 3.1/3.2 checkpoint geometry"));
    }
    Ok(c)
}
