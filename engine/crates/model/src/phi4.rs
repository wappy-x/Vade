//! Phi3ForCausalLM BF16 contract, read from the official Phi-4 checkpoint.
use crate::{Result, invalid, json::{Value, field}, qwen3::Config};

pub(crate) fn parse(value: &Value) -> Result<Config> {
    let o = value.object()?;
    let allowed = ["_name_or_path", "architectures", "attention_bias", "attention_dropout",
        "bos_token_id", "embd_pdrop", "eos_token_id", "hidden_act", "hidden_size",
        "initializer_range", "intermediate_size", "max_position_embeddings", "model_type",
        "num_attention_heads", "num_hidden_layers", "num_key_value_heads",
        "original_max_position_embeddings", "pad_token_id", "resid_pdrop", "rms_norm_eps",
        "rope_scaling", "rope_theta", "sliding_window", "tie_word_embeddings", "torch_dtype",
        "transformers_version", "use_cache", "vocab_size"];
    for key in o.keys() {
        if !allowed.contains(&key.as_str()) { return Err(invalid(format!("unsupported Phi config field {key}"))); }
    }
    let architectures = field(o, "architectures")?.array()?;
    if architectures.len() != 1 || architectures[0].string()? != "Phi3ForCausalLM"
        || field(o, "model_type")?.string()? != "phi3"
        || field(o, "torch_dtype")?.string()? != "bfloat16"
        || field(o, "hidden_act")?.string()? != "silu" {
        return Err(invalid("Phi adapter requires Phi3ForCausalLM BF16 with SiLU"));
    }
    if field(o, "attention_bias")?.boolean()? || field(o, "tie_word_embeddings")?.boolean()?
        || !field(o, "use_cache")?.boolean()? {
        return Err(invalid("unsupported Phi bias, tied embeddings or disabled cache"));
    }
    for key in ["attention_dropout", "embd_pdrop", "resid_pdrop"] {
        if field(o, key)?.f64()? != 0.0 { return Err(invalid(format!("unsupported Phi {key}"))); }
    }
    for key in ["rope_scaling", "sliding_window"] {
        if field(o, key)? != &Value::Null { return Err(invalid(format!("unsupported Phi {key}"))); }
    }
    let n = |key| field(o, key)?.u64();
    let hidden_size = n("hidden_size")?;
    let attention_heads = n("num_attention_heads")?;
    if attention_heads == 0 || hidden_size % attention_heads != 0 {
        return Err(invalid("Phi hidden size must divide into complete attention heads"));
    }
    let c = Config {
        qk_norm: false, fused_projections: true, llama_rope: None,
        hidden_size, attention_heads, head_dim: hidden_size / attention_heads,
        intermediate_size: n("intermediate_size")?, layers: n("num_hidden_layers")?,
        kv_heads: n("num_key_value_heads")?, vocab_size: n("vocab_size")?,
        max_positions: n("max_position_embeddings")?, bos_token: n("bos_token_id")?,
        eos_token: n("eos_token_id")?, rms_norm_eps: field(o, "rms_norm_eps")?.f64()?,
        rope_theta: field(o, "rope_theta")?.f64()?, yarn: false, tie_word_embeddings: false,
        moe_intermediate_size: 0, num_experts: 0, num_experts_per_tok: 0, norm_topk_prob: false,
    };
    for (key, value, cap) in [("hidden_size", c.hidden_size, 65536),
        ("intermediate_size", c.intermediate_size, 1048576), ("layers", c.layers, 1024),
        ("attention_heads", c.attention_heads, 1024), ("kv_heads", c.kv_heads, 1024),
        ("head_dim", c.head_dim, 4096), ("vocab_size", c.vocab_size, 4194304),
        ("max_positions", c.max_positions, 1 << 32)] {
        if value == 0 || value > cap { return Err(invalid(format!("Phi {key} outside safety bounds"))); }
    }
    if c.attention_heads % c.kv_heads != 0 || c.head_dim % 2 != 0
        || c.rms_norm_eps <= 0.0 || c.rope_theta <= 1.0
        || c.bos_token >= c.vocab_size || c.eos_token >= c.vocab_size
        || n("pad_token_id")? >= c.vocab_size
        || n("original_max_position_embeddings")? != c.max_positions {
        return Err(invalid("unsupported Phi geometry, tokens or extended context"));
    }
    Ok(c)
}
