//! Official 31B assistant: query-only decoder over the target's last local/global KV.
use crate::{Result, invalid, gemma4::{Attention, Config as Text, Rope}, json::{self, Value, field}};
use std::collections::BTreeMap;

pub struct Config { pub text: Text, pub backbone_hidden_size: u64 }
impl Config {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let doc = json::parse(bytes, json::Limits { bytes: 1024*1024, ..Default::default() })?;
        let root = doc.object()?;
        let o = field(root, "text_config")?.object()?;
        if field(root, "model_type")?.string()? != "gemma4_assistant"
            || field(root, "architectures")?.array()? != [Value::String("Gemma4AssistantForCausalLM".into())]
            || field(o, "model_type")?.string()? != "gemma4_text"
            || field(o, "dtype")?.string()? != "bfloat16"
            || field(o, "hidden_activation")?.string()? != "gelu_pytorch_tanh"
            || field(root, "use_ordered_embeddings")?.boolean()? {
            return Err(invalid("unsupported Gemma assistant architecture, precision or ordered head"));
        }
        // The official 31B checkpoint uses its complete tied head. Centroids
        // are inactive when use_ordered_embeddings=false; never apply pruning.
        for (name, expected) in [("backbone_hidden_size",5376), ("num_centroids",2048), ("centroid_intermediate_top_k",32)] {
            if field(root, name)?.u64()? != expected { return Err(invalid(format!("unsupported assistant {name}"))); }
        }
        for (name, expected) in [("hidden_size",1024), ("intermediate_size",8192), ("num_attention_heads",32),
            ("num_key_value_heads",16), ("num_global_key_value_heads",4), ("head_dim",256), ("global_head_dim",512),
            ("num_hidden_layers",4), ("num_kv_shared_layers",4), ("vocab_size",262144), ("max_position_embeddings",262144),
            ("sliding_window",1024), ("hidden_size_per_layer_input",0), ("vocab_size_per_layer_input",0)] {
            if field(o, name)?.u64()? != expected { return Err(invalid(format!("unsupported assistant {name}"))); }
        }
        for name in ["attention_bias", "enable_moe_block", "use_double_wide_mlp"] {
            if field(o, name)?.boolean()? { return Err(invalid(format!("unsupported assistant {name}"))); }
        }
        if !field(o, "attention_k_eq_v")?.boolean()? || !field(o, "tie_word_embeddings")?.boolean()?
            || !field(root, "tie_word_embeddings")?.boolean()? || field(o, "final_logit_softcapping")? != &Value::Null
            || field(o, "use_bidirectional_attention")? != &Value::Null
            || root.get("quantization_config").is_some_and(|v| v != &Value::Null) {
            return Err(invalid("unsupported assistant attention, head or quantization"));
        }
        let types = field(o, "layer_types")?.array()?;
        let names = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"];
        if types.len() != names.len() || types.iter().zip(names).any(|(v, n)| v.string().ok() != Some(n)) {
            return Err(invalid("unsupported assistant layer types"));
        }
        let parameters = field(o, "rope_parameters")?.object()?;
        let rope = |name: &str, theta: f64, fraction: f64, kind: &str| -> Result<Rope> {
            let r = field(parameters, name)?.object()?;
            if field(r, "rope_theta")?.f64()? != theta || field(r, "rope_type")?.string()? != kind
                || r.get("partial_rotary_factor").map(Value::f64).transpose()?.unwrap_or(1.0) != fraction {
                return Err(invalid("unsupported assistant RoPE"));
            }
            Ok(Rope { theta, rotary_fraction: fraction, proportional: kind == "proportional" })
        };
        let eps = field(o, "rms_norm_eps")?.f64()?;
        if eps != 1e-6 { return Err(invalid("unsupported assistant RMS epsilon")); }
        Ok(Self { backbone_hidden_size: 5376, text: Text {
            hidden_size: 1024, intermediate_size: 8192, attention_heads: 32, kv_heads: 16, global_kv_heads: 4,
            head_dim: 256, global_head_dim: 512, vocab_size: 262144, max_positions: 262144, sliding_window: 1024,
            layer_types: vec![Attention::Sliding, Attention::Sliding, Attention::Sliding, Attention::Full],
            sliding_rope: rope("sliding_attention", 10000.0, 1.0, "default")?,
            full_rope: rope("full_attention", 1000000.0, 0.25, "proportional")?, rms_norm_eps: eps,
            final_logit_softcapping: 0.0, attention_k_eq_v: true, tie_word_embeddings: true,
            bos_token: field(o,"bos_token_id")?.u64()?, pad_token: field(o,"pad_token_id")?.u64()?,
            eos_tokens: vec![field(o,"eos_token_id")?.u64()?],
        } })
    }
    pub fn expected_shapes(&self) -> BTreeMap<String, Vec<u64>> {
        let mut shapes: BTreeMap<_,_> = self.text.expected_shapes().into_iter()
            .filter(|(n,_)| ![".k_proj.",".v_proj.",".k_norm."].iter().any(|s| n.contains(s)))
            .map(|(n,s)| (n.replace("model.language_model", "model"),s)).collect();
        shapes.insert("pre_projection.weight".into(), vec![self.text.hidden_size, 2*self.backbone_hidden_size]);
        shapes.insert("post_projection.weight".into(), vec![self.backbone_hidden_size, self.text.hidden_size]);
        shapes
    }
    pub fn weight_bytes(&self) -> u64 { self.expected_shapes().values().map(|s| 2*s.iter().product::<u64>()).sum() }
    pub fn validate_target(&self, target: &Text) -> Result<()> {
        let c = &self.text;
        if target.hidden_size != self.backbone_hidden_size || target.vocab_size != c.vocab_size
            || target.geometry(Attention::Sliding) != c.geometry(Attention::Sliding)
            || target.geometry(Attention::Full) != c.geometry(Attention::Full)
            || target.sliding_window != c.sliding_window || target.rms_norm_eps != c.rms_norm_eps
            || target.sliding_rope.theta != c.sliding_rope.theta || target.full_rope.theta != c.full_rope.theta
            || target.full_rope.rotary_fraction != c.full_rope.rotary_fraction {
            return Err(invalid("assistant does not match target hidden, KV, vocabulary or RoPE geometry"));
        }
        Ok(())
    }
}
