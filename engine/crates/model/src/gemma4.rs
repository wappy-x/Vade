//! Gemma 4 dense text geometry; multimodal towers are excluded from residency.
use crate::{
    Result, invalid,
    json::{self, Value, field},
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attention {
    Sliding,
    Full,
}
#[derive(Debug, Clone)]
pub struct Rope {
    pub theta: f64,
    pub rotary_fraction: f64,
    pub proportional: bool,
}
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub global_kv_heads: u64,
    pub head_dim: u64,
    pub global_head_dim: u64,
    pub vocab_size: u64,
    pub max_positions: u64,
    pub sliding_window: u64,
    pub layer_types: Vec<Attention>,
    pub sliding_rope: Rope,
    pub full_rope: Rope,
    pub rms_norm_eps: f64,
    pub final_logit_softcapping: f64,
    pub attention_k_eq_v: bool,
    pub tie_word_embeddings: bool,
    pub bos_token: u64,
    pub pad_token: u64,
    pub eos_tokens: Vec<u64>,
}
impl Config {
    #[allow(clippy::manual_is_multiple_of)] // A zero divisor must still panic in `%`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let doc = json::parse(
            bytes,
            json::Limits {
                bytes: 1024 * 1024,
                ..Default::default()
            },
        )?;
        let root = doc.object()?;
        if field(root, "model_type")?.string()? != "gemma4"
            || field(root, "architectures")?
                .array()?
                .iter()
                .map(Value::string)
                .collect::<Result<Vec<_>>>()?
                != ["Gemma4ForConditionalGeneration"]
        {
            return Err(invalid("expected Gemma4ForConditionalGeneration"));
        }
        let o = field(root, "text_config")?.object()?;
        let allowed = [
            "attention_bias",
            "attention_dropout",
            "attention_k_eq_v",
            "bos_token_id",
            "dtype",
            "enable_moe_block",
            "eos_token_id",
            "expert_intermediate_size",
            "final_logit_softcapping",
            "global_head_dim",
            "head_dim",
            "hidden_activation",
            "hidden_size",
            "hidden_size_per_layer_input",
            "initializer_range",
            "intermediate_size",
            "layer_types",
            "max_position_embeddings",
            "model_type",
            "num_attention_heads",
            "num_experts",
            "num_global_key_value_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_kv_shared_layers",
            "pad_token_id",
            "rms_norm_eps",
            "rope_parameters",
            "sliding_window",
            "tie_word_embeddings",
            "top_k_experts",
            "use_bidirectional_attention",
            "use_cache",
            "use_double_wide_mlp",
            "vocab_size",
            "vocab_size_per_layer_input",
        ];
        if let Some(key) = o.keys().find(|k| !allowed.contains(&k.as_str())) {
            return Err(invalid(format!(
                "unsupported Gemma 4 text config field {key}"
            )));
        }
        if root
            .get("quantization_config")
            .is_some_and(|v| v != &Value::Null)
        {
            return Err(invalid(
                "Gemma 4 text memory accounting requires native BF16 weights",
            ));
        }
        if field(o, "model_type")?.string()? != "gemma4_text"
            || field(o, "dtype")?.string()? != "bfloat16"
            || field(o, "hidden_activation")?.string()? != "gelu_pytorch_tanh"
        {
            return Err(invalid("Gemma 4 text requires BF16 and gelu_pytorch_tanh"));
        }
        for key in ["attention_bias", "enable_moe_block", "use_double_wide_mlp"] {
            if field(o, key)?.boolean()? {
                return Err(invalid(format!("unsupported Gemma 4 {key}")));
            }
        }
        for key in ["num_kv_shared_layers", "hidden_size_per_layer_input"] {
            if field(o, key)?.u64()? != 0 {
                return Err(invalid(format!("unsupported Gemma 4 {key}")));
            }
        }
        if field(o, "attention_dropout")?.f64()? != 0.0
            || field(o, "use_bidirectional_attention")?.string()? != "vision"
        {
            return Err(invalid(
                "Gemma 4 text requires causal attention without dropout",
            ));
        }
        for key in ["expert_intermediate_size", "num_experts", "top_k_experts"] {
            if field(o, key)? != &Value::Null {
                return Err(invalid("Gemma 4 MoE geometry is not implemented"));
            }
        }
        let positive = |key| -> Result<u64> {
            let n = field(o, key)?.u64()?;
            if n == 0 || n > 1_000_000 {
                return Err(invalid(format!("invalid Gemma 4 {key}")));
            }
            Ok(n)
        };
        let layers = positive("num_hidden_layers")?;
        if layers > 4096 {
            return Err(invalid("Gemma 4 layer count exceeds metadata limit"));
        }
        let layer_types = field(o, "layer_types")?
            .array()?
            .iter()
            .map(|v| match v.string()? {
                "sliding_attention" => Ok(Attention::Sliding),
                "full_attention" => Ok(Attention::Full),
                _ => Err(invalid("unknown Gemma 4 layer type")),
            })
            .collect::<Result<Vec<_>>>()?;
        if layer_types.len() as u64 != layers {
            return Err(invalid("Gemma 4 layer_types length mismatch"));
        }
        let parameters = field(o, "rope_parameters")?.object()?;
        let rope = |key: &str, proportional: bool| -> Result<Rope> {
            let r = field(parameters, key)?.object()?;
            let theta = field(r, "rope_theta")?.f64()?;
            let fraction = r
                .get("partial_rotary_factor")
                .map(Value::f64)
                .transpose()?
                .unwrap_or(1.0);
            if field(r, "rope_type")?.string()?
                != if proportional {
                    "proportional"
                } else {
                    "default"
                }
                || theta <= 0.0
                || !(0.0..=1.0).contains(&fraction)
                || (!proportional && fraction != 1.0)
                || r.keys().any(|k| {
                    !["rope_theta", "rope_type", "partial_rotary_factor"].contains(&k.as_str())
                })
            {
                return Err(invalid("unsupported Gemma 4 RoPE parameters"));
            }
            Ok(Rope {
                theta,
                rotary_fraction: fraction,
                proportional,
            })
        };
        let c = Self {
            hidden_size: positive("hidden_size")?,
            intermediate_size: positive("intermediate_size")?,
            attention_heads: positive("num_attention_heads")?,
            kv_heads: positive("num_key_value_heads")?,
            global_kv_heads: positive("num_global_key_value_heads")?,
            head_dim: positive("head_dim")?,
            global_head_dim: positive("global_head_dim")?,
            vocab_size: positive("vocab_size")?,
            max_positions: positive("max_position_embeddings")?,
            sliding_window: positive("sliding_window")?,
            layer_types,
            sliding_rope: rope("sliding_attention", false)?,
            full_rope: rope("full_attention", true)?,
            rms_norm_eps: field(o, "rms_norm_eps")?.f64()?,
            final_logit_softcapping: field(o, "final_logit_softcapping")?.f64()?,
            attention_k_eq_v: field(o, "attention_k_eq_v")?.boolean()?,
            tie_word_embeddings: field(o, "tie_word_embeddings")?.boolean()?,
            bos_token: field(o, "bos_token_id")?.u64()?,
            pad_token: field(o, "pad_token_id")?.u64()?,
            eos_tokens: field(root, "eos_token_id")?
                .array()?
                .iter()
                .map(Value::u64)
                .collect::<Result<_>>()?,
        };
        if c.attention_heads % c.kv_heads != 0
            || c.attention_heads % c.global_kv_heads != 0
            || !c.head_dim.is_multiple_of(2)
            || !c.global_head_dim.is_multiple_of(2)
            || c.rms_norm_eps <= 0.0
            || c.final_logit_softcapping <= 0.0
            || c.eos_tokens.is_empty()
            || c.eos_tokens
                .iter()
                .chain([&c.bos_token, &c.pad_token])
                .any(|&id| id >= c.vocab_size)
            || field(root, "tie_word_embeddings")?.boolean()? != c.tie_word_embeddings
        {
            return Err(invalid(
                "invalid Gemma 4 attention, normalization or token geometry",
            ));
        }
        c.weight_bytes()?;
        Ok(c)
    }
    pub fn geometry(&self, kind: Attention) -> (u64, u64) {
        match kind {
            Attention::Sliding => (self.kv_heads, self.head_dim),
            Attention::Full => (self.global_kv_heads, self.global_head_dim),
        }
    }
    pub fn expected_shapes(&self) -> BTreeMap<String, Vec<u64>> {
        let mut out = BTreeMap::new();
        let prefix = "model.language_model";
        out.insert(
            format!("{prefix}.embed_tokens.weight"),
            vec![self.vocab_size, self.hidden_size],
        );
        out.insert(format!("{prefix}.norm.weight"), vec![self.hidden_size]);
        if !self.tie_word_embeddings {
            out.insert(
                "lm_head.weight".into(),
                vec![self.vocab_size, self.hidden_size],
            );
        }
        for (i, &kind) in self.layer_types.iter().enumerate() {
            let p = format!("{prefix}.layers.{i}");
            let (kv, dim) = self.geometry(kind);
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                out.insert(format!("{p}.{name}.weight"), vec![self.hidden_size]);
            }
            out.insert(format!("{p}.layer_scalar"), vec![1]);
            for name in ["q_norm", "k_norm"] {
                out.insert(format!("{p}.self_attn.{name}.weight"), vec![dim]);
            }
            out.insert(
                format!("{p}.self_attn.q_proj.weight"),
                vec![self.attention_heads * dim, self.hidden_size],
            );
            out.insert(
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv * dim, self.hidden_size],
            );
            if kind == Attention::Sliding || !self.attention_k_eq_v {
                out.insert(
                    format!("{p}.self_attn.v_proj.weight"),
                    vec![kv * dim, self.hidden_size],
                );
            }
            out.insert(
                format!("{p}.self_attn.o_proj.weight"),
                vec![self.hidden_size, self.attention_heads * dim],
            );
            for name in ["gate_proj", "up_proj"] {
                out.insert(
                    format!("{p}.mlp.{name}.weight"),
                    vec![self.intermediate_size, self.hidden_size],
                );
            }
            out.insert(
                format!("{p}.mlp.down_proj.weight"),
                vec![self.hidden_size, self.intermediate_size],
            );
        }
        out
    }
    pub fn weight_bytes(&self) -> Result<u64> {
        self.expected_shapes()
            .values()
            .try_fold(0u64, |sum, shape| {
                shape
                    .iter()
                    .try_fold(2u64, |n, &d| n.checked_mul(d))
                    .and_then(|n| sum.checked_add(n))
            })
            .ok_or_else(|| invalid("Gemma 4 text weight byte overflow"))
    }
    /// BF16 K/V residency: local window plus 63 rows of shared prefill slack.
    pub fn kv_bytes(&self, tokens: u64) -> Result<u64> {
        if tokens == 0 || tokens > self.max_positions {
            return Err(invalid(engine_base::admission::token_refusal(tokens as u128, self.max_positions as usize, &format!("CONTEXT={tokens}; OUTPUT included in CONTEXT; context must be positive"), "the configured position ceiling", engine_base::admission::SERVE)));
        }
        self.layer_types
            .iter()
            .try_fold(0u64, |sum, &kind| {
                let (kv, dim) = self.geometry(kind);
                let n = if kind == Attention::Sliding {
                    tokens.min(self.sliding_window + 63)
                } else {
                    tokens
                };
                kv.checked_mul(dim)
                    .and_then(|x| x.checked_mul(n))
                    .and_then(|x| x.checked_mul(4))
                    .and_then(|x| sum.checked_add(x))
            })
            .ok_or_else(|| invalid("Gemma 4 KV byte overflow"))
    }
    pub fn rope_bytes(&self, tokens: u64) -> Result<u64> {
        self.head_dim
            .checked_add(self.global_head_dim)
            .and_then(|x| x.checked_mul(tokens))
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| invalid("Gemma 4 RoPE byte overflow"))
    }
    /// Estimate, not an implemented allocator contract: one weight copy, two
    /// minimal sessions, shared workspace, scratch, and the usual OS reserve.
    pub fn minimum_free_bytes(&self) -> Result<u64> {
        self.required_free_bytes(1, 2)
    }
    pub fn required_free_bytes(&self, tokens: u64, sessions: u64) -> Result<u64> {
        if sessions == 0 {
            return Err(invalid("Gemma 4 memory estimate requires a session"));
        }
        let kv = self
            .kv_bytes(tokens)?
            .checked_mul(sessions)
            .ok_or_else(|| invalid("KV byte overflow"))?;
        let gpu = self
            .weight_bytes()?
            .checked_add(self.rope_bytes(tokens)?)
            .and_then(|x| x.checked_add(kv))
            .and_then(|x| x.checked_add((2 * 1024 + 64) * 1024 * 1024))
            .ok_or_else(|| invalid("Gemma 4 memory size overflow"))?;
        let reserve = gpu
            .checked_mul(10)
            .map(|x| x.div_ceil(9))
            .ok_or_else(|| invalid("Gemma 4 reserve overflow"))?;
        Ok(reserve.max(
            gpu.checked_add(1024 * 1024 * 1024)
                .ok_or_else(|| invalid("Gemma 4 reserve overflow"))?,
        ))
    }
}
