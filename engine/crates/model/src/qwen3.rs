//! Shared dense geometry and Qwen3 BF16 checkpoint contracts. Architecture
//! restrictions are explicit; the generic safetensors reader has no Qwen logic.
use crate::{
    Result, invalid,
    json::{self, Value, field},
    safetensors::{self, Dtype, Header, Index, Limits},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

#[derive(Debug, Clone)]
pub struct Config {
    pub qk_norm: bool,
    pub fused_projections: bool,
    pub llama_rope: Option<crate::llama::Rope>,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub moe_intermediate_size: u64,
    pub num_experts: u64,
    pub num_experts_per_tok: u64,
    pub norm_topk_prob: bool,
    pub layers: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub vocab_size: u64,
    pub max_positions: u64,
    pub bos_token: u64,
    pub eos_token: u64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Official Qwen3 extension: factor 4, original context 32768.
    pub yarn: bool,
    pub tie_word_embeddings: bool,
}
impl Config {
    pub fn open(path: &Path) -> Result<Self> {
        Self::parse(&safetensors::read_bounded(path, 1024 * 1024)?)
    }
    #[allow(clippy::manual_is_multiple_of)] // A zero divisor must still panic in `%`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let value = json::parse(
            bytes,
            json::Limits {
                bytes: 1024 * 1024,
                ..Default::default()
            },
        )?;
        let o = value.object()?;
        if o.get("model_type").and_then(|v| v.string().ok()) == Some("phi3") {
            return crate::phi4::parse(&value);
        }
        if o.get("model_type").and_then(|v| v.string().ok()) == Some("llama") { return crate::llama::parse(&value); }
        // Reject new architecture-affecting configuration until explicitly implemented.
        let allowed = [
            "architectures",
            "attention_bias",
            "attention_dropout",
            "bos_token_id",
            "eos_token_id",
            "head_dim",
            "hidden_act",
            "hidden_size",
            "initializer_range",
            "intermediate_size",
            "moe_intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "norm_topk_prob",
            "decoder_sparse_step",
            "mlp_only_layers",
            "output_router_logits",
            "router_aux_loss_coef",
            "max_position_embeddings",
            "max_window_layers",
            "model_type",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_scaling",
            "rope_theta",
            "sliding_window",
            "tie_word_embeddings",
            "torch_dtype",
            "transformers_version",
            "use_cache",
            "use_sliding_window",
            "vocab_size",
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        for key in o.keys() {
            if !allowed.contains(key.as_str()) {
                return Err(invalid(format!("unsupported Qwen3 config field {key}")));
            }
        }
        let architectures = field(o, "architectures")?.array()?;
        if architectures.len() != 1 { return Err(invalid("Qwen3 requires one architecture")); }
        let experts = match (architectures[0].string()?, field(o, "model_type")?.string()?) {
            ("Qwen3ForCausalLM", "qwen3") => false,
            ("Qwen3MoeForCausalLM", "qwen3_moe") => true,
            _ => return Err(invalid("adapter requires Qwen3ForCausalLM or Qwen3MoeForCausalLM")),
        };
        if experts {
            if field(o, "decoder_sparse_step")?.u64()? != 1
                || !field(o, "mlp_only_layers")?.array()?.is_empty() {
                return Err(invalid("Qwen3 MoE requires an expert MLP in every layer"));
            }
            if let Some(v) = o.get("output_router_logits") { v.boolean()?; }
            if let Some(v) = o.get("router_aux_loss_coef") { v.f64()?; }
        }
        if field(o, "torch_dtype")?.string()? != "bfloat16" {
            return Err(invalid("Qwen3 adapter requires bfloat16 checkpoint"));
        }
        if field(o, "hidden_act")?.string()? != "silu" {
            return Err(invalid("Qwen3 adapter only implements silu activation"));
        }
        for name in [
            "attention_bias",
            "use_sliding_window",
        ] {
            if field(o, name)?.boolean()? {
                return Err(invalid(format!("unsupported Qwen3 option {name}=true")));
            }
        }
        if !matches!(field(o, "sliding_window")?, Value::Null) {
            return Err(invalid("unsupported Qwen3 option sliding_window"));
        }
        let yarn = match field(o, "rope_scaling")? {
            Value::Null => false,
            v => {
                let scaling = v.object()?;
                let kind = scaling.get("rope_type").or_else(|| scaling.get("type"))
                    .ok_or_else(|| invalid("missing RoPE scaling type"))?.string()?;
                if kind != "yarn" || field(scaling, "factor")?.f64()? != 4.0
                    || field(scaling, "original_max_position_embeddings")?.u64()? != 32768
                    || scaling.keys().any(|key| !matches!(key.as_str(),
                        "rope_type" | "type" | "factor" | "original_max_position_embeddings")) {
                    return Err(invalid("only official Qwen3 YaRN factor=4 original_max_position_embeddings=32768 is supported"));
                }
                true
            }
        };
        if field(o, "attention_dropout")?.f64()? != 0.0 {
            return Err(invalid(
                "nonzero attention_dropout is outside qualified adapter scope",
            ));
        }
        if !field(o, "use_cache")?.boolean()? {
            return Err(invalid("Qwen3 adapter requires use_cache=true"));
        }
        let integer = |key| field(o, key)?.u64();
        let config = Self {
            qk_norm: true, fused_projections: false, llama_rope: None,
            hidden_size: integer("hidden_size")?,
            intermediate_size: integer("intermediate_size")?,
            moe_intermediate_size: if experts { integer("moe_intermediate_size")? } else { 0 },
            num_experts: if experts { integer("num_experts")? } else { 0 },
            num_experts_per_tok: if experts { integer("num_experts_per_tok")? } else { 0 },
            norm_topk_prob: if experts { field(o, "norm_topk_prob")?.boolean()? } else { false },
            layers: integer("num_hidden_layers")?,
            attention_heads: integer("num_attention_heads")?,
            kv_heads: integer("num_key_value_heads")?,
            head_dim: integer("head_dim")?,
            vocab_size: integer("vocab_size")?,
            max_positions: integer("max_position_embeddings")?,
            bos_token: integer("bos_token_id")?,
            eos_token: integer("eos_token_id")?,
            rms_norm_eps: field(o, "rms_norm_eps")?.f64()?,
            rope_theta: field(o, "rope_theta")?.f64()?,
            yarn,
            tie_word_embeddings: field(o, "tie_word_embeddings")?.boolean()?,
        };
        for (name, n, cap) in [
            ("hidden_size", config.hidden_size, 65_536),
            ("intermediate_size", config.intermediate_size, 1_048_576),
            ("layers", config.layers, 1_024),
            ("attention_heads", config.attention_heads, 1_024),
            ("kv_heads", config.kv_heads, 1_024),
            ("head_dim", config.head_dim, 4_096),
            ("vocab_size", config.vocab_size, 4_194_304),
            ("max_positions", config.max_positions, 1u64 << 32),
        ] {
            if n == 0 || n > cap {
                return Err(invalid(format!(
                    "Qwen3 {name} is zero or exceeds adapter safety bound"
                )));
            }
        }
        if experts && (config.moe_intermediate_size == 0 || config.moe_intermediate_size > 1_048_576
            || config.num_experts == 0 || config.num_experts > 1024
            || config.num_experts_per_tok == 0 || config.num_experts_per_tok > config.num_experts) {
            return Err(invalid("Qwen3 expert dimensions are zero or exceed adapter safety bounds"));
        }
        if config.attention_heads % config.kv_heads != 0 {
            return Err(invalid(
                "Qwen3 query-head count must be divisible by KV-head count",
            ));
        }
        if !config.head_dim.is_multiple_of(2) {
            return Err(invalid("Qwen3 RoPE head dimension must be even"));
        }
        if config.bos_token >= config.vocab_size || config.eos_token >= config.vocab_size {
            return Err(invalid("special token outside vocabulary"));
        }
        if config.rms_norm_eps <= 0.0 || config.rope_theta <= 1.0 {
            return Err(invalid("invalid normalization epsilon or RoPE theta"));
        }
        if let Some(v) = o.get("max_window_layers")
            && v.u64()? > config.layers {
                return Err(invalid("max_window_layers exceeds layer count"));
            }
        if let Some(v) = o.get("initializer_range")
            && v.f64()? < 0.0 {
                return Err(invalid("initializer_range must be nonnegative"));
            }
        if let Some(v) = o.get("transformers_version") {
            v.string()?;
        }
        let mut config = config;
        if yarn { config.enable_yarn()?; }
        Ok(config)
    }

    pub fn is_eos(&self, token: u32) -> bool {
        if self.llama_rope.is_some() { [128001, 128008, 128009].contains(&token) }
        else { token as u64 == self.eos_token }
    }

    /// Negative epsilon selects the official BF16 activation boundaries in shared kernels.
    pub fn norm_epsilon(&self) -> f32 {
        if self.qk_norm { self.rms_norm_eps as f32 } else { -(self.rms_norm_eps as f32) }
    }
    pub fn qk_epsilon(&self) -> f32 {
        if self.qk_norm { self.rms_norm_eps as f32 } else { -1.0 }
    }

    /// Explicitly opt into the model card's 131072-position YaRN extension.
    pub fn enable_yarn(&mut self) -> Result<()> {
        if !matches!(self.max_positions, 40960 | 131072) {
            return Err(invalid("Qwen3 YaRN requires a 40960 or 131072-position checkpoint"));
        }
        self.yarn = true;
        self.max_positions = 131072;
        Ok(())
    }

    /// Adapter-owned tensor contract inferred from the observed checkpoint config
    /// and tensor names, not from a translated third-party runtime.
    pub fn expected_shapes(&self) -> BTreeMap<String, Vec<u64>> {
        let mut expected = BTreeMap::new();
        expected.insert(
            "model.embed_tokens.weight".into(),
            vec![self.vocab_size, self.hidden_size],
        );
        if !self.tie_word_embeddings {
            expected.insert("lm_head.weight".into(), vec![self.vocab_size, self.hidden_size]);
        }
        expected.insert("model.norm.weight".into(), vec![self.hidden_size]);
        for layer in 0..self.layers {
            let base = format!("model.layers.{layer}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                expected.insert(format!("{base}.{name}.weight"), vec![self.hidden_size]);
            }
            for name in if self.qk_norm { &["q_norm", "k_norm"][..] } else { &[] } {
                expected.insert(
                    format!("{base}.self_attn.{name}.weight"),
                    vec![self.head_dim],
                );
            }
            expected.insert(
                format!("{base}.self_attn.q_proj.weight"),
                vec![self.attention_heads * self.head_dim, self.hidden_size],
            );
            for name in ["k_proj", "v_proj"] {
                expected.insert(
                    format!("{base}.self_attn.{name}.weight"),
                    vec![self.kv_heads * self.head_dim, self.hidden_size],
                );
            }
            expected.insert(
                format!("{base}.self_attn.o_proj.weight"),
                vec![self.hidden_size, self.attention_heads * self.head_dim],
            );
            if self.num_experts == 0 {
                for name in ["gate_proj", "up_proj"] {
                    expected.insert(format!("{base}.mlp.{name}.weight"),
                        vec![self.intermediate_size, self.hidden_size]);
                }
                expected.insert(format!("{base}.mlp.down_proj.weight"),
                    vec![self.hidden_size, self.intermediate_size]);
            } else {
                expected.insert(format!("{base}.mlp.gate.weight"), vec![self.num_experts, self.hidden_size]);
                for expert in 0..self.num_experts {
                    let mlp = format!("{base}.mlp.experts.{expert}");
                    for name in ["gate_proj", "up_proj"] {
                        expected.insert(format!("{mlp}.{name}.weight"),
                            vec![self.moe_intermediate_size, self.hidden_size]);
                    }
                    expected.insert(format!("{mlp}.down_proj.weight"),
                        vec![self.hidden_size, self.moe_intermediate_size]);
                }
            }
        }
        if self.fused_projections {
            for layer in 0..self.layers {
                let base = format!("model.layers.{layer}");
                for name in ["q_proj", "k_proj", "v_proj"] {
                    expected.remove(&format!("{base}.self_attn.{name}.weight"));
                }
                expected.insert(format!("{base}.self_attn.qkv_proj.weight"),
                    vec![(self.attention_heads + 2 * self.kv_heads) * self.head_dim, self.hidden_size]);
                for name in ["gate_proj", "up_proj"] {
                    expected.remove(&format!("{base}.mlp.{name}.weight"));
                }
                expected.insert(format!("{base}.mlp.gate_up_proj.weight"),
                    vec![2 * self.intermediate_size, self.hidden_size]);
            }
        }
        expected
    }

    pub fn validate_tensors(&self, headers: &BTreeMap<String, Header>) -> Result<()> {
        let mut expected = self.expected_shapes();
        let mut found = BTreeMap::new();
        for header in headers.values() {
            for (name, tensor) in &header.tensors {
                if found.insert(name, tensor).is_some() {
                    return Err(invalid("duplicate tensor across checkpoint shards"));
                }
            }
        }
        // Some official tied checkpoints retain a redundant serialized head.
        if self.tie_word_embeddings && found.keys().any(|n| n.as_str() == "lm_head.weight") {
            expected.insert("lm_head.weight".into(), vec![self.vocab_size, self.hidden_size]);
        }
        if found.len() != expected.len() {
            return Err(invalid(format!(
                "checkpoint tensor count mismatch: expected {}, observed {}",
                expected.len(),
                found.len()
            )));
        }
        for (name, shape) in expected {
            let tensor = found
                .get(&name)
                .ok_or_else(|| invalid(format!("missing checkpoint tensor {name}")))?;
            if tensor.dtype != Dtype::Bf16 || tensor.shape != shape {
                return Err(invalid(format!(
                    "checkpoint tensor {name} dtype or shape mismatch; expected BF16 {shape:?}, got {:?} {:?}",
                    tensor.dtype, tensor.shape
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Checkpoint {
    pub config: Config,
    pub index: Index,
    pub shards: BTreeMap<String, Header>,
}
impl Checkpoint {
    pub fn inspect(root: &Path) -> Result<Self> {
        let config = Config::open(&root.join("config.json"))?;
        let index_path = root.join("model.safetensors.index.json");
        let index = if index_path.exists() {
            Index::open(&index_path, Limits::default())?
        } else {
            let header = Header::open(&root.join("model.safetensors"), Limits::default())?;
            Index { total_size: header.tensors.values().map(|t| t.byte_len()).sum(), size_declared: true,
                weight_map: header.tensors.keys().map(|n| (n.clone(), "model.safetensors".into())).collect() }
        };
        let shards = index.inspect_shards(root, Limits::default())?;
        config.validate_tensors(&shards)?;
        Ok(Self {
            config,
            index,
            shards,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const OBSERVED: &str = include_str!("../evidence/qwen3-8b-config.json");
    #[test]
    fn observed_qwen3_8b_contract() {
        let c = Config::parse(OBSERVED.as_bytes()).unwrap();
        assert_eq!(c.layers, 36);
        assert_eq!(c.max_positions, 40960);
        let expected = c.expected_shapes();
        assert_eq!(expected.len(), 399);
        assert_eq!(
            expected["model.layers.0.self_attn.k_proj.weight"],
            [1024, 4096]
        );
        let total: u64 = expected
            .values()
            .map(|shape| shape.iter().product::<u64>() * 2)
            .sum();
        assert_eq!(total, 16_381_470_720);
        let index = Index::parse(
            include_bytes!("../evidence/qwen3-8b-index.json"),
            Limits::default(),
        )
        .unwrap();
        assert_eq!(
            index.weight_map.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        );
    }
    #[test]
    fn unsupported_or_malformed_configs() {
        for (from, to) in [
            ("\"qwen3\"", "\"qwen3_next\""),
            ("\"head_dim\": 128", "\"head_dim\": 127"),
            ("\"num_key_value_heads\": 8", "\"num_key_value_heads\": 0"),
            ("\"num_attention_heads\": 32", "\"num_attention_heads\": 31"),
            ("\"rope_scaling\": null", "\"rope_scaling\": {}"),
            ("\"attention_bias\": false", "\"attention_bias\": true"),
            ("\"bfloat16\"", "\"float16\""),
            ("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 0"),
            ("\"eos_token_id\": 151645", "\"eos_token_id\": 151936"),
            (
                "\"num_hidden_layers\": 36",
                "\"num_hidden_layers\": 18446744073709551615",
            ),
        ] {
            assert_ne!(OBSERVED.replace(from, to), OBSERVED);
            assert!(
                Config::parse(OBSERVED.replace(from, to).as_bytes()).is_err(),
                "accepted {to}"
            );
        }
    }
}
