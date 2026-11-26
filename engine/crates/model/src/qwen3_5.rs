//! Text-only Qwen3.5/3.6/3.8 dense and expert hybrid checkpoint contract.
use crate::{Result, invalid, json::{self, Value, field}, safetensors::{self, Header, Index, Tensor}};
use crate::flash::LayerKind;
use std::{collections::BTreeMap, path::Path};

pub const PREFIX: &str = "model.language_model";
#[path = "qwen3_5_plan.rs"]
pub mod plan;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: u64,
    pub intermediate: u64,
    pub experts: u64,
    pub selected_experts: u64,
    pub shared_intermediate: u64,
    pub layers: Vec<LayerKind>,
    pub query_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub linear_key_heads: u64,
    pub linear_value_heads: u64,
    pub linear_key_dim: u64,
    pub linear_value_dim: u64,
    pub linear_conv: u64,
    pub vocab: u64,
    pub max_positions: u64,
    pub bos_token: u64,
    pub eos_token: u64,
    pub epsilon: f64,
    pub rope_theta: f64,
    pub rotary_dim: u64,
    pub mrope_sections: Vec<u64>,
    pub mtp_layers: u64,
    pub mtp_dedicated_embeddings: bool,
    pub tied_embeddings: bool,
    pub fp8: bool,
    pub unquantized: Vec<String>,
}
impl Config {
    pub fn open(path: &Path) -> Result<Self> {
        Self::parse(&safetensors::read_bounded(path, 1024 * 1024)?)
    }
    #[allow(clippy::manual_is_multiple_of)] // A zero divisor must still panic in `%`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let doc = json::parse(bytes, json::Limits::default())?;
        let root = doc.object()?;
        let moe = field(root, "model_type")?.string()? == "qwen3_5_moe";
        let architecture = if moe { "Qwen3_5MoeForConditionalGeneration" } else { "Qwen3_5ForConditionalGeneration" };
        if field(root, "model_type")?.string()? != if moe { "qwen3_5_moe" } else { "qwen3_5" }
            || field(root, "architectures")?.array()? != [Value::String(architecture.into())] {
            return Err(invalid("expected a Qwen3_5 text architecture"));
        }
        let o = field(root, "text_config")?.object()?;
        let rope = field(o, "rope_parameters")?.object()?;
        let quant = root.get("quantization_config").map(Value::object).transpose()?;
        let mut unquantized = Vec::new();
        if let Some(q) = quant {
            if field(q, "quant_method")?.string()? != "fp8"
                || field(q, "fmt")?.string()? != "e4m3"
                || field(q, "activation_scheme")?.string()? != "dynamic"
                || field(q, "weight_block_size")?.array()?.iter().map(Value::u64).collect::<Result<Vec<_>>>()? != [128, 128] {
                return Err(invalid("unsupported qwen3_5 FP8 quantization"));
            }
            unquantized = field(q, "modules_to_not_convert")?.array()?.iter()
                .map(|v| Ok(v.string()?.to_owned())).collect::<Result<Vec<_>>>()?;
        }
        if field(o, "model_type")?.string()? != if moe { "qwen3_5_moe_text" } else { "qwen3_5_text" }
            || field(o, "dtype")?.string()? != "bfloat16"
            || field(o, "hidden_act")?.string()? != "silu"
            || o.get("output_gate_type").is_some_and(|v| v.string().ok() != Some("swish"))
            || !field(o, "attn_output_gate")?.boolean()?
            || field(o, "attention_bias")?.boolean()?
            || field(o, "attention_dropout")?.f64()? != 0.0
            || field(o, "tie_word_embeddings")?.boolean()? != field(root, "tie_word_embeddings")?.boolean()?
            || !field(o, "use_cache")?.boolean()?
            || field(o, "mamba_ssm_dtype")?.string()? != "float32"
            || field(rope, "rope_type")?.string()? != "default"
            || !field(rope, "mrope_interleaved")?.boolean()? {
            return Err(invalid("unsupported qwen3_5 text precision, attention, or RoPE setting"));
        }
        let n = |key| field(o, key)?.u64();
        let partial = o.get("partial_rotary_factor").unwrap_or(field(rope, "partial_rotary_factor")?).f64()?;
        let head_dim = n("head_dim")?;
        let rotary = head_dim as f64 * partial;
        if partial <= 0.0 || partial > 1.0 || rotary.fract() != 0.0
            || o.get("partial_rotary_factor").is_some_and(|v| !v.f64().is_ok_and(|p| p == partial)) {
            return Err(invalid("invalid partial rotary dimension"));
        }
        let layers = field(o, "layer_types")?.array()?.iter().map(|v| match v.string()? {
            "linear_attention" => Ok(LayerKind::DeltaNet),
            "full_attention" => Ok(LayerKind::SparseAttention),
            kind => Err(invalid(format!("unknown qwen3_5 layer {kind}"))),
        }).collect::<Result<Vec<_>>>()?;
        let c = Self {
            hidden: n("hidden_size")?, intermediate: n(if moe { "moe_intermediate_size" } else { "intermediate_size" })?, layers,
            experts: if moe { n("num_experts")? } else { 0 },
            selected_experts: if moe { n("num_experts_per_tok")? } else { 0 },
            shared_intermediate: if moe { n("shared_expert_intermediate_size")? } else { 0 },
            query_heads: n("num_attention_heads")?, kv_heads: n("num_key_value_heads")?, head_dim,
            linear_key_heads: n("linear_num_key_heads")?, linear_value_heads: n("linear_num_value_heads")?,
            linear_key_dim: n("linear_key_head_dim")?, linear_value_dim: n("linear_value_head_dim")?,
            linear_conv: n("linear_conv_kernel_dim")?, vocab: n("vocab_size")?,
            max_positions: n("max_position_embeddings")?,
            bos_token: o.get("bos_token_id").map(Value::u64).transpose()?.unwrap_or(n("eos_token_id")?), eos_token: n("eos_token_id")?,
            epsilon: field(o, "rms_norm_eps")?.f64()?, rope_theta: field(rope, "rope_theta")?.f64()?,
            rotary_dim: rotary as u64,
            mrope_sections: field(rope, "mrope_section")?.array()?.iter().map(Value::u64).collect::<Result<Vec<_>>>()?,
            mtp_layers: n("mtp_num_hidden_layers")?,
            mtp_dedicated_embeddings: o.get("mtp_use_dedicated_embeddings").map(Value::boolean).transpose()?.unwrap_or(false),
            tied_embeddings: field(o, "tie_word_embeddings")?.boolean()?,
            fp8: quant.is_some(), unquantized,
        };
        for (name, value, cap) in [
            ("hidden", c.hidden, 65536), ("intermediate", c.intermediate, 1048576),
            ("query heads", c.query_heads, 1024), ("KV heads", c.kv_heads, 1024),
            ("head dim", c.head_dim, 4096), ("linear key heads", c.linear_key_heads, 1024),
            ("linear value heads", c.linear_value_heads, 1024), ("linear key dim", c.linear_key_dim, 4096),
            ("linear value dim", c.linear_value_dim, 4096), ("linear conv", c.linear_conv, 1024),
            ("vocabulary", c.vocab, 4194304), ("positions", c.max_positions, 1 << 32),
        ] {
            if value == 0 || value > cap { return Err(invalid(format!("invalid qwen3_5 {name}"))); }
        }
        let rope_sections = c.mrope_sections.iter().try_fold(0u64, |n, &s| n.checked_add(s))
            .and_then(|n| n.checked_mul(2)).ok_or_else(|| invalid("rotary section size overflow"))?;
        if c.layers.is_empty() || c.layers.len() > 1024 || c.layers.len() as u64 != n("num_hidden_layers")?
            || c.query_heads % c.kv_heads != 0 || c.linear_value_heads % c.linear_key_heads != 0
            || !c.rotary_dim.is_multiple_of(2) || rope_sections != c.rotary_dim
            || c.epsilon <= 0.0 || c.rope_theta <= 1.0 || c.bos_token >= c.vocab || c.eos_token >= c.vocab {
            return Err(invalid("inconsistent qwen3_5 dimensions or special tokens"));
        }
        if moe && (c.experts == 0 || c.experts > 1024 || c.selected_experts == 0
            || c.selected_experts > c.experts || c.shared_intermediate == 0 || c.shared_intermediate > 1048576) {
            return Err(invalid("invalid qwen3_5 expert dimensions"));
        }
        Ok(c)
    }
    pub fn expected_shapes(&self) -> BTreeMap<String, Vec<u64>> {
        let mut shapes = BTreeMap::new();
        shapes.insert(format!("{PREFIX}.embed_tokens.weight"), vec![self.vocab, self.hidden]);
        if !self.tied_embeddings { shapes.insert("lm_head.weight".into(), vec![self.vocab, self.hidden]); }
        shapes.insert(format!("{PREFIX}.norm.weight"), vec![self.hidden]);
        let key = self.linear_key_heads * self.linear_key_dim;
        let value = self.linear_value_heads * self.linear_value_dim;
        let qkv = 2 * key + value;
        for (i, kind) in self.layers.iter().enumerate() {
            let p = format!("{PREFIX}.layers.{i}");
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                shapes.insert(format!("{p}.{norm}.weight"), vec![self.hidden]);
            }
            if self.experts == 0 {
                for projection in ["gate_proj", "up_proj"] {
                    shapes.insert(format!("{p}.mlp.{projection}.weight"), vec![self.intermediate, self.hidden]);
                }
                shapes.insert(format!("{p}.mlp.down_proj.weight"), vec![self.hidden, self.intermediate]);
            } else {
                shapes.insert(format!("{p}.mlp.gate.weight"), vec![self.experts, self.hidden]);
                shapes.insert(format!("{p}.mlp.experts.gate_up_proj"), vec![self.experts, 2 * self.intermediate, self.hidden]);
                shapes.insert(format!("{p}.mlp.experts.down_proj"), vec![self.experts, self.hidden, self.intermediate]);
                for projection in ["gate_proj", "up_proj"] {
                    shapes.insert(format!("{p}.mlp.shared_expert.{projection}.weight"), vec![self.shared_intermediate, self.hidden]);
                }
                shapes.insert(format!("{p}.mlp.shared_expert.down_proj.weight"), vec![self.hidden, self.shared_intermediate]);
                shapes.insert(format!("{p}.mlp.shared_expert_gate.weight"), vec![1, self.hidden]);
            }
            let parts = match kind {
                LayerKind::DeltaNet => vec![
                    ("A_log", vec![self.linear_value_heads]), ("dt_bias", vec![self.linear_value_heads]),
                    ("conv1d.weight", vec![qkv, 1, self.linear_conv]),
                    ("in_proj_qkv.weight", vec![qkv, self.hidden]), ("in_proj_z.weight", vec![value, self.hidden]),
                    ("in_proj_a.weight", vec![self.linear_value_heads, self.hidden]),
                    ("in_proj_b.weight", vec![self.linear_value_heads, self.hidden]),
                    ("norm.weight", vec![self.linear_value_dim]), ("out_proj.weight", vec![self.hidden, value]),
                ],
                LayerKind::SparseAttention => vec![
                    ("q_proj.weight", vec![2 * self.query_heads * self.head_dim, self.hidden]),
                    ("k_proj.weight", vec![self.kv_heads * self.head_dim, self.hidden]),
                    ("v_proj.weight", vec![self.kv_heads * self.head_dim, self.hidden]),
                    ("o_proj.weight", vec![self.hidden, self.query_heads * self.head_dim]),
                    ("q_norm.weight", vec![self.head_dim]), ("k_norm.weight", vec![self.head_dim]),
                ],
            };
            let attention = if *kind == LayerKind::DeltaNet { "linear_attn" } else { "self_attn" };
            for (part, shape) in parts { shapes.insert(format!("{p}.{attention}.{part}"), shape); }
        }
        shapes
    }
    pub fn weight_bytes(&self) -> u64 {
        self.expected_shapes().iter().map(|(name, s)| {
            let elements = s.iter().product::<u64>();
            if self.quantized(name, s) { elements + s[0].div_ceil(128) * s[1].div_ceil(128) * 2 }
            else { elements * 2 }
        }).sum()
    }
    pub fn quantized(&self, name: &str, shape: &[u64]) -> bool {
        self.fp8 && shape.len() == 2 && name.ends_with(".weight")
            && !name.ends_with("embed_tokens.weight")
            && !self.unquantized.iter().any(|s| name.strip_suffix(".weight")
                .is_some_and(|m| m == s || m.strip_prefix(s).is_some_and(|tail| tail.starts_with('.'))))
    }
    pub fn stored_shapes(&self) -> BTreeMap<String, Vec<u64>> {
        let mut shapes = self.expected_shapes();
        for (name, shape) in self.expected_shapes() {
            if self.quantized(&name, &shape) {
                shapes.insert(format!("{name}_scale_inv"), vec![shape[0].div_ceil(128), shape[1].div_ceil(128)]);
            }
        }
        shapes
    }
    pub fn mtp_shapes(&self) -> Result<BTreeMap<String, Vec<u64>>> {
        if self.experts != 0 { return Err(invalid("hybrid MoE MTP is not implemented")); }
        if self.mtp_layers != 1 || self.mtp_dedicated_embeddings {
            return Err(invalid("MTP requires one layer and shared embeddings"));
        }
        let mut shapes = BTreeMap::new();
        shapes.insert("mtp.fc.weight".into(), vec![self.hidden, 2 * self.hidden]);
        for norm in ["norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden",
            "layers.0.input_layernorm", "layers.0.post_attention_layernorm"] {
            shapes.insert(format!("mtp.{norm}.weight"), vec![self.hidden]);
        }
        for (name, shape) in [
            ("q_proj", vec![2 * self.query_heads * self.head_dim, self.hidden]),
            ("k_proj", vec![self.kv_heads * self.head_dim, self.hidden]),
            ("v_proj", vec![self.kv_heads * self.head_dim, self.hidden]),
            ("o_proj", vec![self.hidden, self.query_heads * self.head_dim]),
            ("q_norm", vec![self.head_dim]), ("k_norm", vec![self.head_dim]),
        ] { shapes.insert(format!("mtp.layers.0.self_attn.{name}.weight"), shape); }
        for name in ["gate_proj", "up_proj"] {
            shapes.insert(format!("mtp.layers.0.mlp.{name}.weight"), vec![self.intermediate, self.hidden]);
        }
        shapes.insert("mtp.layers.0.mlp.down_proj.weight".into(), vec![self.hidden, self.intermediate]);
        Ok(shapes)
    }
    pub fn mtp_weight_bytes(&self) -> Result<u64> {
        Ok(self.mtp_shapes()?.values().map(|s| s.iter().product::<u64>() * 2).sum())
    }
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.layers.iter().filter(|&&k| k == LayerKind::SparseAttention).count() as u64
            * 2 * self.kv_heads * self.head_dim * 2
    }
    pub fn state_bytes(&self) -> u64 {
        let heads = self.linear_value_heads;
        let conv = 2 * self.linear_key_heads * self.linear_key_dim + heads * self.linear_value_dim;
        self.layers.iter().filter(|&&k| k == LayerKind::DeltaNet).count() as u64
            * (heads * self.linear_key_dim * self.linear_value_dim * 4 + conv * self.linear_conv * 2)
    }
}

pub struct Checkpoint {
    pub config: Config,
    pub index: Index,
    pub shards: BTreeMap<String, Header>,
    pub text_bytes: u64,
}
impl Checkpoint {
    pub fn inspect(root: &Path) -> Result<Self> {
        plan::inspect(root, false).map(|(_, checkpoint)| checkpoint).map_err(invalid)
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        let shard = self.index.weight_map.get(name).ok_or_else(|| invalid(format!("missing text tensor {name}")))?;
        self.shards[shard].tensors.get(name).ok_or_else(|| invalid(format!("missing tensor header {name}")))
    }
}
