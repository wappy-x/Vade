//! Identify a cartridge using bounded metadata, before any tokenizer or GPU work.
use crate::{Result, invalid, json, qwen3, gemma4, safetensors::{self, Header, Index, Limits}};
use std::path::Path;

pub const SUPPORTED: &str = "Gemma 4 dense text BF16, Qwen3.5/3.8 dense hybrid BF16 and official FP8, Qwen3.6 hybrid MoE BF16 and official FP8, Qwen3 dense BF16 and Qwen3 MoE BF16, Phi-4 BF16, Llama 3.1/3.2 BF16 (official checkpoints; Llama supports declared tied embeddings)";

pub struct Cartridge {
    pub kind: Kind,
    pub skipped_bytes: u64, pub state_bytes: u64, pub kv_bytes_per_token: u64, pub max_positions: u64,
    pub architecture: String,
    pub precision: String,
    pub weight_bytes: u64,
    pub config: Option<qwen3::Config>,
    pub reason: Option<String>,
    pub gemma4: Option<gemma4::Config>,
    pub weights_complete: bool,
}

impl Cartridge {
    /// Includes bounded execution/staging reserve; driver and OS stay outside this budget.
    pub fn fit(&self, context_tokens: u64, budget_bytes: u64) -> Result<Fit> {
        if context_tokens == 0 || context_tokens > self.max_positions { return Err(invalid(engine_base::admission::token_refusal(context_tokens as u128, self.max_positions as usize, &format!("CONTEXT={context_tokens}; OUTPUT included in CONTEXT; context must be positive"), "the cartridge position ceiling", engine_base::admission::SERVE))); }
        let kv_bytes = if let Some(config) = &self.gemma4 { config.kv_bytes(context_tokens)? }
            else { context_tokens.checked_mul(self.kv_bytes_per_token).ok_or_else(|| invalid("KV size overflow"))? };
        let scratch_reserve_bytes = 256 * 1024 * 1024;
        let required_bytes = self.weight_bytes.checked_add(self.state_bytes).and_then(|n| n.checked_add(kv_bytes))
            .and_then(|n| n.checked_add(scratch_reserve_bytes)).ok_or_else(|| invalid("cartridge memory overflow"))?;
        Ok(Fit { context_tokens, kv_bytes, scratch_reserve_bytes, required_bytes, budget_bytes, fits: required_bytes <= budget_bytes })
    }

    fn identify(root: &Path) -> Result<Self> {
        let bytes = safetensors::read_bounded(&root.join("config.json"), 1024 * 1024)?;
        let doc = json::parse(&bytes, json::Limits { bytes: 1024 * 1024, ..Default::default() })?;
        let object = doc.object()?;
        let text = object.get("text_config").map(|v| v.object()).transpose()?.unwrap_or(object);
        let string = |o: &std::collections::BTreeMap<String, json::Value>, key: &str|
            o.get(key).and_then(|v| v.string().ok()).unwrap_or("").to_owned();
        let model_type = string(object, "model_type");
        let classes = object.get("architectures").map(|v| v.array()).transpose()?
            .unwrap_or(&[]).iter().map(|v| v.string()).collect::<Result<Vec<_>>>()?;
        let dense = model_type == "qwen3" && classes.as_slice() == ["Qwen3ForCausalLM"];
        let moe = model_type == "qwen3_moe" && classes.as_slice() == ["Qwen3MoeForCausalLM"];
        let phi = model_type == "phi3" && classes.as_slice() == ["Phi3ForCausalLM"];
        let llama = model_type == "llama" && classes.as_slice() == ["LlamaForCausalLM"];
        let architecture = if llama { "Llama dense (LlamaForCausalLM)".into() } else if phi { "Phi-4 dense (Phi3ForCausalLM)".into() } else if dense { "Qwen3 dense (Qwen3ForCausalLM)".into() }
            else if moe { "Qwen3 MoE (Qwen3MoeForCausalLM)".into() }
            else if model_type == "gemma4" { "Gemma 4 text (Gemma4ForConditionalGeneration)".into() }
            else if model_type == "qwen4_exp" || model_type == "qwen4_exp_text" {
                "Qwen3.8 Flash Next (qwen4_exp)".into()
            } else {
                format!("{} ({})", if classes.is_empty() { "unknown".into() } else { classes.join(",") },
                    if model_type.is_empty() { "unknown model_type" } else { &model_type })
            };
        let dtype = [string(text, "dtype"), string(text, "torch_dtype"),
            string(object, "dtype"), string(object, "torch_dtype")]
            .into_iter().find(|s| !s.is_empty()).unwrap_or_else(|| "unknown".into());
        let mut precision = match dtype.as_str() {
            "bfloat16" => "BF16".into(), "float16" => "FP16".into(),
            "float32" => "FP32".into(), _ => dtype,
        };
        if let Some(quant) = object.get("quantization_config").or_else(|| text.get("quantization_config")) {
            if !matches!(quant, json::Value::Null) {
                let method = string(quant.object()?, "quant_method");
                precision = if method.is_empty() { "quantized (unknown)".into() } else { method.to_ascii_uppercase() };
            }
        }
        let (config, reason) = if dense || moe || phi || llama {
            match qwen3::Config::parse(&bytes) {
                Ok(c) => match unsupported_geometry(&c) {
                    Some(reason) => (None, Some(reason)), None => (Some(c), None),
                },
                Err(e) => (None, Some(e.to_string())),
            }
        } else { (None, Some("architecture has no spark-serve backend".into())) };
        let (gemma4, reason) = if model_type == "gemma4" {
            match gemma4::Config::parse(&bytes) {
                Ok(c) => (Some(c), None),
                Err(e) => (None, Some(e.to_string())),
            }
        } else { (None, reason) };
        Ok(Self { kind: if model_type == "gemma4" { Kind::Gemma4 } else { Kind::Qwen3 }, skipped_bytes: 0, state_bytes: 0,
            kv_bytes_per_token: config.as_ref().map_or(0, |c| c.layers * c.kv_heads * c.head_dim * 4),
            max_positions: config.as_ref().map_or_else(|| gemma4.as_ref().map_or(0, |c| c.max_positions), |c| c.max_positions),
            gemma4, weights_complete: false, architecture, precision, weight_bytes: 0, config, reason })
    }

    pub fn inspect(root: &Path) -> Result<Self> {
        if Kind::detect(root).ok() == Some(Kind::Qwen3_5) {
            let c = crate::qwen3_5::Checkpoint::inspect(root)?;
            return Ok(Self { kind: Kind::Qwen3_5, architecture: if c.config.experts == 0 { "Qwen3.5/3.8 dense hybrid" } else { "Qwen3.6 hybrid MoE" }.into(),
                precision: if c.config.fp8 { "FP8" } else { "BF16" }.into(), weight_bytes: c.text_bytes,
                skipped_bytes: c.index.total_size - c.text_bytes, state_bytes: c.config.state_bytes(),
                kv_bytes_per_token: c.config.kv_bytes_per_token(), max_positions: c.config.max_positions,
                config: None, reason: None, gemma4: None, weights_complete: false });
        }
        Self::identify(root)?.inspect_weights(root)
    }

    fn inspect_weights(self, root: &Path) -> Result<Self> {
        let mut cartridge = self;
        let index_path = root.join("model.safetensors.index.json");
        if index_path.exists() {
            let index = match Index::open(&index_path, Limits {
                header_bytes: 64 * 1024 * 1024, tensors: 200_000, ..Limits::default()
            }) {
                Ok(index) => index,
                Err(error) => {
                    cartridge.config = None;
                    cartridge.reason = Some(format!("invalid safetensors index: {error}"));
                    return Ok(cartridge);
                }
            };
            cartridge.weight_bytes = index.total_size;
            if let Some(config) = &cartridge.gemma4 {
                let expected = config.expected_shapes();
                let validation = (|| -> Result<()> {
                    let mut seen = std::collections::BTreeSet::new();
                    for header in index.inspect_shards(root, Limits::default())?.values() {
                        for (name, tensor) in &header.tensors {
                            if !name.starts_with("model.language_model.") && name != "lm_head.weight" { continue; }
                            if !seen.insert(name.clone()) || expected.get(name) != Some(&tensor.shape) || tensor.dtype != safetensors::Dtype::Bf16 {
                                return Err(invalid(format!("Gemma 4 text tensor mismatch: {name}")));
                            }
                        }
                    }
                    if seen.len() != expected.len() { return Err(invalid("missing Gemma 4 text tensors")); }
                    Ok(())
                })();
                match validation {
                    Ok(()) => cartridge.weights_complete = true,
                    Err(e) => cartridge.reason = Some(format!("Gemma 4 weights incomplete or invalid: {e}")),
                }
            }
            let experts = index.weight_map.keys().any(|name| name.contains(".mlp.experts."));
            let fp8 = index.weight_map.keys().any(|name| name.ends_with(".weight_scale_inv"));
            if fp8 { cartridge.precision = "FP8".into(); }
            if cartridge.config.as_ref().is_some_and(|c| experts != (c.num_experts != 0) || fp8) {
                cartridge.config = None;
                cartridge.reason = Some("config and safetensors index disagree on architecture or BF16 precision".into());
            }
        } else if root.join("model.safetensors").is_file() {
            let header = Header::open(&root.join("model.safetensors"), Limits::default())?;
            cartridge.weight_bytes = header.tensors.values().map(|t| t.byte_len()).sum();
            if header.tensors.values().any(|t| t.dtype == safetensors::Dtype::F8E4M3) {
                cartridge.precision = "FP8".into();
                cartridge.config = None;
                cartridge.reason = Some("FP8 Qwen3 serving is not implemented".into());
            } else if let Some(config) = &cartridge.config {
                let shards = [("model.safetensors".into(), header)].into_iter().collect();
                if let Err(e) = config.validate_tensors(&shards) {
                    cartridge.config = None;
                    cartridge.reason = Some(e.to_string());
                }
            }
        } else if cartridge.config.is_some() {
            cartridge.config = None;
            cartridge.reason = Some("checkpoint has no safetensors weights/index".into());
        }
        Ok(cartridge)
    }

    pub fn require_supported(&self) -> Result<&qwen3::Config> {
        self.config.as_ref().ok_or_else(|| invalid(format!(
            "found architecture={}, precision={}; not supported yet: {}. Supported: {SUPPORTED}",
            self.architecture, self.precision, self.reason.as_deref().unwrap_or("unsupported checkpoint")
        )))
    }

    /// Inspect headers only. Unsupported cartridges never reach CUDA or read payloads.
    pub fn preflight(root: &Path) -> Result<Self> {
        if Kind::detect(root).ok() == Some(Kind::Qwen3_5) { return Self::inspect(root); }
        // Reject architecture/configuration before touching a possibly huge index.
        let cartridge = Self::identify(root)?;
        if cartridge.gemma4.is_some() {
            let cartridge = cartridge.inspect_weights(root)?;
            if !cartridge.weights_complete { return Err(invalid(cartridge.reason.as_deref().unwrap_or("Gemma requires an indexed complete BF16 text checkpoint"))); }
            return Ok(cartridge);
        }
        cartridge.require_supported()?;
        let cartridge = cartridge.inspect_weights(root)?;
        cartridge.require_supported()?;
        qwen3::Checkpoint::inspect(root).map_err(|e| invalid(format!(
            "found architecture={}, precision={}; cannot start checkpoint: {e}. Supported: {SUPPORTED}",
            cartridge.architecture, cartridge.precision
        )))?;
        Ok(cartridge)
    }
}

fn unsupported_geometry(config: &qwen3::Config) -> Option<String> {
    // Tied Qwen embeddings remain draft-only; official Llama uses one shared matrix.
    if config.tie_word_embeddings && config.llama_rope.is_none() {
        return Some("tied embeddings are supported only as a draft model".into());
    }
    let per_layer = 8 + if config.num_experts == 0 { 3 } else { 1 + 3 * config.num_experts };
    let tensors = config.layers.checked_mul(per_layer).and_then(|n| n.checked_add(3));
    if tensors.is_none_or(|n| n > Limits::default().tensors as u64) {
        return Some("checkpoint exceeds supported tensor count".into());
    }
    for (name, shape) in config.expected_shapes() {
        if shape.len() == 2 && name != "model.embed_tokens.weight"
            && (shape[0] % 8 != 0 || shape[1] % 16 != 0) {
            return Some(format!("tensor {name} shape {shape:?}: packed projections require rows divisible by 8 and columns divisible by 16"));
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind { Qwen3, FlashNext, Qwen3_5, Gemma4 }
impl Kind {
    pub fn name(self) -> &'static str {
        match self { Self::Qwen3 => "qwen3", Self::FlashNext => "flash-next", Self::Qwen3_5 => "qwen3_5", Self::Gemma4 => "gemma4" }
    }
    pub fn detect(root: &Path) -> Result<Self> {
        let bytes = safetensors::read_bounded(&root.join("config.json"), 1024 * 1024)?;
        let doc = json::parse(&bytes, json::Limits::default())?;
        match crate::json::field(doc.object()?, "model_type")?.string()? {
            "qwen3" | "qwen3_moe" | "phi3" | "llama" => Ok(Self::Qwen3),
            "qwen4_exp" => Ok(Self::FlashNext),
            "qwen3_5" | "qwen3_5_moe" => Ok(Self::Qwen3_5),
            "gemma4" => Ok(Self::Gemma4),
            other => Err(invalid(format!("unsupported cartridge model_type {other}"))),
        }
    }
}


pub struct Fit {
    pub context_tokens: u64,
    pub kv_bytes: u64,
    pub scratch_reserve_bytes: u64,
    pub required_bytes: u64,
    pub budget_bytes: u64,
    pub fits: bool,
}
