//! Qwen3.8-Flash-Next's observed checkpoint contract. No GPU/runtime dependency.
pub mod dump;
pub mod manifest;
pub mod mixed;
pub mod mxfp8;
pub mod nvfp4;
pub mod plan;
pub mod refcpu;
pub mod shard;
use crate::{
    Result, invalid,
    json::{self, Value, field},
    safetensors::{self, Dtype, Header, Index, Limits, Tensor},
};
use std::{collections::BTreeMap, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    DeltaNet,
    SparseAttention,
}
impl LayerKind {
    pub fn profile_name(self) -> &'static str { match self { Self::DeltaNet => "gdn", Self::SparseAttention => "qsa" } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat { Fp8, Nvfp4 }
impl WeightFormat {
    pub fn name(self) -> &'static str {
        match self { Self::Fp8 => "FP8", Self::Nvfp4 => "NVFP4" }
    }
    /// Loading a lower-precision checkpoint is always an explicit choice.
    pub fn require_opt_in(self, enabled: bool) -> Result<()> {
        if self == Self::Nvfp4 && !enabled {
            return Err(invalid("NVFP4 checkpoint requires FLASH_NVFP4=1; FP8 remains the default"));
        }
        Ok(())
    }
    pub fn opt_in() -> Result<bool> {
        match std::env::var("FLASH_NVFP4") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(v) if v == "0" => Ok(false),
            Ok(v) if v == "1" => Ok(true),
            _ => Err(invalid("FLASH_NVFP4 must be 0 or 1")),
        }
    }
}

fn quantization_format(quant: &BTreeMap<String, Value>) -> Result<WeightFormat> {
    match field(quant, "quant_method")?.string()? {
        "fp8" => Ok(WeightFormat::Fp8),
        "modelopt" => {
            if quant.contains_key("config_groups") { nvfp4::validate_config(quant)?; }
            else { mixed::Mixed::parse(quant)?; }
            Ok(WeightFormat::Nvfp4)
        }
        _ => Err(invalid("unsupported Flash quantization method")),
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub format: WeightFormat,
    pub mixed: Option<crate::plan::ModelPlan>,
    pub hidden: u64,
    pub layers: Vec<LayerKind>,
    pub query_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub linear_key_heads: u64,
    pub linear_value_heads: u64,
    pub linear_key_dim: u64,
    pub linear_value_dim: u64,
    pub linear_conv: u64,
    pub experts: u64,
    pub top_k: u64,
    pub renormalize_top_k: bool,
    pub expert_width: u64,
    pub shared_width: u64,
    pub branches: u64,
    pub mix_rank: u64,
    pub vocab: u64,
    pub max_positions: u64,
    pub eos_token: u64,
    pub epsilon: f64,
    pub rope_theta: f64,
    pub rotary_dim: u64,
    pub mrope_sections: Vec<u64>,
    pub index_heads: u64,
    pub index_dim: u64,
    pub index_budget: u64,
    pub index_compression: u64,
    /// Zero-based decoder layer ids (the config stores one-based ids).
    pub ple_layers: Vec<u64>,
    pub ngram_size: u64,
    pub ngram_heads: u64,
    pub ple_dim: u64,
    pub ple_conv: u64,
    pub ngram_parts: u64,
    pub mtp_layers: u64,
    pub block: [u64; 2],
}
impl Config {
    pub fn open(path: &Path) -> Result<Self> {
        Self::parse(&safetensors::read_bounded(path, 1024 * 1024)?)
    }
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let doc = json::parse(bytes, json::Limits::default())?;
        let root = doc.object()?;
        let o = field(root, "text_config")?.object()?;
        let n = |key| field(o, key)?.u64();
        let rope = field(o, "rope_parameters")?.object()?;
        let quant = field(root, "quantization_config")?.object()?;
        let require = |ok, name| {
            if ok {
                Ok(())
            } else {
                Err(invalid(format!("unsupported Flash setting: {name}")))
            }
        };
        require(
            field(root, "model_type")?.string()? == "qwen4_exp",
            "model_type",
        )?;
        require(
            field(o, "model_type")?.string()? == "qwen4_exp_text",
            "text model_type",
        )?;
        require(field(o, "hidden_act")?.string()? == "silu", "hidden_act")?;
        require(
            field(o, "output_gate_type")?.string()? == "sigmoid",
            "output_gate_type",
        )?;
        require(
            field(o, "mamba_ssm_dtype")?.string()? == "float32",
            "mamba_ssm_dtype",
        )?;
        require(
            !field(o, "attention_bias")?.boolean()?
                && !field(o, "tie_word_embeddings")?.boolean()?,
            "bias/tied embeddings",
        )?;
        require(
            field(rope, "rope_type")?.string()? == "default"
                && field(rope, "mrope_interleaved")?.boolean()?,
            "RoPE",
        )?;
        let format = quantization_format(quant)?;
        if format == WeightFormat::Fp8 {
            require(
                field(quant, "activation_scheme")?.string()? == "dynamic"
                    && !field(quant, "weight_per_tensor")?.boolean()?
                    && !field(quant, "act_per_tensor")?.boolean()?,
                "quantization",
            )?;
        }
        require(
            !field(o, "mtp_use_dedicated_embeddings")?.boolean()?,
            "MTP embeddings",
        )?;
        let layers = field(o, "layer_types")?
            .array()?
            .iter()
            .map(|v| match v.string()? {
                "linear_attention" => Ok(LayerKind::DeltaNet),
                "full_attention" => Ok(LayerKind::SparseAttention),
                name => Err(invalid(format!("unknown Flash layer {name}"))),
            })
            .collect::<Result<Vec<_>>>()?;
        if format == WeightFormat::Fp8 {
            let block = field(quant, "weight_block_size")?
                .array()?
                .iter()
                .map(Value::u64)
                .collect::<Result<Vec<_>>>()?;
            require(block == [128, 128], "FP8 block size")?;
        }
        let ple_layers = field(o, "ple_layer_ids")?
            .array()?
            .iter()
            .map(|v| {
                v.u64()?
                    .checked_sub(1)
                    .ok_or_else(|| invalid("PLE layer ids must be one-based"))
            })
            .collect::<Result<Vec<_>>>()?;
        let partial = field(o, "partial_rotary_factor")?.f64()?;
        require(
            partial == field(rope, "partial_rotary_factor")?.f64()?,
            "partial rotary factors differ",
        )?;
        let result = Self {
            format,
            mixed: if format == WeightFormat::Nvfp4 && !quant.contains_key("config_groups") {
                Some(crate::plan::ModelPlan::from_config(bytes).map_err(|e| invalid(e.to_string()))?)
            } else { None },
            hidden: n("hidden_size")?,
            layers,
            query_heads: n("num_attention_heads")?,
            kv_heads: n("num_key_value_heads")?,
            head_dim: n("head_dim")?,
            linear_key_heads: n("linear_num_key_heads")?,
            linear_value_heads: n("linear_num_value_heads")?,
            linear_key_dim: n("linear_key_head_dim")?,
            linear_value_dim: n("linear_value_head_dim")?,
            linear_conv: n("linear_conv_kernel_dim")?,
            experts: n("num_experts")?,
            top_k: n("num_experts_per_tok")?,
            renormalize_top_k: o
                .get("norm_topk_prob")
                .map(Value::boolean)
                .transpose()?
                .unwrap_or(true),
            expert_width: n("moe_intermediate_size")?,
            shared_width: n("shared_expert_intermediate_size")?,
            branches: n("hc_count")?,
            mix_rank: n("hc_lowrank")?,
            vocab: n("vocab_size")?,
            max_positions: n("max_position_embeddings")?,
            eos_token: n("eos_token_id")?,
            epsilon: field(o, "rms_norm_eps")?.f64()?,
            rope_theta: field(rope, "rope_theta")?.f64()?,
            rotary_dim: (n("head_dim")? as f64 * partial) as u64,
            mrope_sections: field(rope, "mrope_section")?
                .array()?
                .iter()
                .map(Value::u64)
                .collect::<Result<Vec<_>>>()?,
            index_heads: n("indexer_n_heads")?,
            index_dim: n("indexer_head_dim")?,
            index_budget: n("indexer_budget")?,
            index_compression: n("indexer_compress_ratio")?,
            ple_layers,
            ngram_size: n("ngram_size")?,
            ngram_heads: n("heads_per_ngram")?,
            ple_dim: n("ple_embed_dim")?,
            ple_conv: n("ple_conv_kernel_size")?,
            ngram_parts: n("split_ngram_parts")?,
            mtp_layers: n("mtp_num_hidden_layers")?,
            block: [128, 128],
        };
        // This adapter deliberately accepts one measured architecture, not guessed variants.
        require(
            result.hidden == 2560 && result.layers.len() == 48 && n("num_hidden_layers")? == 48,
            "model dimensions",
        )?;
        require(
            result.query_heads == 24 && result.kv_heads == 2 && result.head_dim == 256,
            "attention dimensions",
        )?;
        require(
            (
                result.linear_key_heads,
                result.linear_value_heads,
                result.linear_key_dim,
                result.linear_value_dim,
                result.linear_conv,
            ) == (16, 48, 128, 128, 4),
            "DeltaNet dimensions",
        )?;
        require(
            (
                result.experts,
                result.top_k,
                result.expert_width,
                result.shared_width,
            ) == (512, 10, 640, 640),
            "MoE dimensions",
        )?;
        require(result.renormalize_top_k, "router normalization")?;
        require(
            (result.branches, result.mix_rank, result.vocab) == (4, 320, 248320),
            "HC/vocab dimensions",
        )?;
        require(
            result.rotary_dim == 64 && partial == 0.25 && result.mrope_sections == [11, 11, 10],
            "rotary dimensions",
        )?;
        require(
            (
                result.index_heads,
                result.index_dim,
                result.index_budget,
                result.index_compression,
                n("indexer_kv_heads")?,
            ) == (4, 128, 2048, 4, 1),
            "indexer dimensions",
        )?;
        require(
            result.ple_layers == [1]
                && (
                    result.ngram_size,
                    result.ngram_heads,
                    result.ple_dim,
                    result.ple_conv,
                    result.ngram_parts,
                ) == (3, 8, 2560, 4, 128),
            "PLE dimensions",
        )?;
        let mtp = field(o, "mtp")?.object()?;
        require(
            result.mtp_layers == 1
                && field(mtp, "num_hidden_layers")?.u64()? == 1
                && field(mtp, "layer_types")?.array()? == [Value::String("full_attention".into())],
            "MTP layers",
        )?;
        require(
            field(mtp, "hybrid")?.boolean()?
                && field(mtp, "mtp_use_hidden_state_from_layer")? == &Value::Null
                && field(mtp, "rope_theta")?.f64()? == result.rope_theta,
            "MTP state/RoPE",
        )?;
        require(
            result.layers.iter().enumerate().all(|(i, k)| {
                *k == if i % 4 == 3 {
                    LayerKind::SparseAttention
                } else {
                    LayerKind::DeltaNet
                }
            }),
            "layer schedule",
        )?;
        require(
            result.epsilon > 0. && result.rope_theta > 1. && result.max_positions > 0,
            "norm/context",
        )?;
        Ok(result)
    }
}

/// The chunk-form GDN state (S plus the open chunk's tail rows) is the default
/// with a fixed chunk size: its switch-on path passed the two-Spark gate
/// (bytegate streams alone == batched, deterministic across processes and
/// boxes) on the measured build. VADE_GDN_CHUNK=0 selects the previous sequential
/// scan; VADE_GDN_CHUNK=1 (the old opt-in spelling) is accepted and means on.
/// Read once per process; the engine and the memory plan share this switch.
pub fn gdn_chunk_form() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VADE_GDN_CHUNK").is_ok_and(|v| v == "0"))
}
/// Extra state bytes of one chunk-form GDN layer: 64 tail key rows per key
/// head, 64 v' rows and 64 gamma values per value head (zero when off).
pub fn gdn_chunk_tail_bytes(value_heads: u64, key_heads: u64, key_dim: u64, value_dim: u64) -> u64 {
    if !gdn_chunk_form() { return 0; }
    (key_heads * 64 * key_dim + value_heads * 64 * value_dim + value_heads * 64) * 4
}
pub const PLE_ROWS: u64 = 320_001_536;
pub const PLE_PART_ROWS: u64 = 2_500_012;
pub const PLE_ROW_DIM: u64 = 160;
pub const PLE_PREFIX: &str = "model.language_model.layers.1.ple.ple_embedding";

pub struct Checkpoint {
    pub config: Config,
    pub index: Index,
    pub shards: BTreeMap<String, Header>,
    manifest: std::sync::OnceLock<std::result::Result<manifest::Verification, String>>,
}
/// Shared by the loader and serving receipts for official Flash indexes.
pub const CHECKPOINT_INDEX_LIMITS: Limits = Limits {
    header_bytes: 64 << 20,
    tensors: 200_000,
    rank: 16,
    shards: 1024,
};
pub const fn checkpoint_limits(format: WeightFormat, mixed: bool) -> Limits {
    Limits {
        tensors: if mixed && matches!(format, WeightFormat::Nvfp4) {
            mixed::MAX_MIXED_TENSORS
        } else {
            match format {
                WeightFormat::Fp8 => CHECKPOINT_INDEX_LIMITS.tensors,
                WeightFormat::Nvfp4 => crate::shapes::MAX_TENSORS,
            }
        },
        ..CHECKPOINT_INDEX_LIMITS
    }
}

impl Checkpoint {
    pub fn mixed_plan(&self, world: u64) -> Result<mixed::Plan> {
        if world != 1 { return Err(invalid("mixed Flash storage currently requires world 1")); }
        let model = self.config.mixed.clone().ok_or_else(|| invalid("expected mixed checkpoint"))?;
        let headers = self.tensors().map(|(n, t)| (n.to_owned(), t.clone())).collect();
        let model = model.with_headers(&headers).map_err(|e| invalid(e.to_string()))?;
        mixed::Plan::from_model(model, self.index.weight_map.clone())
    }
    pub fn inspect(root: &Path) -> Result<Self> {
        let config = Config::open(&root.join("config.json"))?;
        if config.mixed.is_none() && nvfp4::detect(root)? != config.format {
            return Err(invalid(
                "config.json and hf_quant_config.json disagree on Flash weight format",
            ));
        }
        let limits = checkpoint_limits(config.format, config.mixed.is_some());
        let path = root.join("model.safetensors.index.json");
        let index = if config.mixed.is_some() {
            Index::parse_mixed(
                &crate::safetensors::read_bounded(&path, limits.header_bytes)?,
                limits,
            )?
        } else {
            Index::open(&path, limits)?
        };
        let shards = index.inspect_shards(root, limits)?;
        let result = Self {
            config,
            index,
            shards,
            manifest: std::sync::OnceLock::new(),
        };
        result.validate_tensors()?;
        Ok(result)
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        let shard = self
            .index
            .weight_map
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        Ok(&self.shards[shard].tensors[name])
    }
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.shards
            .values()
            .flat_map(|h| h.tensors.iter().map(|(n, t)| (n.as_str(), t)))
    }
    /// A bounded payload read, for small constants or a row; never loads a shard.
    pub fn read_range(
        &self,
        root: &Path,
        name: &str,
        offset: u64,
        output: &mut [u8],
    ) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let tensor = self.tensor(name)?;
        if offset
            .checked_add(output.len() as u64)
            .is_none_or(|end| end > tensor.byte_len())
        {
            return Err(invalid("Flash read exceeds tensor"));
        }
        let shard = &self.index.weight_map[name];
        let root = root.canonicalize()?;
        let path = root.join(shard).canonicalize()?;
        if !path.starts_with(root) {
            return Err(invalid("shard escapes checkpoint"));
        }
        let file = std::fs::File::open(path)?;
        let header = &self.shards[shard];
        if file.metadata()?.len() != header.file_bytes {
            return Err(invalid("shard changed after inspection"));
        }
        file.read_exact_at(output, header.data_start + tensor.start + offset)?;
        Ok(())
    }
    fn validate_tensors(&self) -> Result<()> {
        if let Some(plan) = &self.config.mixed {
            let headers = self.tensors().map(|(n, t)| (n.to_owned(), t.clone())).collect();
            plan.clone().with_headers(&headers).map_err(|e| invalid(e.to_string()))?;
            return Ok(());
        }
        let mut checked = 0;
        let mut check = |name: String, dtype, shape: &[u64]| -> Result<()> {
            let t = self.tensor(&name)?;
            if t.dtype != dtype || t.shape != shape {
                return Err(invalid(format!(
                    "unexpected dtype/shape for {name}: {:?} {:?}",
                    t.dtype, t.shape
                )));
            }
            checked += 1;
            Ok(())
        };
        let c = &self.config;
        check("lm_head.weight".into(), Dtype::Bf16, &[c.vocab, c.hidden])?;
        check(
            "model.language_model.embed_tokens.weight".into(),
            Dtype::Bf16,
            &[c.vocab, c.hidden],
        )?;
        for prefix in [
            "model.language_model.hyper_connection_mixer",
            "mtp.hyper_connection_mixer",
        ] {
            check_hc(prefix, false, &mut check)?;
        }
        for (prefix, kinds) in [
            ("model.language_model.layers", c.layers.as_slice()),
            ("mtp.layers", &[LayerKind::SparseAttention][..]),
        ] {
            for (layer, kind) in kinds.iter().enumerate() {
                let p = format!("{prefix}.{layer}");
                for role in ["attn", "mlp"] {
                    check_hc(&format!("{p}.{role}_hyper_connection"), true, &mut check)?;
                }
                let projections = [
                    ("gate_proj", [640, 2560]),
                    ("up_proj", [640, 2560]),
                    ("down_proj", [2560, 640]),
                ];
                for expert in 0..c.experts {
                    for (proj, shape) in projections {
                        let name = format!("{p}.mlp.experts.{expert}.{proj}");
                        if c.format == WeightFormat::Nvfp4 && prefix == "model.language_model.layers" {
                            check(format!("{name}.weight"), Dtype::U8, &[shape[0], shape[1] / 2])?;
                            check(format!("{name}.weight_scale"), Dtype::F8E4M3, &[shape[0], shape[1] / 16])?;
                            check(format!("{name}.weight_scale_2"), Dtype::F32, &[])?;
                            check(format!("{name}.input_scale"), Dtype::F32, &[])?;
                        } else {
                            check(format!("{name}.weight"), Dtype::F8E4M3, &shape)?;
                            check(format!("{name}.weight_scale_inv"), Dtype::Bf16,
                                &[shape[0] / 128, shape[1] / 128])?;
                        }
                    }
                }
                for (proj, shape) in projections {
                    check(
                        format!("{p}.mlp.shared_expert.{proj}.weight"),
                        Dtype::Bf16,
                        &shape,
                    )?;
                }
                check(format!("{p}.mlp.gate.weight"), Dtype::Bf16, &[512, 2560])?;
                check(
                    format!("{p}.mlp.shared_expert_gate.weight"),
                    Dtype::Bf16,
                    &[1, 2560],
                )?;
                let shapes: &[(&str, &[u64])] = match kind {
                    LayerKind::DeltaNet => &[
                        ("A_log", &[48]),
                        ("dt_bias", &[48]),
                        ("conv1d.weight", &[10240, 1, 4]),
                        ("in_proj_a.weight", &[48, 2560]),
                        ("in_proj_b.weight", &[48, 2560]),
                        ("in_proj_qkv.weight", &[10240, 2560]),
                        ("in_proj_z.weight", &[6144, 2560]),
                        ("norm.weight", &[128]),
                        ("out_proj.weight", &[2560, 6144]),
                    ],
                    LayerKind::SparseAttention => &[
                        ("q_proj.weight", &[12288, 2560]),
                        ("k_proj.weight", &[512, 2560]),
                        ("v_proj.weight", &[512, 2560]),
                        ("o_proj.weight", &[2560, 6144]),
                        ("q_norm.weight", &[256]),
                        ("k_norm.weight", &[256]),
                        ("indexer.index_qk_proj.weight", &[640, 2560]),
                        ("indexer.q_layernorm.weight", &[128]),
                        ("indexer.k_layernorm.weight", &[128]),
                    ],
                };
                let block = if *kind == LayerKind::DeltaNet {
                    "linear_attn"
                } else {
                    "self_attn"
                };
                for (name, shape) in shapes {
                    check(format!("{p}.{block}.{name}"), Dtype::Bf16, shape)?;
                }
            }
        }
        for (name, shape) in [
            ("pre_fc_norm_embedding", vec![2560]),
            ("pre_fc_norm_hidden", vec![10240]),
            ("fc_embedding", vec![2560, 2560]),
            ("fc_hidden", vec![2560, 2560]),
        ] {
            check(format!("mtp.{name}.weight"), Dtype::Bf16, &shape)?;
        }
        let p = "model.language_model.layers.1.ple";
        for (name, shape) in [
            ("conv1d", vec![10240, 1, 4]),
            ("key_proj", vec![10240, 2560]),
            ("value_proj", vec![2560, 2560]),
            ("norm_conv", vec![10240]),
            ("norm_key", vec![10240]),
            ("norm_query", vec![10240]),
        ] {
            check(format!("{p}.{name}.weight"), Dtype::Bf16, &shape)?;
        }
        for (name, shape) in [
            ("layer_multipliers", &[3][..]),
            ("ngram_heads_offsets", &[16][..]),
            ("ngram_heads_vocab_sizes", &[16][..]),
        ] {
            check(format!("{PLE_PREFIX}.{name}"), Dtype::I64, shape)?;
        }
        check(
            format!("{PLE_PREFIX}.ngram_embedding.weight_scale"),
            Dtype::Bf16,
            &[1],
        )?;
        for part in 0..128 {
            check(
                format!("{PLE_PREFIX}.ngram_embedding.shard_{part}.weight"),
                Dtype::F8E4M3,
                &[PLE_PART_ROWS, PLE_ROW_DIM],
            )?;
        }
        // Vision is inventoried separately; text-only execution does not allocate it.
        let vision = self
            .tensors()
            .filter(|(n, _)| n.starts_with("model.visual."))
            .count();
        if checked + vision != self.index.weight_map.len() {
            return Err(invalid("unrecognized non-vision Flash tensors"));
        }
        Ok(())
    }
}
fn check_hc(
    prefix: &str,
    inject: bool,
    check: &mut impl FnMut(String, Dtype, &[u64]) -> Result<()>,
) -> Result<()> {
    for (name, shape) in [
        ("hc_norm", &[10240][..]),
        ("input_mix_weight_down", &[320, 10240][..]),
        ("input_mix_weight_up", &[10240, 320][..]),
    ] {
        check(format!("{prefix}.{name}.weight"), Dtype::Bf16, shape)?;
    }
    if inject {
        check(
            format!("{prefix}.block_inject_weight.weight"),
            Dtype::Bf16,
            &[4, 10240],
        )?;
    }
    Ok(())
}

/// Uneven contiguous partition; no fake tokens, heads or experts are introduced.
pub fn partition(total: u64, rank: u64, ranks: u64) -> Result<std::ops::Range<u64>> {
    if rank >= ranks {
        return Err(invalid("invalid rank/count"));
    }
    let base = total / ranks;
    let extra = total % ranks;
    let start = rank * base + rank.min(extra);
    Ok(start..start + base + u64::from(rank < extra))
}
