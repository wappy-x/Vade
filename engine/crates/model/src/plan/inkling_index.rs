//! Give each stored tensor name of the looped block-attention text family a role, a type and a shape.
//! Header metadata only: this module reads no payload and admits no execution.
//!
//! Sources. Names, types and shapes: the headers of the main weight file and of the draft file.
//! Stored name to module: `transformers` v5.24.0, `conversion_mapping.py` lines 353-416 and 2684-2708.
//! Head sizes and relative extent of a window layer: `modeling_inkling2_loop.py` lines 552-557 and 645-650.
//! Three block-mix rows of a full layer: the same file, lines 1037-1039.
//! Packed expert tensors: `core_model_loading.py` lines 772-816. Loop message queries: lines 823-849.
//! The reference has no draft model. The draft roles follow the draft file header and `mtp_config`.
//! This file holds none of the reference code.
use super::{PlanError, Result, description::{ConvTaps, LoopTextDescription, loop_description}, read_error,
    storage_groups::{GroupKind, SIDECAR_FILE, StorageGroup, StorageGroups}};
use crate::{json, safetensors::{self, Dtype, Header, Index, Limits, Tensor},
    shapes::{CONFIG_BYTES, ExpertStorage, Ffn, HEADER_BYTES, MAX_SHARDS, MAX_TENSORS, MAX_TENSOR_RANK, Place, Proj}};
use std::{collections::{BTreeMap, BTreeSet}, path::Path};

/// Units of the two expert storage forms. They belong to the forms, not to a checkpoint.
/// 4-bit: two codes in a byte and one scale for 16 values of a row. 8-bit: one scale for 32 values of a row.
const FP4_CODES_PER_BYTE: u64 = 2;
const FP4_BLOCK: u64 = 16;
const MX8_BLOCK: u64 = 32;
/// Stored name of the draft file. The config names no file. The storage sidecar calls this group `mtp`.
pub const DRAFT_FILE: &str = "mtp.safetensors";
const TEXT: &str = "model.llm.";
const VISION: &str = "model.visual.";

/// Block-mix query `attn_queries.i` belongs to the attention input of layer `i + 1`. Layer 0 has none.
fn mix_attention_layer(index: u64) -> u64 { index + 1 }

/// The stored form that the storage sidecar declares for one listed name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stored { Bf16, F32, Nvfp4, Mxfp8 }

/// Sizes of one stored layer. Every value comes from the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSizes {
    pub query_heads: u64, pub kv_heads: u64, pub head_dim: u64,
    /// The attention sees a window of tokens. This names the operation only. It changes no stored name.
    pub windowed: bool,
    /// The query, key and value inputs each have a norm. Otherwise the layer has one attention norm.
    pub separate_inputs: bool,
    /// Rows of the block-mix query for attention. Zero: the layer stores none.
    pub mix_rows: u64,
    /// The key and value convolutions store fixed taps and a projection that makes taps for each token.
    pub token_taps: bool,
    pub relative_dim: u64, pub relative_extent: u64,
    /// Taps of the key and value convolutions.
    pub kv_taps: u64,
    pub experts: u64, pub expert_width: u64,
    /// The layer is inside the loop range and has the four message gates.
    pub looped: bool,
}
/// Sizes of the draft model that the config declares. The draft is bound and never run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftSizes {
    pub hidden: u64, pub layers: u64, pub query_heads: u64, pub kv_heads: u64, pub head_dim: u64,
    pub width: u64, pub markov_rank: u64,
    /// Count of main-model states that the draft reads (`target_layer_ids`).
    pub taps: u64,
    pub stored: BTreeMap<String, Stored>,
}
/// All that the index needs from the config and the storage sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexInput {
    pub hidden: u64,
    /// Stored rows of the embedding and of the head. This is the padded count.
    pub stored_vocab: u64,
    /// Taps of the convolutions after attention and after the experts.
    pub conv_taps: u64,
    pub gate_rank: u64,
    pub layers: Vec<LayerSizes>,
    pub draft: Option<DraftSizes>,
    /// Each name that the sidecar lists for the main file. Scale companions are not listed.
    pub stored: BTreeMap<String, Stored>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream { Key, Value }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageGate { Query, Key, Value, Expert }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatePart { Down, Forget, Gain }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPart { Codes, BlockScale, ExpertScale, ScaleDivisor }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftRole {
    Query, Key, Value, ContextKey, ContextValue, Output, QueryNorm, KeyNorm, ContextKeyNorm,
    Gate, Up, Down, InputNorm, PostAttentionNorm, ContextNorm,
    TokenTable, StateProjection, StateBias, LatentOutput, LatentBias, HeadContextNorm,
    Fuse, MaskEmbedding, FinalNorm,
}
/// What one stored tensor is. Later plans read roles, never names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredRole {
    Embedding, EmbeddingNorm, Head, FinalNorm,
    /// Block-mix query of an attention input. One row, or three rows (query, key, value) for separate inputs.
    MixAttention,
    MixExperts, MixOutput,
    /// Rows `2j` and `2j + 1`: the query and the expert message query of loop layer `j`.
    MessageQueries,
    /// Rows `2j` and `2j + 1`: the key and the value message query of loop layer `j`.
    MessageKvQueries,
    Query, Key, Value, Output, OutputGate, Sink, QueryClip,
    RelativeProjection, RelativeTable, RelativeClip,
    /// Fixed taps of a layer that has only fixed taps, stored `[channels, 1, taps]`.
    ConvFixed(Stream),
    /// Fixed taps of a layer that also makes taps for each token, stored `[taps, channels]`.
    ConvStatic(Stream),
    /// Projection that makes taps for each token: one row for each key and value head and tap.
    ConvProjection(Stream),
    ConvAfterAttention, ConvAfterExperts,
    AttentionNorm, QueryInputNorm, KeyInputNorm, ValueInputNorm, ExpertNorm,
    Router, RouterBias,
    /// `Proj::GateUp` holds gate and up rows in one matrix. `Proj::Down` is the down matrix.
    Expert { proj: Proj, part: ExpertPart },
    Gate { gate: MessageGate, part: GatePart },
    /// A tensor of the draft file. It is bound and never run.
    Draft(DraftRole),
    /// A `model.visual.*` tensor. Text serving does not use it and this index does not check its shape.
    Vision,
}
/// An operation that stored tensors need. The order is the order of the refusal text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operation {
    EmbeddingNorm, WindowAttention, FullAttention, RelativeBias, Sink, OutputGate, Convolution,
    BlockMix, LoopMessages, Router, Experts4, Experts8, Head, Draft,
}
/// Parts of a service that no stored tensor shows. Remove an entry when its part lands.
const SERVICE_PARTS: &[&str] = &["weight load", "state for each layer visit", "tokenizer split rule", "chat template", "stop set", "token-ID proof"];
impl Operation {
    pub fn label(self) -> &'static str {
        match self {
            Self::EmbeddingNorm => "embedding norm", Self::WindowAttention => "window attention", Self::FullAttention => "full attention",
            Self::RelativeBias => "relative position bias", Self::Sink => "attention sink", Self::OutputGate => "attention output gate",
            Self::Convolution => "short convolutions", Self::BlockMix => "block mix", Self::LoopMessages => "loop messages and gates",
            Self::Router => "router", Self::Experts4 => "4-bit expert stacks", Self::Experts8 => "8-bit expert stacks",
            Self::Head => "final norm and padded head", Self::Draft => "draft model",
        }
    }
}
impl StoredRole {
    /// The operation that reads this tensor. `dtype` tells the two expert forms apart.
    /// `None` for the image tower: text serving has no operation for it.
    fn operation(self, windowed: bool, dtype: Dtype) -> Option<Operation> {
        use StoredRole::*;
        Some(match self {
            Embedding | EmbeddingNorm => Operation::EmbeddingNorm,
            Head | FinalNorm | MixOutput => Operation::Head,
            MixAttention | MixExperts => Operation::BlockMix,
            MessageQueries | MessageKvQueries | Gate { .. } => Operation::LoopMessages,
            Query | Key | Value | Output | QueryClip | AttentionNorm | QueryInputNorm | KeyInputNorm | ValueInputNorm =>
                if windowed { Operation::WindowAttention } else { Operation::FullAttention },
            OutputGate => Operation::OutputGate, Sink => Operation::Sink,
            RelativeProjection | RelativeTable | RelativeClip => Operation::RelativeBias,
            ConvFixed(_) | ConvStatic(_) | ConvProjection(_) | ConvAfterAttention | ConvAfterExperts => Operation::Convolution,
            Router | RouterBias | ExpertNorm => Operation::Router,
            Expert { part: ExpertPart::Codes, .. } => if dtype == Dtype::U8 { Operation::Experts4 } else { Operation::Experts8 },
            Expert { part: ExpertPart::BlockScale, .. } => if dtype == Dtype::U8 { Operation::Experts8 } else { Operation::Experts4 },
            Expert { .. } => Operation::Experts4,
            Draft(_) => Operation::Draft, Vision => return None,
        })
    }
}

/// What the engine does with a bound tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorState {
    /// Part of the text model.
    Text,
    /// Part of the draft model. It has a role and a checked shape. Nothing runs from it.
    NotRun,
    /// Part of the image tower. Text serving never loads it.
    Unused,
}
/// The role, type and shape that the config and the sidecar give one name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expected { pub role: StoredRole, pub place: Place, pub dtype: Dtype, pub shape: Vec<u64> }
/// One stored name after its header passed the check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTensor {
    pub role: StoredRole, pub place: Place, pub state: TensorState,
    pub dtype: Dtype, pub shape: Vec<u64>, pub bytes: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    /// Names in the main file, and how many of them have a role. A bound index has no name without a role.
    pub stored: usize, pub bound: usize,
    pub text: usize, pub unused: usize,
    /// Names in the draft file, and how many of them have a role.
    pub draft_stored: usize, pub draft_bound: usize,
    pub stored_bytes: u64, pub draft_bytes: u64,
}
/// Every stored name of a checkpoint with its role. This is metadata. It admits no execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundIndex {
    pub tensors: BTreeMap<String, BoundTensor>,
    pub draft: BTreeMap<String, BoundTensor>,
    /// The config declares a draft model. The draft file can still be absent.
    pub draft_declared: bool,
    pub operations: BTreeSet<Operation>,
}

fn unsupported(field: &'static str, value: String) -> PlanError { PlanError::Unsupported { field, value } }
fn mul(a: u64, b: u64) -> Result<u64> { super::mul(a, b) }
fn layer_id(layer: usize) -> Result<u32> { u32::try_from(layer).map_err(|_| PlanError::Overflow { what: "layer id" }) }

struct Names<'a> { stored: &'a BTreeMap<String, Stored>, out: BTreeMap<String, Expected> }
impl Names<'_> {
    fn insert(&mut self, name: String, role: StoredRole, place: Place, dtype: Dtype, shape: Vec<u64>) -> Result<()> {
        if shape.contains(&0) { return Err(PlanError::Invalid { field: "tensor shape", reason: "a size in the config gives an empty tensor" }); }
        shape.iter().try_fold(dtype.bytes(), |n, &d| mul(n, d))?;
        if self.out.len() >= MAX_TENSORS { return Err(PlanError::Invalid { field: "tensors", reason: "expected tensor count exceeds limit" }); }
        if self.out.insert(name, Expected { role, place, dtype, shape }).is_some() {
            return Err(PlanError::Invalid { field: "tensors", reason: "two roles give the same stored name" });
        }
        Ok(())
    }
    /// A tensor stored as it is. Its type is the type of its storage group.
    fn plain(&mut self, name: String, role: StoredRole, place: Place, shape: Vec<u64>) -> Result<()> {
        let dtype = match self.stored.get(&name) {
            Some(Stored::Bf16) => Dtype::Bf16, Some(Stored::F32) => Dtype::F32,
            Some(other) => return Err(unsupported("tensor storage", format!("{name}: {other:?} storage for a tensor that is not an expert stack"))),
            None => return Err(unsupported("tensor storage", format!("{name}: the storage sidecar does not list this tensor"))),
        };
        self.insert(name, role, place, dtype, shape)
    }
    /// One expert stack: `count` matrices of `rows` by `columns`, with the scale tensors of its storage form.
    fn experts(&mut self, name: String, place: Place, proj: Proj, count: u64, rows: u64, columns: u64) -> Result<()> {
        let role = |part| StoredRole::Expert { proj, part };
        let whole = |ok: bool| if ok { Ok(()) } else {
            Err(PlanError::Invalid { field: "expert columns", reason: "columns do not fill complete scale blocks" }) };
        match self.stored.get(&name) {
            Some(Stored::Nvfp4) => {
                whole(columns.is_multiple_of(FP4_BLOCK))?;
                self.insert(format!("{name}_scale"), role(ExpertPart::BlockScale), place, Dtype::F8E4M3, vec![count, rows, columns / FP4_BLOCK])?;
                self.insert(format!("{name}_amax"), role(ExpertPart::ExpertScale), place, Dtype::F32, vec![count])?;
                self.insert(format!("{name}_outer_scale_denominator"), role(ExpertPart::ScaleDivisor), place, Dtype::F32, vec![1])?;
                self.insert(name, role(ExpertPart::Codes), place, Dtype::U8, vec![count, rows, columns / FP4_CODES_PER_BYTE])
            }
            Some(Stored::Mxfp8) => {
                whole(columns.is_multiple_of(MX8_BLOCK))?;
                self.insert(format!("{name}_scale"), role(ExpertPart::BlockScale), place, Dtype::U8, vec![count, rows, columns / MX8_BLOCK])?;
                self.insert(name, role(ExpertPart::Codes), place, Dtype::F8E4M3, vec![count, rows, columns])
            }
            Some(other) => Err(unsupported("tensor storage", format!("{name}: {other:?} storage for an expert stack"))),
            None => Err(unsupported("tensor storage", format!("{name}: the storage sidecar does not list this tensor"))),
        }
    }
    /// Every listed name must be an expected name. `skip` names the tensors that have no role by design.
    fn finish(self, skip: &str) -> Result<BTreeMap<String, Expected>> {
        match self.stored.keys().find(|name| !(self.out.contains_key(*name) || !skip.is_empty() && name.starts_with(skip))) {
            Some(name) => Err(unsupported("storage sidecar", format!("{name}: the config gives this listed tensor no role"))),
            None => Ok(self.out),
        }
    }
}

impl IndexInput {
    /// The sizes and the storage groups that the index needs, from the parsed config and the parsed sidecar.
    pub fn from_description(d: &LoopTextDescription, groups: &StorageGroups) -> Result<Self> {
        let stored = |groups: &[StorageGroup]| -> Result<BTreeMap<String, Stored>> {
            let mut out = BTreeMap::new();
            for g in groups {
                let stored = match g.kind {
                    GroupKind::Plain { dtype: Dtype::Bf16, .. } => Stored::Bf16, GroupKind::Plain { dtype: Dtype::F32, .. } => Stored::F32,
                    GroupKind::Nvfp4 => Stored::Nvfp4, GroupKind::Mxfp8 => Stored::Mxfp8,
                    GroupKind::Plain { dtype, .. } => return Err(unsupported("tensor storage", format!("a storage group stored as {dtype:?}"))),
                };
                for name in &g.tensors {
                    if out.insert(name.clone(), stored).is_some() { return Err(PlanError::Invalid { field: "storage sidecar", reason: "a tensor is in two storage groups" }); }
                }
            }
            Ok(out)
        };
        let layers = d.layers.iter().map(|l| {
            // The stored names below hold all experts of a layer in one stack and no shared expert.
            let Ffn::Experts(e) = l.ffn else { return Err(unsupported("feed-forward", "a dense layer has no stored names in this schema".into())); };
            if e.storage != ExpertStorage::Stacked || e.shared_width != 0 {
                return Err(unsupported("feed-forward", "experts that are not one stack, or a shared expert, have no stored names in this schema".into()));
            }
            let a = l.attention;
            Ok(LayerSizes { query_heads: a.query_heads, kv_heads: a.kv_heads, head_dim: a.head_dim, windowed: a.window.is_some(),
                separate_inputs: a.separate_inputs, mix_rows: l.attention_queries, token_taps: matches!(a.key_value_conv.taps, ConvTaps::FixedAndToken { .. }),
                relative_dim: a.relative.dim, relative_extent: a.relative.extent, kv_taps: a.key_value_conv.kernel,
                experts: e.count, expert_width: e.width, looped: l.message_gates })
        }).collect::<Result<Vec<_>>>()?;
        let draft = d.declared_draft.as_ref().map(|m| Ok(DraftSizes { hidden: m.hidden, layers: m.layers, query_heads: m.query_heads, kv_heads: m.kv_heads,
            head_dim: m.head_dim, width: m.width, markov_rank: m.markov_rank, taps: m.taps.len() as u64, stored: stored(&groups.draft)? })).transpose()?;
        Ok(Self { hidden: d.hidden, stored_vocab: d.vocab, conv_taps: d.conv.kernel, gate_rank: d.core.gate_rank, layers, draft, stored: stored(&groups.model)? })
    }
    /// Every name that the text model needs in the main file. `model.visual.*` names are not in this map.
    pub fn expected_main(&self) -> Result<BTreeMap<String, Expected>> {
        use StoredRole::*;
        let h = self.hidden;
        let mut n = Names { stored: &self.stored, out: BTreeMap::new() };
        let model = Place::Model;
        n.plain(format!("{TEXT}embed.weight"), Embedding, model, vec![self.stored_vocab, h])?;
        n.plain(format!("{TEXT}embed_norm.weight"), EmbeddingNorm, model, vec![h])?;
        n.plain(format!("{TEXT}unembed.weight"), Head, model, vec![self.stored_vocab, h])?;
        n.plain(format!("{TEXT}norm.weight"), FinalNorm, model, vec![h])?;
        n.plain(format!("{TEXT}attn_res.output_query"), MixOutput, model, vec![h])?;
        let looped = self.layers.iter().filter(|l| l.looped).count() as u64;
        if looped > 0 {
            n.plain(format!("{TEXT}middle_cycle_cross_pass_queries"), MessageQueries, model, vec![mul(2, looped)?, h])?;
            n.plain(format!("{TEXT}middle_cycle_cross_pass_kv_queries"), MessageKvQueries, model, vec![mul(2, looped)?, h])?;
        }
        for (i, l) in self.layers.iter().enumerate() {
            let place = Place::Layer(layer_id(i)?);
            let (q, kv) = (mul(l.query_heads, l.head_dim)?, mul(l.kv_heads, l.head_dim)?);
            if l.mix_rows > 0 {
                let index = (0..i as u64).find(|&index| mix_attention_layer(index) == i as u64)
                    .ok_or(PlanError::Invalid { field: "attention_queries", reason: "the first layer cannot store a block-mix query for attention" })?;
                n.plain(format!("{TEXT}attn_res.attn_queries.{index}"), MixAttention, place, if l.mix_rows == 1 { vec![h] } else { vec![l.mix_rows, h] })?;
            }
            n.plain(format!("{TEXT}attn_res.mlp_queries.{i}"), MixExperts, place, vec![h])?;
            let p = format!("{TEXT}layers.{i}.");
            for (suffix, role, shape) in [
                ("attn.wq_du.weight", Query, vec![q, h]), ("attn.wk_dv.weight", Key, vec![kv, h]), ("attn.wv_dv.weight", Value, vec![kv, h]),
                ("attn.wo_ud.weight", Output, vec![h, q]), ("attn.gate_attn.weight", OutputGate, vec![q, h]),
                ("attn.wr_du.weight", RelativeProjection, vec![mul(l.query_heads, l.relative_dim)?, h]),
                ("attn.rel_logits_proj.proj", RelativeTable, vec![l.relative_dim, l.relative_extent]),
                ("attn.rel_l2_clip.scale", RelativeClip, vec![l.query_heads, l.relative_dim]),
                ("attn.attn_sink_proj.weight", Sink, vec![l.query_heads, h]), ("attn.q_l2_clip.scale", QueryClip, vec![l.query_heads, l.head_dim]),
                ("attn_sconv.weight", ConvAfterAttention, vec![h, 1, self.conv_taps]), ("mlp_sconv.weight", ConvAfterExperts, vec![h, 1, self.conv_taps]),
                ("mlp_norm.weight", ExpertNorm, vec![h]),
                ("mlp.gate.weight", Router, vec![l.experts, h]), ("mlp.gate.bias", RouterBias, vec![l.experts]),
            ] { n.plain(format!("{p}{suffix}"), role, place, shape)?; }
            for (stream, s) in [(Stream::Key, "k"), (Stream::Value, "v")] {
                if l.token_taps {
                    n.plain(format!("{p}attn.{s}_sconv.static_w"), ConvStatic(stream), place, vec![l.kv_taps, kv])?;
                    n.plain(format!("{p}attn.{s}_sconv.weight_proj"), ConvProjection(stream), place, vec![mul(l.kv_heads, l.kv_taps)?, h])?;
                } else {
                    n.plain(format!("{p}attn.{s}_sconv.weight"), ConvFixed(stream), place, vec![kv, 1, l.kv_taps])?;
                }
            }
            if l.separate_inputs {
                for (s, role) in [("q", QueryInputNorm), ("k", KeyInputNorm), ("v", ValueInputNorm)] {
                    n.plain(format!("{p}{s}_input_norm.weight"), role, place, vec![h])?;
                }
            } else { n.plain(format!("{p}attn_norm.weight"), AttentionNorm, place, vec![h])?; }
            // Gate and up rows share one stored matrix: twice the expert width in rows, hidden in columns.
            n.experts(format!("{p}mlp.experts.w13_weight"), place, Proj::GateUp, l.experts, mul(2, l.expert_width)?, h)?;
            n.experts(format!("{p}mlp.experts.w2_weight"), place, Proj::Down, l.experts, h, l.expert_width)?;
            if l.looped {
                for (gate, g) in [(MessageGate::Query, "attn"), (MessageGate::Key, "k"), (MessageGate::Value, "v"), (MessageGate::Expert, "mlp")] {
                    for (part, s, shape) in [(GatePart::Down, "down_weight", vec![self.gate_rank, mul(2, h)?]),
                        (GatePart::Forget, "forget_weight", vec![h, self.gate_rank]), (GatePart::Gain, "gain_weight", vec![h, self.gate_rank])] {
                        n.plain(format!("{p}controllers.{g}.{s}"), Gate { gate, part }, place, shape)?;
                    }
                }
            }
        }
        n.finish(VISION)
    }
    /// Every name of the draft file. Empty when the config declares no draft model.
    pub fn expected_draft(&self) -> Result<BTreeMap<String, Expected>> {
        use DraftRole::*;
        let Some(d) = &self.draft else { return Ok(BTreeMap::new()); };
        let h = d.hidden;
        let mut n = Names { stored: &d.stored, out: BTreeMap::new() };
        let (q, kv) = (mul(d.query_heads, d.head_dim)?, mul(d.kv_heads, d.head_dim)?);
        for i in 0..usize::try_from(d.layers).map_err(|_| PlanError::Overflow { what: "draft layer count" })? {
            let place = Place::Draft { layer: Some(layer_id(i)?) };
            for (suffix, role, shape) in [
                ("self_attn.q_proj.weight", Query, vec![q, h]), ("self_attn.k_proj.weight", Key, vec![kv, h]), ("self_attn.v_proj.weight", Value, vec![kv, h]),
                ("self_attn.k_proj_ctx.weight", ContextKey, vec![kv, h]), ("self_attn.v_proj_ctx.weight", ContextValue, vec![kv, h]),
                ("self_attn.o_proj.weight", Output, vec![h, q]), ("self_attn.q_norm.weight", QueryNorm, vec![d.head_dim]),
                ("self_attn.k_norm.weight", KeyNorm, vec![d.head_dim]), ("self_attn.k_norm_ctx.weight", ContextKeyNorm, vec![d.head_dim]),
                ("mlp.gate_proj.weight", Gate, vec![d.width, h]), ("mlp.up_proj.weight", Up, vec![d.width, h]), ("mlp.down_proj.weight", Down, vec![h, d.width]),
                ("input_layernorm.weight", InputNorm, vec![h]), ("post_attention_layernorm.weight", PostAttentionNorm, vec![h]),
                ("dflare_norm.weight", ContextNorm, vec![h]),
            ] { n.plain(format!("layers.{i}.{suffix}"), StoredRole::Draft(role), place, shape)?; }
        }
        let head = Place::Draft { layer: None };
        let (gates, input) = draft_state_shape(d)?;
        for (name, role, shape) in [
            ("markov_head.markov_w1.weight", TokenTable, vec![self.stored_vocab, d.markov_rank]),
            ("markov_head.rnn_proj.weight", StateProjection, vec![gates, input]), ("markov_head.rnn_proj.bias", StateBias, vec![gates]),
            ("markov_head.hidden_out.weight", LatentOutput, vec![h, d.markov_rank]), ("markov_head.hidden_out.bias", LatentBias, vec![h]),
            ("markov_head.dflare_norm.weight", HeadContextNorm, vec![h]),
            // One row of mixing weights for each draft layer and one for the head.
            ("fuse_weight", Fuse, vec![d.layers.checked_add(1).ok_or(PlanError::Overflow { what: "draft layer count" })?, d.taps]),
            ("mask_embed", MaskEmbedding, vec![h]), ("norm.weight", FinalNorm, vec![h]),
        ] { n.plain(name.into(), StoredRole::Draft(role), head, shape)?; }
        n.finish("")
    }
    /// Check every stored name against its expected role, type and shape. `draft` holds the header of the
    /// draft file when that file is present. A name without a role is a refusal, so a bound index has none.
    pub fn bind(&self, main: &BTreeMap<String, Tensor>, draft: Option<&BTreeMap<String, Tensor>>) -> Result<BoundIndex> {
        let expected = self.expected_main()?;
        let mut operations = BTreeSet::new();
        let mut tensors = BTreeMap::new();
        for (name, t) in main {
            let bound = if name.starts_with(VISION) {
                BoundTensor { role: StoredRole::Vision, place: Place::Unused, state: TensorState::Unused, dtype: t.dtype, shape: t.shape.clone(), bytes: extent(name, t)? }
            } else {
                let e = expected.get(name).ok_or_else(|| PlanError::UnexpectedTensor { name: name.clone() })?;
                let windowed = matches!(e.place, Place::Layer(i) if self.layers.get(i as usize).is_some_and(|l| l.windowed));
                operations.extend(e.role.operation(windowed, e.dtype));
                check(name, e, t, TensorState::Text)?
            };
            tensors.insert(name.clone(), bound);
        }
        if let Some(name) = expected.keys().find(|name| !main.contains_key(*name)) { return Err(PlanError::MissingTensor { name: name.clone() }); }
        let mut bound_draft = BTreeMap::new();
        if let Some(headers) = draft {
            if self.draft.is_none() { return Err(PlanError::Invalid { field: "mtp_config", reason: "a draft file is stored but the config declares no draft model" }); }
            let expected = self.expected_draft()?;
            for (name, t) in headers {
                let e = expected.get(name).ok_or_else(|| PlanError::UnexpectedTensor { name: format!("{DRAFT_FILE}: {name}") })?;
                bound_draft.insert(name.clone(), check(name, e, t, TensorState::NotRun)?);
            }
            if let Some(name) = expected.keys().find(|name| !headers.contains_key(*name)) {
                return Err(PlanError::MissingTensor { name: format!("{DRAFT_FILE}: {name}") });
            }
            operations.insert(Operation::Draft);
        }
        Ok(BoundIndex { tensors, draft: bound_draft, draft_declared: self.draft.is_some(), operations })
    }
}
/// Rows and columns of the draft head's state projection: three groups of `markov_rank` rows (gate, candidate,
/// output); the input is the state, the row of the token before, the draft vector and the main-model vector.
/// Read from the draft file header. The reference has no draft code, so no run has confirmed this reading.
fn draft_state_shape(d: &DraftSizes) -> Result<(u64, u64)> {
    let input = mul(2, d.markov_rank)?.checked_add(mul(2, d.hidden)?).ok_or(PlanError::Overflow { what: "draft state input" })?;
    Ok((mul(3, d.markov_rank)?, input))
}
fn extent(name: &str, t: &Tensor) -> Result<u64> {
    let bytes = t.shape.iter().try_fold(t.dtype.bytes(), |n, &d| mul(n, d))?;
    if t.shape.len() > MAX_TENSOR_RANK || t.end.checked_sub(t.start) != Some(bytes) {
        return Err(unsupported("tensor byte extent", format!("{name}: offsets differ from shape and type")));
    }
    Ok(bytes)
}
fn check(name: &str, e: &Expected, t: &Tensor, state: TensorState) -> Result<BoundTensor> {
    let bytes = extent(name, t)?;
    if e.dtype != t.dtype { return Err(unsupported("tensor dtype", format!("{name}: expected {:?}, actual {:?}", e.dtype, t.dtype))); }
    if e.shape != t.shape { return Err(PlanError::TensorMismatch { name: name.into(), expected: e.shape.clone(), actual: t.shape.clone() }); }
    Ok(BoundTensor { role: e.role, place: e.place, state, dtype: t.dtype, shape: t.shape.clone(), bytes })
}

/// The start option of the preview text service of this family. Without it a start is refused with `BoundIndex::reason`.
pub const PREVIEW_OPTION: &str = "--diagnostic-unqualified-text";

impl BoundIndex {
    pub fn counts(&self) -> Counts {
        let unused = self.tensors.values().filter(|t| t.state == TensorState::Unused).count();
        Counts { stored: self.tensors.len(), bound: self.tensors.len(), text: self.tensors.len() - unused, unused,
            draft_stored: self.draft.len(), draft_bound: self.draft.len(),
            stored_bytes: self.tensors.values().map(|t| t.bytes).sum(), draft_bytes: self.draft.values().map(|t| t.bytes).sum() }
    }
    /// What this engine does not have yet: the operations that the stored tensors need, then the parts of a
    /// service that no tensor shows. This engine has an execution binding for none of the operations.
    pub fn missing_operations(&self) -> Vec<String> {
        self.operations.iter().map(|o| o.label()).chain(SERVICE_PARTS.iter().copied()).map(str::to_owned).collect()
    }
    /// A bound index is metadata. Success stays uninhabited until every operation has an execution binding.
    pub fn require_execution(&self) -> Result<std::convert::Infallible> { Err(PlanError::MissingOps { ops: self.missing_operations() }) }
    /// This engine runs the text part only. Success stays uninhabited: image input has no path.
    pub fn require_image_input(&self) -> Result<std::convert::Infallible> {
        let unused = self.counts().unused;
        Err(unsupported("image input", if unused == 0 { "this engine runs text only and this model stores no image tower".into() }
            else { format!("this engine runs the text part only; the {unused} image tower tensors are bound as not used") }))
    }
    /// The refusal that `vade models`, the launcher and a start without `PREVIEW_OPTION` print for this family: what
    /// runs (text chat, only with that option), what is not built (image input, the draft model, when the checkpoint
    /// stores them), that it has not been through qualification, then the counts of the binding.
    pub fn reason(&self, model_type: &str) -> String {
        let c = self.counts();
        let mut missing = Vec::new();
        if c.unused > 0 { missing.push("image input"); }
        if !self.draft.is_empty() || self.draft_declared { missing.push("the draft model"); }
        let mut text = format!("This engine runs this model only as a preview, and only with the start option {PREVIEW_OPTION}: text chat runs");
        if !missing.is_empty() { text += &format!("; not built yet: {}", missing.join(" and ")); }
        text += &format!(". It has not been through qualification. Model type {model_type}: {} of {} stored tensors have a role", c.bound, c.stored);
        if c.unused > 0 { text += &format!(" ({} text, {} image tower and not used)", c.text, c.unused); }
        if !self.draft.is_empty() { text += &format!("; {} of {} draft tensors are bound and not run", c.draft_bound, c.draft_stored); }
        else if self.draft_declared { text += "; the draft file is absent"; }
        text += ".";
        text
    }
}

fn read(root: &Path, file: &'static str, limit: usize) -> Result<Vec<u8>> {
    safetensors::read_bounded(&root.join(file), limit).map_err(|e| read_error(file, e))
}
/// Read the sizes and the storage groups of the checkpoint in `root` from its config and its storage sidecar.
pub fn read_input(root: &Path) -> Result<IndexInput> {
    let description = LoopTextDescription::from_config(&read(root, "config.json", CONFIG_BYTES)?)?;
    let groups = StorageGroups::from_json(&read(root, SIDECAR_FILE, CONFIG_BYTES)?).map_err(|e| read_error(SIDECAR_FILE, e))?;
    // The sidecar must agree with the storage labels of the config.
    groups.storage_map(&description).map_err(|e| read_error("config.json and hf_quant_config.json", e))?;
    IndexInput::from_description(&description, &groups)
}
/// Read the config, the storage sidecar, the index and the headers of the checkpoint in `root`, then bind
/// every stored name. This reads no tensor payload.
pub fn inspect(root: &Path) -> Result<BoundIndex> {
    let shape = read_input(root)?;
    let read = |file: &'static str, limit: usize| read(root, file, limit);
    let limits = Limits { header_bytes: HEADER_BYTES, tensors: MAX_TENSORS, rank: MAX_TENSOR_RANK, shards: MAX_SHARDS };
    const INDEX: &str = "model.safetensors.index.json";
    let index = Index::parse(&read(INDEX, HEADER_BYTES)?, limits).map_err(|e| read_error(INDEX, e))?;
    let shards = index.inspect_shards(root, limits).map_err(|e| read_error("safetensors headers", e))?;
    let main = shards.values().flat_map(|header| header.tensors.iter().map(|(name, t)| (name.clone(), t.clone()))).collect();
    let path = root.join(DRAFT_FILE);
    let draft = if path.try_exists().map_err(|e| read_error(DRAFT_FILE, e))? {
        let (root, path) = (root.canonicalize().map_err(|e| read_error(DRAFT_FILE, e))?, path.canonicalize().map_err(|e| read_error(DRAFT_FILE, e))?);
        if !path.starts_with(&root) { return Err(read_error(DRAFT_FILE, "symlink escapes checkpoint directory")); }
        Some(Header::open(&path, limits).map_err(|e| read_error(DRAFT_FILE, e))?.tensors)
    } else { None };
    shape.bind(&main, draft.as_ref())
}

/// Why this engine cannot serve the checkpoint in `root`, when its config belongs to this family.
/// `None` for every other config, so a caller keeps its own text for other models.
pub fn unserved_reason(root: &Path, config: &[u8]) -> Option<String> {
    let doc = json::parse(config, json::Limits { bytes: CONFIG_BYTES, ..Default::default() }).ok()?;
    let model_type = doc.object().ok()?.get("model_type")?.string().ok()?.to_owned();
    // `Ok(None)` is another schema. An error is a config of this schema with a refused field.
    if matches!(loop_description(&doc), Ok(None)) { return None; }
    Some(match inspect(root) {
        Ok(bound) => bound.reason(&model_type),
        Err(error) => format!("This engine cannot load this model. Model type {model_type}: {error}."),
    })
}
