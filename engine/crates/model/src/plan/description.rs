//! Host descriptions for operations that do not yet have an execution binding.
use super::{ModelPlan, PlanError, Rope};
use crate::{json::{self, Value}, shapes::*};
use super::storage::StorageMap;
use crate::host_ops::router::RouterKind;

/// Schema selection occurs once while reading config data. Kernels must use
/// the operation fields below. A deferred description cannot supply ModelShape.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigPlan { Supported(ModelPlan), Deferred(TextDescription), Loop(LoopTextDescription) }
impl ConfigPlan {
    pub fn from_config(bytes: &[u8]) -> super::Result<Self> {
        let doc = json::parse(bytes, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })
            .map_err(|e| super::read_error("config.json", e))?;
        if let Some(description) = deferred_description(&doc)? { return Ok(Self::Deferred(description)); }
        match loop_description(&doc)? {
            Some(description) => Ok(Self::Loop(description)),
            None => ModelPlan::from_config(bytes).map(Self::Supported),
        }
    }
    pub fn into_model_plan(self) -> super::Result<ModelPlan> {
        match self {
            Self::Supported(plan) => Ok(plan),
            Self::Deferred(description) => description.require_resident_service().map(|never| match never {}),
            Self::Loop(description) => description.require_resident_service().map(|never| match never {}),
        }
    }
}

/// Share metadata schema detection with cartridge admission. No execution kind is added.
pub(crate) fn deferred_description(doc: &Value) -> super::Result<Option<TextDescription>> {
    let root = doc.object().map_err(|_| PlanError::WrongType { field: "config", expected: "object" })?;
    match root.get("model_type").map(Value::string).transpose().map_err(|e| super::read_error("model_type", e))? {
        Some("glm5_next" | "glm5_next_text") => super::glm::parse(doc).map(Some),
        _ => Ok(None),
    }
}
/// The looped schema has its own description type. The text service above never receives it.
pub(crate) fn loop_description(doc: &Value) -> super::Result<Option<LoopTextDescription>> {
    let root = doc.object().map_err(|_| PlanError::WrongType { field: "config", expected: "object" })?;
    match root.get("model_type").map(Value::string).transpose().map_err(|e| super::read_error("model_type", e))? {
        Some("inkling2_loop" | "inkling2_loop_text") => super::inkling::parse(doc).map(Some),
        _ => Ok(None),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextDescription {
    pub hidden: u64,
    pub vocab: u64,
    pub max_positions: u64,
    pub epsilon: f64,
    pub eos_tokens: Vec<u64>,
    pub pad_token: u64,
    pub tied_embeddings: bool,
    pub rope: Option<Rope>,
    pub layers: Vec<DecoderDescription>,
    /// The config declares this count. It does not define the draft forward pass.
    pub declared_draft_layers: u64,
    pub residual: SinkhornDescription,
    pub routing: RoutingDescription,
    pub swiglu_limit: f64,
    pub storage: StorageMap,
}
impl TextDescription {
    /// Complete resident producers must precede a service Factory or executable kind.
    /// Metadata, headers, scalar references, and individual primitives grant no admission.
    /// Success stays uninhabited until the complete owner and its proofs exist.
    pub fn require_resident_service(&self) -> super::Result<std::convert::Infallible> {
        Err(PlanError::Unsupported { field: "execution", value: concat!(
            "complete resident decoder is absent. ",
            "Complete layer, text, causal state, and ordered transport producers are unbound. ",
            "Checkpoint service Factory is absent. ",
            "Source precision, casts, quantizers, and reductions remain unqualified. ",
            "Tokenizer, chat template, stop set, appended prediction semantics, and token-ID proof remain open."
        ).into() })
    }
    /// Activation data for a future execution binding. This does not make
    /// the deferred model executable.
    pub fn swiglu(&self) -> crate::Result<crate::swiglu::SwiGlu> {
        crate::swiglu::SwiGlu::new(Some(self.swiglu_limit))
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecoderDescription { pub mixer: TokenMixer, pub ffn: Ffn }
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenMixer { Kda(KdaDescription), LatentAttention(LatentDescription) }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KdaDescription {
    pub heads: u64, pub head_dim: u64, pub conv: u64, pub gate_lower_bound: f64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentDescription {
    pub heads: u64, pub query_rank: u64, pub kv_rank: u64,
    pub key_dim: u64, pub value_dim: u64, pub rotary_dim: u64,
    pub indexer: PoolIndexerDescription,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolIndexerDescription {
    pub heads: u64, pub dim: u64, pub token_budget: u64, pub pool: u64,
    pub learned_pool: bool, pub select_tail: bool, pub share_for_draft_iteration: bool,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinkhornDescription { pub streams: u64, pub iterations: u64, pub epsilon: f64 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingDescription {
    pub kind: RouterKind, pub normalize_selected: bool,
    pub scale: f64, pub groups: u64, pub selected_groups: u64, pub shared_gate: bool,
}

/// A text decoder that runs one range of layers more than once for each token.
/// It is a sibling of TextDescription and shares no execution with it.
/// Parsed metadata only: no engine runs this description yet.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopTextDescription {
    pub hidden: u64,
    /// Stored rows of the embedding and of the output head.
    pub vocab: u64,
    /// Head rows that the model can write. Rows from here to `vocab` are padding.
    pub head_rows: u64,
    pub max_positions: u64,
    pub epsilon: f64,
    pub eos_tokens: Vec<u64>,
    pub pad_token: Option<u64>,
    /// One entry for each stored layer.
    pub layers: Vec<LoopLayerDescription>,
    /// Run order for one token. A stored layer can occur more than once.
    pub visits: Vec<Visit>,
    pub residual: BlockMixDescription,
    pub core: LoopRange,
    pub routing: RoutingDescription,
    pub caps: SoftCaps,
    /// The convolution after attention and the convolution after the experts.
    pub conv: ShortConvDescription,
    /// The two labels of config.json. The sidecar holds the storage map.
    pub declared_storage: Option<DeclaredStorage>,
    /// The config declares a draft model. This type does not define its forward pass.
    pub declared_draft: Option<DraftDeclaration>,
    /// The config declares an image tower. Image input is not supported.
    pub declares_vision: bool,
}
impl LoopTextDescription {
    /// Any other schema is refused. Use ConfigPlan::from_config to accept all schemas.
    pub fn from_config(bytes: &[u8]) -> super::Result<Self> {
        let doc = json::parse(bytes, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })
            .map_err(|e| super::read_error("config.json", e))?;
        loop_description(&doc)?.ok_or_else(|| PlanError::Unsupported { field: "model_type", value: "not a looped text schema".into() })
    }
    /// A parsed config and a storage map are not support. Success stays uninhabited
    /// until a complete owner binds every part named here and its proofs exist.
    pub fn require_resident_service(&self) -> super::Result<std::convert::Infallible> {
        Err(PlanError::Unsupported { field: "execution", value: concat!(
            "no engine runs this looped decoder. The config and the storage map are metadata only. ",
            "Tensor load, state for each layer visit, window attention with sink and relative bias, ",
            "short convolutions, block mix, loop messages and gates, and soft-capped experts are unbound. ",
            "Tokenizer split rule, chat template, stop set and token-ID proof remain open. ",
            "Image input is not supported."
        ).into() })
    }
    /// Stored layers whose attention sees every earlier token.
    pub fn full_layers(&self) -> Vec<u64> {
        self.layers.iter().enumerate().filter(|(_, l)| l.attention.window.is_none()).map(|(i, _)| i as u64).collect()
    }
    /// Stored layers that run once for each pass.
    pub fn loop_layers(&self) -> std::ops::Range<u64> { self.core.start..self.core.end }
    /// Blocks of the block mix. The last block can be shorter.
    pub fn blocks(&self) -> u64 { (self.layers.len() as u64).div_ceil(self.residual.block_layers) }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopLayerDescription {
    pub attention: SinkAttentionDescription,
    pub ffn: Ffn,
    /// First layer of a block. Its input joins the block list and is not added to the stream.
    pub boundary: bool,
    /// Rows of this layer's block-mix query for attention: 0, 1, or 3 (query, key, value).
    pub attention_queries: u64,
    /// The layer is in the loop range and stores the four message gates.
    pub message_gates: bool,
}
/// Softmax attention with one extra score that has no value (the sink), a learned
/// relative position term, length clips, and a short convolution on keys and values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinkAttentionDescription {
    pub query_heads: u64, pub kv_heads: u64, pub head_dim: u64,
    /// Visible keys: this token and `window - 1` tokens before it. None: every earlier token.
    pub window: Option<u64>,
    pub relative: RelativeBiasDescription,
    /// Largest length of one query head and of one key head.
    pub query_key_clip: f64,
    /// Scale of queries and relative states that grows with the position.
    pub position_scale: Option<PositionScale>,
    pub key_value_conv: ShortConvDescription,
    /// The query, key and value inputs each have their own block mix and norm.
    pub separate_inputs: bool,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeBiasDescription {
    /// Numbers of one head's relative state.
    pub dim: u64,
    /// Distances 0 to `extent - 1` have a term. Other distances have none.
    pub extent: u64,
    /// Largest length of one head's relative state.
    pub clip: f64,
}
/// `1 + alpha * ln(max((position + 1) / floor, 1))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionScale { pub floor: u64, pub alpha: f64 }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortConvDescription { pub kernel: u64, pub taps: ConvTaps }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvTaps {
    /// One stored weight for each channel and tap.
    Fixed { order: TapOrder },
    /// Stored taps plus taps that a projection makes from the current token.
    FixedAndToken { order: TapOrder, index: TapIndex },
}
/// Which stored tap meets the current token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapOrder { OldestFirst, NewestFirst }
/// Layout of the projected taps: index = head * kernel + lag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapIndex { HeadThenLag }
/// A layer reads the past through a softmax mix over finished blocks and the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMixDescription {
    pub block_layers: u64,
    /// The mix blends plain-normed vectors.
    pub normalize: bool,
}
/// Layers `start..end` run `passes` times. Later passes read messages through gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopRange { pub start: u64, pub end: u64, pub passes: u64, pub gate_rank: u64 }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Visit {
    /// Stored layer that supplies the weights.
    pub layer: u64,
    /// 0 for a layer that runs once. Otherwise the pass number, from 1.
    pub pass: u64,
    /// State slot of this visit, in the numbering of the reference cache.
    pub slot: u64,
    /// The visit reads messages from the pass before it.
    pub messages: bool,
}
/// `cap * tanh(x / cap)` limits for the expert gate, the expert up branch and the logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftCaps { pub gate: f64, pub up: f64, pub logits: f64 }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredStorage { pub method: String, pub algo: String }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftDeclaration {
    pub layers: u64,
    /// States the draft model reads. A number under the visit count is the stream at the
    /// input of that visit. The visit count itself is the state after the last norm.
    pub taps: Vec<u64>,
    pub hidden: u64, pub query_heads: u64, pub kv_heads: u64, pub head_dim: u64,
    pub width: u64, pub markov_rank: u64,
}
