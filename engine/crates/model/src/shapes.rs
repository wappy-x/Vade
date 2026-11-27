//! Shapes and sizes shared by the model plan (piece A) and the splitter
//! (piece C): the one constants file for both. No model dimension is written
//! here. Piece A fills these records from config.json and safetensors headers;
//! the constants are limits and layout units that no checkpoint states.
//! Everything is logical (layers, heads, experts, rows, bytes), never an address.
use crate::safetensors::Dtype;

/// Spark counts a split may use.
pub const MIN_SPARKS: u32 = 1;
pub const MAX_SPARKS: u32 = 8;

/// FP8 block-scale edge the kernels accept. config.json's weight_block_size
/// is still read and must equal [128, 128] (flash/mod.rs:159, qwen3_5.rs:59).
pub const FP8_BLOCK: u64 = 128;
/// Split BF16 channel widths move in groups of eight (flash/plan.rs:164-166).
pub const BF16_GROUP: u64 = 8;
/// Row-owned expert down projections split output rows in 128-row blocks
/// (engine flash/session.rs:594-595).
pub const ROW_BLOCK: u64 = 128;

/// Bounded metadata reads, copied from today's readers (flash/mod.rs:84, 323-329).
pub const CONFIG_BYTES: usize = 1 << 20;
pub const HEADER_BYTES: usize = 64 << 20;
pub const MAX_TENSORS: usize = 320_000;
pub const MAX_TENSOR_RANK: usize = 16;
pub const MAX_SHARDS: usize = 1024;

/// Sanity ceilings for config values (qwen3_5.rs:111-131).
pub const MAX_LAYERS: u64 = 1024;
pub const MAX_HIDDEN: u64 = 65_536;
pub const MAX_WIDTH: u64 = 1_048_576;
pub const MAX_HEADS: u64 = 1024;
pub const MAX_HEAD_DIM: u64 = 4096;
pub const MAX_EXPERTS: u64 = 1024;
pub const MAX_VOCAB: u64 = 4_194_304;
pub const MAX_POSITIONS: u64 = 1 << 32;
/// Host resource limit for a configured Sinkhorn loop, not a model default.
pub const MAX_SINKHORN_ITERS: u64 = 1024;

/// A whole text model as the splitter sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelShape {
    pub hidden: u64,
    pub vocab: u64,
    pub max_positions: u64,
    /// Decoder layers in execution order.
    pub layers: Vec<LayerShape>,
    /// Multi-token-prediction (draft) layers, outside the decoder count.
    pub draft: Vec<LayerShape>,
    pub residual: Residual,
    pub lexical: Option<LexicalShape>,
    /// Every stored tensor, sorted by name. Empty until headers are read.
    pub tensors: Vec<TensorEntry>,
}

/// One layer: a token mixer, then a feed-forward block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerShape {
    pub mixer: Mixer,
    pub ffn: Ffn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// Gated DeltaNet linear attention.
    Linear(LinearShape),
    Attention(AttentionShape),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearShape {
    pub key_heads: u64,
    pub value_heads: u64,
    pub key_dim: u64,
    pub value_dim: u64,
    pub conv: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionShape {
    pub query_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub rotary_dim: u64,
    /// q_proj stores gate rows next to the query rows (2 x head_dim per head).
    pub gated: bool,
    /// Sparse (indexed) attention; `None` means full attention.
    pub index: Option<IndexShape>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexShape {
    pub heads: u64,
    pub kv_heads: u64,
    pub dim: u64,
    pub budget: u64,
    pub compression: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ffn {
    Dense { width: u64 },
    Experts(ExpertShape),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertShape {
    pub count: u64,
    pub selected: u64,
    pub width: u64,
    /// Zero when the model has no shared expert.
    pub shared_width: u64,
    pub storage: ExpertStorage,
}

/// How routed experts are stored, read from the checkpoint index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertStorage {
    /// One 2-D tensor per expert and projection.
    PerExpert,
    /// One 3-D tensor per projection, experts on the first axis.
    Stacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residual {
    Single,
    /// Hyper-connection streams with a low-rank mixer.
    Streams { count: u64, rank: u64 },
}

/// Hashed n-gram lexical table, stored as `parts` blocks of `part_rows` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexicalShape {
    pub layer: u64,
    pub parts: u64,
    /// Read from the safetensors headers; zero until headers are read.
    pub part_rows: u64,
    pub row_dim: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub role: Role,
}

/// What a tensor is. The splitter plans from roles, never from names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Role {
    pub place: Place,
    pub part: Part,
    pub store: Store,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    Model,
    Layer(u32),
    /// Draft (MTP) tensors; `layer` is `None` for the draft's own head tensors.
    Draft { layer: Option<u32> },
    /// Stored but never resident for text serving (vision).
    Unused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Embedding,
    Head,
    Query,
    Key,
    Value,
    Output,
    QueryNorm,
    KeyNorm,
    Index,
    IndexNorm,
    LinearQkv,
    LinearConv,
    LinearZ,
    LinearOut,
    /// A_log, dt_bias, in_proj_a and in_proj_b: one row per value head.
    LinearHeads,
    LinearNorm,
    /// `id` is `None` for stacked storage (all experts in one tensor).
    Expert { id: Option<u32>, proj: Proj },
    Shared(Proj),
    Dense(Proj),
    Lexical { part: u32 },
    /// Router, norms, residual mixers, lexical projections and other small
    /// tensors that every rank keeps whole.
    Whole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proj {
    Gate,
    Up,
    /// Fused gate and up rows (stacked storage).
    GateUp,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Store {
    Weight,
    /// Block scales of a quantized weight (FP8 `weight_scale_inv`).
    BlockScale,
    /// Per-tensor scalar scales (NVFP4 `weight_scale_2`, `input_scale`).
    Scalar,
}
