//! Host descriptions for operations that do not yet have an execution binding.
use super::{ModelPlan, PlanError, Rope};
use crate::{json::{self, Value}, shapes::*};
use super::storage::StorageMap;
use crate::host_ops::router::RouterKind;

/// Schema selection occurs once while reading config data. Kernels must use
/// the operation fields below. A deferred description cannot supply ModelShape.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigPlan { Supported(ModelPlan), Deferred(TextDescription) }
impl ConfigPlan {
    pub fn from_config(bytes: &[u8]) -> super::Result<Self> {
        let doc = json::parse(bytes, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })
            .map_err(|e| super::read_error("config.json", e))?;
        let root = doc.object().map_err(|_| PlanError::WrongType { field: "config", expected: "object" })?;
        match root.get("model_type").map(Value::string).transpose().map_err(|e| super::read_error("model_type", e))? {
            Some("glm5_next" | "glm5_next_text") => super::glm::parse(&doc).map(Self::Deferred),
            _ => ModelPlan::from_config(bytes).map(Self::Supported),
        }
    }
    pub fn into_model_plan(self) -> super::Result<ModelPlan> {
        match self {
            Self::Supported(plan) => Ok(plan),
            Self::Deferred(_) => Err(PlanError::Unsupported { field: "execution",
                value: "deferred operation description; kernel and header checks are incomplete".into() }),
        }
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
