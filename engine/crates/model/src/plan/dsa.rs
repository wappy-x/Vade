//! Logical tensor and selection metadata for a compressed-pool sparse indexer.
//! This plan does not admit a model or choose a device execution path.
use super::{PlanError, Result, add, invalid, mul, read_error};
use crate::{json::{self, Value}, safetensors::{Dtype, Tensor}, shapes::*};
use std::{collections::BTreeMap, ops::Range};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub hidden: u64,
    /// Input from the caller's normalized low-rank Q projection.
    pub query_rank: u64,
    pub heads: u64,
    pub head_dim: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightRole { Query, Key, HeadWeight, KeyNormGain, KeyNormBias, PoolPosition, PoolGate }

/// Logical shapes only. Matrix axes are `[output, input]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightPlan {
    pub role: WeightRole,
    /// Source module suffix. The caller supplies the checkpoint module prefix.
    pub name: &'static str,
    pub shape: Vec<u64>,
}

/// Header evidence for an unpacked floating-point weight. No payload is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundWeight { pub role: WeightRole, pub name: String, pub tensor: Tensor }

/// One packed cache row is `[key channels, gate channels, validity]`.
/// APE axes are `[pool slot, channel]`. Compression applies a separate softmax
/// over slots for each channel, then a weighted sum of keys over those slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolPlan {
    pub tokens: u64,
    pub channels: u64,
    pub key_offset: u64,
    pub gate_offset: u64,
    pub valid_offset: u64,
    pub packed_width: u64,
}

/// The public source does not define the order of equal scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieOrder { Unspecified }

/// Per-query budget. Score complete pools, then expand them in slot order.
/// The optional tail is extra. Duplicate indices form a set in the sparse mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionPlan {
    pub token_budget: u64,
    pub pool_budget: u64,
    pub tail_width: u64,
    pub output_width: u64,
    pub index_dtype: Dtype,
    pub invalid_index: i32,
    pub tie_order: TieOrder,
}

/// Logical axes before removal of pool slots invalid in every batch row.
/// The pool axis is an upper bound. Runtime validity determines its final size.
/// This describes tensors, not a persistent compressed-cache allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolLayout {
    pub packed: [u64; 3],
    pub keys: [u64; 3],
    pub indices: [u64; 3],
    pub valid: [u64; 2],
}

/// Eligible raw-slot ranges for one query and one sequence, before scoring.
/// No score or ranking is computed here. Invalid queries have empty lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuerySelection {
    pub candidate_pools: Vec<Range<usize>>,
    pub select_pools: usize,
    pub tail_indices: Vec<usize>,
}

/// Standalone metadata. It cannot be converted into an executable ModelPlan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaPlan {
    geometry: Geometry,
    pool: PoolPlan,
    selection: SelectionPlan,
    weights: Vec<WeightPlan>,
}
impl DsaPlan {
    /// Read explicit fields from a text config or nested `text_config`.
    /// Model names, layer schedules and storage recipes do not select behavior.
    pub fn from_config(bytes: &[u8]) -> Result<Self> {
        let doc = json::parse(bytes, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })
            .map_err(|e| read_error("config.json", e))?;
        let root = object(&doc, "config")?;
        let text = root.get("text_config").map(|v| object(v, "text_config")).transpose()?.unwrap_or(root);
        let geometry = Geometry { hidden: count(text, "hidden_size", MAX_HIDDEN)?,
            query_rank: count(text, "q_lora_rank", MAX_HIDDEN)?,
            heads: count(text, "index_n_heads", MAX_HEADS)?,
            head_dim: count(text, "index_head_dim", MAX_HEAD_DIM)? };
        let tokens = count(text, "index_kpool", i32::MAX as u64)?;
        let token_budget = count(text, "index_topk", i32::MAX as u64)?;
        let tail = boolean(get(text, "index_kpool_always_select_tail")?, "index_kpool_always_select_tail")?;
        // The source always compresses. It defines no operation for false.
        if let Some(v) = text.get("index_kpool_compress") {
            if !boolean(v, "index_kpool_compress")? {
                return Err(invalid("index_kpool_compress", "uncompressed pools have no source operation"));
            }
        }
        let valid_offset = mul(geometry.head_dim, 2)?;
        let pool = PoolPlan { tokens, channels: geometry.head_dim, key_offset: 0,
            gate_offset: geometry.head_dim, valid_offset, packed_width: add(valid_offset, 1)? };
        let tail_width = if tail { tokens - 1 } else { 0 };
        let output_width = add(token_budget, tail_width)?;
        if output_width > i32::MAX as u64 { return Err(invalid("index_topk", "output width exceeds the plan limit")); }
        let selection = SelectionPlan { token_budget, pool_budget: token_budget / tokens,
            tail_width, output_width, index_dtype: Dtype::I32, invalid_index: -1, tie_order: TieOrder::Unspecified };
        let g = geometry;
        let weights = [
            (WeightRole::Query, "wq_b.weight", vec![mul(g.heads, g.head_dim)?, g.query_rank]),
            (WeightRole::Key, "wk.weight", vec![g.head_dim, g.hidden]),
            (WeightRole::HeadWeight, "weights_proj.weight", vec![g.heads, g.hidden]),
            (WeightRole::KeyNormGain, "k_norm.weight", vec![g.head_dim]),
            (WeightRole::KeyNormBias, "k_norm.bias", vec![g.head_dim]),
            (WeightRole::PoolPosition, "index_kpool_compress_ape", vec![tokens, g.head_dim]),
            (WeightRole::PoolGate, "index_kpool_compress_gate", vec![g.head_dim, g.hidden]),
        ].into_iter().map(|(role, name, shape)| WeightPlan { role, name, shape }).collect::<Vec<_>>();
        for weight in &weights { elements(&weight.shape)?; }
        Ok(Self { geometry, pool, selection, weights })
    }
    pub fn geometry(&self) -> Geometry { self.geometry }
    pub fn pool(&self) -> &PoolPlan { &self.pool }
    pub fn selection(&self) -> &SelectionPlan { &self.selection }
    pub fn weights(&self) -> &[WeightPlan] { &self.weights }

    /// Bind only the supplied indexer subtree. Preserve stored dtype and offsets.
    /// Packed weights, scales and other companions need a storage plan first.
    pub fn with_headers(&self, prefix: &str, headers: &BTreeMap<String, Tensor>) -> Result<Vec<BoundWeight>> {
        if prefix.is_empty() || prefix.split('.').any(str::is_empty) {
            return Err(invalid("indexer prefix", "must contain nonempty module components"));
        }
        if headers.len() > crate::flash::mixed::MAX_MIXED_TENSORS {
            return Err(invalid("tensors", "header count exceeds limit"));
        }
        let prefix = format!("{prefix}.");
        let mut bound = Vec::new();
        for weight in &self.weights {
            let name = format!("{prefix}{}", weight.name);
            let tensor = headers.get(&name).ok_or_else(|| PlanError::MissingTensor { name: name.clone() })?;
            if tensor.shape != weight.shape {
                return Err(PlanError::TensorMismatch { name, expected: weight.shape.clone(), actual: tensor.shape.clone() });
            }
            if !matches!(tensor.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
                return Err(PlanError::Unsupported { field: "DSA storage", value: format!("{name}: {:?}; unpacked floating-point weights required", tensor.dtype) });
            }
            if tensor.end.checked_sub(tensor.start) != Some(mul(elements(&tensor.shape)?, tensor.dtype.bytes())?) {
                return Err(invalid("tensor offsets", "byte span differs from shape and dtype"));
            }
            bound.push(BoundWeight { role: weight.role, name, tensor: tensor.clone() });
        }
        for name in headers.keys() {
            if let Some(suffix) = name.strip_prefix(&prefix) {
                if !self.weights.iter().any(|w| w.name == suffix) {
                    return Err(PlanError::UnexpectedTensor { name: name.clone() });
                }
            }
        }
        Ok(bound)
    }

    /// Plan packed rows and the maximum pool slots. Cache dtype stays unspecified.
    pub fn layout(&self, batch: u64, cache_slots: u64) -> Result<PoolLayout> {
        if batch == 0 || batch > isize::MAX as u64 { return Err(invalid("batch", "outside the address range")); }
        if cache_slots > i32::MAX as u64 { return Err(invalid("cache slots", "exceeds the index limit")); }
        let pools = cache_slots.div_ceil(self.pool.tokens);
        let layout = PoolLayout { packed: [batch, cache_slots, self.pool.packed_width],
            keys: [batch, pools, self.pool.channels], indices: [batch, pools, self.pool.tokens], valid: [batch, pools] };
        for shape in [&layout.packed[..], &layout.keys[..], &layout.indices[..], &layout.valid[..]] { elements(shape)?; }
        Ok(layout)
    }

    /// Pools start at the first valid raw slot. A hole invalidates its pool.
    /// The tail uses the source's visible-valid-count formula, not compaction.
    pub fn query(&self, valid: &[bool], position: usize) -> Result<QuerySelection> {
        if position >= valid.len() { return Err(invalid("query position", "outside the cache")); }
        self.layout(1, valid.len() as u64)?;
        let mut result = QuerySelection { candidate_pools: Vec::new(), select_pools: 0, tail_indices: Vec::new() };
        if !valid[position] { return Ok(result); }
        let first = valid.iter().position(|&v| v).unwrap_or(valid.len());
        let tokens = usize::try_from(self.pool.tokens).map_err(|_| PlanError::Overflow { what: "pool slots" })?;
        for group in 0..(position + 1 - first) / tokens {
            let start = first + group * tokens;
            let end = start + tokens;
            if valid[start..end].iter().all(|&v| v) { result.candidate_pools.push(start..end); }
        }
        result.select_pools = result.candidate_pools.len().min(self.selection.pool_budget as usize);
        if self.selection.tail_width != 0 {
            let visible = valid[..=position].iter().filter(|&&v| v).count();
            let tail = visible % tokens;
            let start = first + visible - tail;
            for index in start..start + tail {
                if index <= position && valid[index] { result.tail_indices.push(index); }
            }
        }
        Ok(result)
    }
}

type Object = BTreeMap<String, Value>;
fn get<'a>(o: &'a Object, field: &'static str) -> Result<&'a Value> {
    o.get(field).ok_or(PlanError::Missing { field })
}
fn object<'a>(v: &'a Value, field: &'static str) -> Result<&'a Object> {
    v.object().map_err(|_| PlanError::WrongType { field, expected: "object" })
}
fn boolean(v: &Value, field: &'static str) -> Result<bool> {
    v.boolean().map_err(|_| PlanError::WrongType { field, expected: "boolean" })
}
fn count(o: &Object, field: &'static str, max: u64) -> Result<u64> {
    let n = get(o, field)?.u64().map_err(|_| PlanError::WrongType { field, expected: "unsigned integer" })?;
    if n == 0 || n > max { return Err(invalid(field, "count is outside its limit")); }
    Ok(n)
}
fn elements(shape: &[u64]) -> Result<u64> {
    let count = shape.iter().try_fold(1, |n, &d| mul(n, d))?;
    // Check the address range with the widest scalar used by these layouts.
    if mul(count, size_of::<u64>() as u64)? > isize::MAX as u64 {
        return Err(invalid("tensor shape", "exceeds the address range"));
    }
    Ok(count)
}
