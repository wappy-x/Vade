//! Metadata admission for schemas whose execution operations are not available.
use super::*;
use crate::{flash::mixed::{self, Mixed, Storage}, json::{self, Value}, safetensors::Dtype,
    shapes::MAX_TENSOR_RANK as MAX_TENSOR_DIMS};
use std::collections::BTreeSet;
type Object = BTreeMap<String, Value>;

struct Adapter {
    root: &'static str,
    text: &'static str,
    layer_prefix: &'static str,
    draft_count: &'static str,
    shared: &'static str,
    canonical_shared: &'static str,
    operations: &'static [&'static str],
}
// Names select metadata rules only. No execution path dispatches on these names.
const ADAPTERS: &[Adapter] = &[Adapter {
    root: "glm5_next", text: "glm5_next_text", layer_prefix: "model.language_model.layers.",
    draft_count: "num_nextn_predict_layers", shared: "mlp.shared_experts", canonical_shared: "mlp.shared_expert",
    operations: &[
        "KDA recurrence and separate convolution/gate tensor plan",
        "MLA low-rank projections and latent cache",
        "DSA indexer and compressed-pool selection",
        "mHC Sinkhorn residual mixing",
        "corrected sigmoid router (noaux_tc) and routed scaling",
        "capped SwiGLU",
        "per-layer dense/expert FFN execution schedule",
    ],
}];
const PROJECTIONS: &[(&str, Proj)] = &[("gate_proj", Proj::Gate), ("up_proj", Proj::Up), ("down_proj", Proj::Down)];
const MTP_GAP: &str = "appended MTP eh_proj/enorm/hnorm/shared_head.norm mapping (local public semantics missing)";

/// A parsed config can be useful before its execution operations exist.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigPlan { Supported(ModelPlan), Deferred(DeferredPlan) }

/// Stored names remain the keys for storage recipes and file lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappedTensor {
    pub stored: TensorEntry,
    pub plan_name: String,
}

/// Partial metadata only. This type cannot supply an executable ModelShape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredPlan {
    pub hidden: u64,
    pub decoder_layers: u64,
    pub draft_layers: u64,
    /// Validated decoder attention labels. Unknown operations fail admission.
    pub layer_types: Vec<String>,
    /// Raw decoder labels. Empty when the config omits the list.
    pub mlp_layer_types: Vec<String>,
    /// Absolute layer order: decoder layers, then appended draft layers.
    pub ffn_layers: Vec<FfnKind>,
    pub tensors: Vec<MappedTensor>,
    pub unmapped_tensors: Vec<String>,
    pub missing_ops: Vec<String>,
    shared_width: u64,
    adapter: usize,
    mixed: Mixed,
    storage_gaps: BTreeMap<String, String>,
}

impl ConfigPlan {
    /// Read config data. Deferred schemas stay separate from executable shapes.
    pub fn from_config(bytes: &[u8]) -> Result<Self> {
        let doc = json::parse(bytes, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })
            .map_err(|e| read_error("config.json", e))?;
        let root = object(&doc, "config")?;
        let name = string(get(root, "model_type")?, "model_type")?;
        if let Some((id, adapter)) = ADAPTERS.iter().enumerate().find(|(_, a)| name == a.root || name == a.text) {
            let text = if name == adapter.text { root } else { object(get(root, "text_config")?, "text_config")? };
            if string(get(text, "model_type")?, "model_type")? != adapter.text {
                return Err(invalid("model_type", "text schema differs from root schema"));
            }
            return deferred(root, text, id).map(Self::Deferred);
        }
        config::parse(bytes).map(Self::Supported)
    }
    /// Validate known header mappings. Unmapped deferred tensors remain explicit.
    pub fn with_headers(self, headers: &BTreeMap<String, Tensor>) -> Result<Self> {
        match self {
            Self::Supported(plan) => plan.with_headers(headers).map(Self::Supported),
            Self::Deferred(plan) => plan.with_headers(headers).map(Self::Deferred),
        }
    }
    /// Refuse before any weight load or execution when an operation is missing.
    pub fn into_model_plan(self) -> Result<ModelPlan> {
        match self {
            Self::Supported(plan) => Ok(plan),
            Self::Deferred(plan) => Err(PlanError::MissingOps { ops: plan.missing_ops }),
        }
    }
}

fn get<'a>(o: &'a Object, field: &'static str) -> Result<&'a Value> {
    o.get(field).ok_or(PlanError::Missing { field })
}
fn object<'a>(v: &'a Value, field: &'static str) -> Result<&'a Object> {
    v.object().map_err(|_| PlanError::WrongType { field, expected: "object" })
}
fn string<'a>(v: &'a Value, field: &'static str) -> Result<&'a str> {
    v.string().map_err(|_| PlanError::WrongType { field, expected: "string" })
}
fn count(o: &Object, field: &'static str, max: u64, zero: bool) -> Result<u64> {
    let n = get(o, field)?.u64().map_err(|_| PlanError::WrongType { field, expected: "unsigned integer" })?;
    if n > max || n == 0 && !zero { return Err(invalid(field, "count is outside its limit")); }
    Ok(n)
}
fn list(o: &Object, field: &'static str, count: u64) -> Result<Vec<String>> {
    let values = get(o, field)?.array().map_err(|_| PlanError::WrongType { field, expected: "array" })?;
    if values.len() as u64 != count { return Err(invalid(field, "length differs from num_hidden_layers")); }
    values.iter().map(|v| string(v, field).map(str::to_owned)).collect()
}
fn deferred(root: &Object, text: &Object, adapter: usize) -> Result<DeferredPlan> {
    let a = &ADAPTERS[adapter];
    let hidden = count(text, "hidden_size", MAX_HIDDEN, false)?;
    let decoder_layers = count(text, "num_hidden_layers", MAX_LAYERS, false)?;
    let draft_layers = count(text, a.draft_count, MAX_LAYERS, true)?;
    let layer_types = list(text, "layer_types", decoder_layers)?;
    if layer_types.iter().any(|kind| !matches!(kind.as_str(), "linear_attention" | "deepseek_sparse_attention")) {
        return Err(invalid("layer_types", "unknown attention operation"));
    }
    let mlp_layer_types = if text.contains_key("mlp_layer_types") { list(text, "mlp_layer_types", decoder_layers)? } else { Vec::new() };
    let ffn_layers = crate::host_ops::FfnSchedule::from_object(text, draft_layers)
        .map_err(|e| read_error("FFN layer schedule", e))?.layers;
    let width = count(text, "moe_intermediate_size", MAX_WIDTH, false)?;
    let shared = count(text, "n_shared_experts", MAX_EXPERTS, true)?;
    let shared_width = mul(width, shared)?;
    if shared_width > MAX_WIDTH { return Err(invalid("n_shared_experts", "combined shared width exceeds limit")); }
    let q = root.get("quantization_config").or_else(|| text.get("quantization_config"))
        .ok_or(PlanError::Missing { field: "quantization_config" })?;
    let q = object(q, "quantization_config")?;
    let (mixed, storage_gaps) = storage_metadata(q)?;
    let mut missing_ops = a.operations.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    if mixed.layers.values().any(|s| matches!(s, Storage::Fp8Block { .. })) {
        missing_ops.push("FP8_PB_WO weight_scale execution binding".into());
    }
    if draft_layers != 0 { missing_ops.push(MTP_GAP.into()); }
    if root.contains_key("vision_config") { missing_ops.push("vision/video tower, merger and input path".into()); }
    for (module, reason) in &storage_gaps { missing_ops.push(format!("{reason}: {module}")); }
    Ok(DeferredPlan { hidden, decoder_layers, draft_layers, layer_types, mlp_layer_types, ffn_layers,
        tensors: Vec::new(), unmapped_tensors: Vec::new(), missing_ops, shared_width, adapter, mixed, storage_gaps })
}

fn storage_metadata(q: &Object) -> Result<(Mixed, BTreeMap<String, String>)> {
    let mut validation = q.clone();
    let mut recipes = object(get(q, "quantized_layers")?, "quantized_layers")?.clone();
    let mut gaps = BTreeMap::new();
    for (module, value) in &mut recipes {
        let recipe = object(value, "quantized_layers")?;
        if let Some(encoding) = recipe.get("weight_scale_encoding") {
            let encoding = string(encoding, "weight_scale_encoding")?;
            if encoding != "csf" {
                if string(get(recipe, "quant_algo")?, "quant_algo")? != "NVFP4" {
                    return Err(invalid("weight_scale_encoding", "codec requires NVFP4 storage"));
                }
                gaps.insert(module.clone(), format!("scale codec and component layout {encoding}"));
                let mut plain = recipe.clone();
                plain.remove("weight_scale_encoding");
                *value = Value::Object(plain);
            }
        }
    }
    // Validate the rest of every recipe, including those with an unknown codec.
    // Then remove opaque recipes. They must never become raw-scale storage.
    validation.insert("quantized_layers".into(), Value::Object(recipes));
    let mut mixed = Mixed::parse(&validation).map_err(|e| read_error("quantization_config", e))?;
    for module in gaps.keys() { mixed.layers.remove(module); }
    Ok((mixed, gaps))
}

impl DeferredPlan {
    /// Unknown scale codecs fail closed unless an exclusion or child recipe wins.
    pub fn storage(&self, module: &str) -> Result<Storage> {
        if self.mixed.excluded(module) { return Ok(Storage::Plain); }
        let matches = |prefix: &str| module == prefix || module.strip_prefix(prefix).is_some_and(|tail| tail.starts_with('.'));
        let known = self.mixed.layers.keys().filter(|p| matches(p)).map(String::len).max().unwrap_or(0);
        if let Some((prefix, reason)) = self.storage_gaps.iter().filter(|(p, _)| matches(p)).max_by_key(|(p, _)| p.len())
            && prefix.len() >= known { return Err(PlanError::MissingOps { ops: vec![reason.clone()] }); }
        Ok(self.mixed.storage(module))
    }
    fn with_headers(mut self, headers: &BTreeMap<String, Tensor>) -> Result<Self> {
        if headers.len() > mixed::MAX_MIXED_TENSORS { return Err(invalid("tensors", "header count exceeds limit")); }
        let a = &ADAPTERS[self.adapter];
        let mut entries = BTreeMap::new();
        let mut canonical = BTreeMap::new();
        for (name, t) in headers {
            if t.shape.len() > MAX_TENSOR_DIMS { return Err(invalid("tensors", "header rank exceeds limit")); }
            mul(t.shape.iter().try_fold(1, |n, &d| mul(n, d))?, t.dtype.bytes())?;
            let Some(tail) = name.strip_prefix(a.layer_prefix) else { continue; };
            let (layer, tail) = tail.split_once('.').ok_or(invalid("tensor layer", "missing layer suffix"))?;
            let layer = layer.parse::<u64>().map_err(|_| invalid("tensor layer", "invalid layer index"))?;
            if layer >= add(self.decoder_layers, self.draft_layers)? {
                return Err(invalid("tensor layer", "outside decoder and appended draft range"));
            }
            // Appended MTP names stay opaque until a local public source defines them.
            if layer >= self.decoder_layers { continue; }
            for &(projection, proj) in PROJECTIONS {
                let module_tail = format!("{}.{}", a.shared, projection);
                if !tail.strip_prefix(&module_tail).is_some_and(|s| s.starts_with('.')) { continue; }
                if self.shared_width == 0 { return Err(invalid("n_shared_experts", "shared tensor without shared experts")); }
                if self.ffn_layers.get(layer as usize) != Some(&FfnKind::Experts) {
                    // Without a list, any expert layer proves a positive expert count.
                    // Then only the dense prefix can make this layer dense.
                    let sparse = self.ffn_layers.contains(&FfnKind::Experts);
                    let field = if !self.mlp_layer_types.is_empty() { "mlp_layer_types" }
                        else if sparse { "first_k_dense_replace" }
                        else { "first_k_dense_replace or expert count" };
                    return Err(invalid(field, "shared expert tensor outside a sparse layer"));
                }
                let module = format!("{}{layer}.{module_tail}", a.layer_prefix);
                if self.storage(&module).is_err() { continue; }
                let shape = if proj == Proj::Down { vec![self.hidden, self.shared_width] } else { vec![self.shared_width, self.hidden] };
                let stored = format!("{module}.weight");
                let role = Role { place: Place::Layer(u32::try_from(layer).map_err(|_| PlanError::Overflow { what: "layer id" })?),
                    part: Part::Shared(proj), store: Store::Weight };
                entries.insert(stored.clone(), TensorEntry { name: stored, dtype: Dtype::Bf16, shape, role });
                canonical.insert(module, format!("{}{layer}.{}.{projection}", a.layer_prefix, a.canonical_shared));
            }
        }
        mixed::inventory(&self.mixed, &mut entries, headers).map_err(|e| read_error("mixed tensor inventory", e))?;
        let mut mapped = Vec::new();
        for (name, entry) in &entries {
            let actual = headers.get(name).ok_or_else(|| PlanError::MissingTensor { name: name.clone() })?;
            if entry.shape != actual.shape { return Err(PlanError::TensorMismatch { name: name.clone(), expected: entry.shape.clone(), actual: actual.shape.clone() }); }
            if entry.dtype != actual.dtype { return Err(PlanError::Unsupported { field: "tensor dtype",
                value: format!("{name}: expected {:?}, actual {:?}", entry.dtype, actual.dtype) }); }
            let (module, suffix) = canonical.keys().find_map(|module| name.strip_prefix(module).map(|s| (module, s)))
                .ok_or(invalid("tensor mapping", "missing source module"))?;
            mapped.push(MappedTensor { stored: entry.clone(), plan_name: format!("{}{suffix}", canonical[module]) });
        }
        let mapped_names = entries.keys().collect::<BTreeSet<_>>();
        let mut unmapped = Vec::new();
        for name in headers.keys().filter(|n| !mapped_names.contains(n)) {
            if canonical.keys().any(|module| name.strip_prefix(module).is_some_and(|s| s.starts_with('.'))) {
                return Err(PlanError::UnexpectedTensor { name: name.clone() });
            }
            unmapped.push(name.clone());
        }
        self.tensors = mapped;
        self.unmapped_tensors = unmapped;
        Ok(self)
    }
}
