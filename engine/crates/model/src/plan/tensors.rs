//! Expected tensor names, storage, roles and header reconciliation.
use super::*;
use super::config::Family;
use crate::safetensors::Dtype;
const MODEL: &str = "model.language_model";

struct Inventory<'a> { plan: &'a ModelPlan, entries: BTreeMap<String, TensorEntry> }
impl Inventory<'_> {
    fn put(&mut self, name: String, dtype: Dtype, shape: Vec<u64>, role: Role) -> Result<()> {
        if self.entries.len() >= self.plan.tensor_limit() {
            return Err(invalid("tensors", "inventory exceeds limit"));
        }
        let elements = shape.iter().try_fold(1, |n, &v| mul(n, v))?;
        mul(elements, dtype.bytes())?;
        let entry = TensorEntry {
            name: name.clone(),
            dtype,
            shape,
            role,
        };
        self.entries.insert(name, entry);
        Ok(())
    }
    fn weight(&mut self, prefix: &str, suffix: &str, shape: Vec<u64>, place: Place, part: Part) -> Result<()> {
        let name = format!("{prefix}.{suffix}");
        let role = Role { place, part, store: Store::Weight };
        let quantized = match &self.plan.format {
            Format::Bf16 | Format::Mixed(_) => false,
            Format::Fp8 { unquantized, .. } => shape.len() == 2 && name.ends_with(".weight")
                && !name.ends_with("embed_tokens.weight")
                && !unquantized.iter().any(|s| name.strip_suffix(".weight").is_some_and(|m|
                    m == s || m.strip_prefix(s).is_some_and(|tail| tail.starts_with('.')))),
        };
        if quantized {
            let rows = shape.first().copied().ok_or(invalid("tensor shape", "missing rows"))?;
            let cols = shape.get(1).copied().ok_or(invalid("tensor shape", "missing columns"))?;
            self.put(format!("{name}_scale_inv"), Dtype::Bf16,
                vec![rows.div_ceil(FP8_BLOCK), cols.div_ceil(FP8_BLOCK)], Role { store: Store::BlockScale, ..role })?;
        }
        self.put(name, if quantized { Dtype::F8E4M3 } else { Dtype::Bf16 }, shape, role)
    }
    fn hc(&mut self, p: &str, place: Place, inject: bool) -> Result<()> {
        if let Residual::Streams { count, rank } = self.plan.shape.residual {
            let wide = mul(count, self.plan.shape.hidden)?;
            for (name, shape) in [("hc_norm.weight", vec![wide]),
                ("input_mix_weight_down.weight", vec![rank, wide]), ("input_mix_weight_up.weight", vec![wide, rank])] {
                self.weight(p, name, shape, place, Part::Whole)?;
            }
            if inject { self.weight(p, "block_inject_weight.weight", vec![count, wide], place, Part::Whole)?; }
        }
        Ok(())
    }
    fn projections(&mut self, p: &str, width: u64, place: Place, kind: impl Fn(Proj) -> Part) -> Result<()> {
        let hidden = self.plan.shape.hidden;
        for (name, proj, shape) in [("gate_proj.weight", Proj::Gate, vec![width, hidden]),
            ("up_proj.weight", Proj::Up, vec![width, hidden]), ("down_proj.weight", Proj::Down, vec![hidden, width])] {
            self.weight(p, name, shape, place, kind(proj))?;
        }
        Ok(())
    }
    fn layer(&mut self, p: &str, layer: LayerShape, place: Place) -> Result<()> {
        let hidden = self.plan.shape.hidden;
        if self.plan.schema.family == Family::Flash {
            for name in ["attn_hyper_connection", "mlp_hyper_connection"] { self.hc(&format!("{p}.{name}"), place, true)?; }
        } else {
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                self.weight(p, name, vec![hidden], place, Part::Whole)?;
            }
        }
        let mlp = format!("{p}.mlp");
        match layer.ffn {
            Ffn::Dense { width } => self.projections(&mlp, width, place, Part::Dense)?,
            Ffn::Experts(e) => {
                self.weight(&mlp, "gate.weight", vec![e.count, hidden], place, Part::Whole)?;
                let experts = format!("{mlp}.experts");
                match e.storage {
                    ExpertStorage::Stacked => {
                        self.weight(&experts, "gate_up_proj", vec![e.count, mul(2, e.width)?, hidden], place,
                            Part::Expert { id: None, proj: Proj::GateUp })?;
                        self.weight(&experts, "down_proj", vec![e.count, hidden, e.width], place,
                            Part::Expert { id: None, proj: Proj::Down })?;
                    }
                    ExpertStorage::PerExpert => for id in 0..e.count {
                        let expert = u32::try_from(id).map_err(|_| PlanError::Overflow { what: "expert id" })?;
                        self.projections(&format!("{experts}.{id}"), e.width, place,
                            |proj| Part::Expert { id: Some(expert), proj })?;
                    },
                }
                if e.shared_width != 0 {
                    self.projections(&format!("{mlp}.shared_expert"), e.shared_width, place, Part::Shared)?;
                    self.weight(&mlp, "shared_expert_gate.weight", vec![1, hidden], place, Part::Whole)?;
                }
            }
        }
        let (block, parts) = match layer.mixer {
            Mixer::Linear(s) => {
                let key = mul(s.key_heads, s.key_dim)?;
                let value = mul(s.value_heads, s.value_dim)?;
                let qkv = add(mul(2, key)?, value)?;
                ("linear_attn", vec![
                    ("A_log", vec![s.value_heads], Part::LinearHeads), ("dt_bias", vec![s.value_heads], Part::LinearHeads),
                    ("conv1d.weight", vec![qkv, 1, s.conv], Part::LinearConv),
                    ("in_proj_qkv.weight", vec![qkv, hidden], Part::LinearQkv),
                    ("in_proj_z.weight", vec![value, hidden], Part::LinearZ),
                    ("in_proj_a.weight", vec![s.value_heads, hidden], Part::LinearHeads),
                    ("in_proj_b.weight", vec![s.value_heads, hidden], Part::LinearHeads),
                    ("norm.weight", vec![s.value_dim], Part::LinearNorm),
                    ("out_proj.weight", vec![hidden, value], Part::LinearOut),
                ])
            }
            Mixer::Attention(s) => {
                let query = mul(s.query_heads, s.head_dim)?;
                let kv = mul(s.kv_heads, s.head_dim)?;
                let mut parts = vec![
                    ("q_proj.weight", vec![mul(if s.gated { 2 } else { 1 }, query)?, hidden], Part::Query),
                    ("k_proj.weight", vec![kv, hidden], Part::Key), ("v_proj.weight", vec![kv, hidden], Part::Value),
                    ("o_proj.weight", vec![hidden, query], Part::Output),
                    ("q_norm.weight", vec![s.head_dim], Part::QueryNorm), ("k_norm.weight", vec![s.head_dim], Part::KeyNorm),
                ];
                if let Some(i) = s.index {
                    parts.extend([("indexer.index_qk_proj.weight", vec![mul(add(i.heads, i.kv_heads)?, i.dim)?, hidden], Part::Index),
                        ("indexer.q_layernorm.weight", vec![i.dim], Part::IndexNorm),
                        ("indexer.k_layernorm.weight", vec![i.dim], Part::IndexNorm)]);
                }
                ("self_attn", parts)
            }
        };
        for (name, shape, part) in parts { self.weight(&format!("{p}.{block}"), name, shape, place, part)?; }
        Ok(())
    }
}
pub(super) fn expected(plan: &ModelPlan) -> Result<BTreeMap<String, TensorEntry>> {
    let mut i = Inventory { plan, entries: BTreeMap::new() };
    let s = &plan.shape;
    i.weight(MODEL, "embed_tokens.weight", vec![s.vocab, s.hidden], Place::Model, Part::Embedding)?;
    if !plan.schema.tied { i.weight("lm_head", "weight", vec![s.vocab, s.hidden], Place::Model, Part::Head)?; }
    if plan.schema.family == Family::Flash { i.hc(&format!("{MODEL}.hyper_connection_mixer"), Place::Model, false)?; }
    else { i.weight(MODEL, "norm.weight", vec![s.hidden], Place::Model, Part::Whole)?; }
    for (prefix, layers, draft) in [(format!("{MODEL}.layers"), &s.layers, false), ("mtp.layers".into(), &s.draft, true)] {
        for (layer, &shape) in layers.iter().enumerate() {
            let id = u32::try_from(layer).map_err(|_| PlanError::Overflow { what: "layer id" })?;
            i.layer(&format!("{prefix}.{layer}"), shape, if draft { Place::Draft { layer: Some(id) } } else { Place::Layer(id) })?;
        }
    }
    if !s.draft.is_empty() {
        let place = Place::Draft { layer: None };
        if plan.schema.family == Family::Flash {
            i.hc("mtp.hyper_connection_mixer", place, false)?;
            let wide = match s.residual { Residual::Single => s.hidden, Residual::Streams { count, .. } => mul(count, s.hidden)? };
            for (name, shape) in [("pre_fc_norm_embedding.weight", vec![s.hidden]), ("pre_fc_norm_hidden.weight", vec![wide]),
                ("fc_embedding.weight", vec![s.hidden, s.hidden]), ("fc_hidden.weight", vec![s.hidden, s.hidden])] {
                i.weight("mtp", name, shape, place, Part::Whole)?;
            }
        } else {
            i.weight("mtp", "fc.weight", vec![s.hidden, mul(2, s.hidden)?], place, Part::Whole)?;
            for name in ["norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"] {
                i.weight("mtp", name, vec![s.hidden], place, Part::Whole)?;
            }
        }
    }
    if let (Some(l), Some(ple)) = (s.lexical, plan.schema.ple) {
        let place = Place::Layer(u32::try_from(l.layer).map_err(|_| PlanError::Overflow { what: "lexical layer" })?);
        let prefix = format!("{MODEL}.layers.{}.ple", l.layer);
        let wide = match s.residual { Residual::Single => s.hidden, Residual::Streams { count, .. } => mul(count, s.hidden)? };
        for (name, shape) in [("conv1d.weight", vec![wide, 1, ple.conv]), ("key_proj.weight", vec![wide, ple.embed]),
            ("value_proj.weight", vec![s.hidden, ple.embed]), ("norm_conv.weight", vec![wide]),
            ("norm_key.weight", vec![wide]), ("norm_query.weight", vec![wide])] {
            i.weight(&prefix, name, shape, place, Part::Whole)?;
        }
        let embedding = format!("{prefix}.ple_embedding");
        let degree = ple.size.checked_sub(1).ok_or(invalid("ngram_size", "must exceed one"))?;
        for (name, length) in [("layer_multipliers", ple.size), ("ngram_heads_offsets", mul(ple.heads, degree)?),
            ("ngram_heads_vocab_sizes", mul(ple.heads, degree)?)] {
            i.put(format!("{embedding}.{name}"), Dtype::I64, vec![length], Role { place, part: Part::Whole, store: Store::Weight })?;
        }
        i.put(format!("{embedding}.ngram_embedding.weight_scale"), Dtype::Bf16, vec![1],
            Role { place, part: Part::Whole, store: Store::Weight })?;
        // Flash's lexical shards use one global scale, never per-block scales.
        for part in 0..l.parts {
            let id = u32::try_from(part).map_err(|_| PlanError::Overflow { what: "lexical part" })?;
            i.put(format!("{embedding}.ngram_embedding.shard_{part}.weight"), Dtype::F8E4M3, vec![l.part_rows, l.row_dim],
                Role { place, part: Part::Lexical { part: id }, store: Store::Weight })?;
        }
    }
    Ok(i.entries)
}
pub(super) fn reconcile(
    mut plan: ModelPlan,
    headers: &BTreeMap<String, Tensor>,
) -> Result<ModelPlan> {
    if headers.len() > plan.tensor_limit() {
        return Err(invalid("tensors", "header count exceeds limit"));
    }
    // A stored head is independent; without it, the embedding supplies logits.
    let tied = !headers.contains_key("lm_head.weight");
    if plan.schema.declared_tied.is_some_and(|v| v != tied) {
        return Err(invalid("tie_word_embeddings", "config contradicts checkpoint head storage"));
    }
    plan.schema.tied = tied;
    for t in headers.values() {
        if t.shape.len() > MAX_TENSOR_RANK { return Err(invalid("tensors", "header rank exceeds limit")); }
        mul(t.shape.iter().try_fold(1, |n, &v| mul(n, v))?, t.dtype.bytes())?;
    }
    if plan.schema.family == Family::Hybrid {
        let storage = if headers.keys().any(|n| (n.starts_with(&format!("{MODEL}.layers.")) || n.starts_with("mtp.layers."))
            && n.ends_with(".mlp.experts.0.gate_proj.weight")) {
            ExpertStorage::PerExpert
        } else { ExpertStorage::Stacked };
        for l in plan.shape.layers.iter_mut().chain(&mut plan.shape.draft) {
            if let Ffn::Experts(e) = &mut l.ffn { e.storage = storage; }
        }
    }
    if let Some(l) = &mut plan.shape.lexical {
        let name = format!("{MODEL}.layers.{}.ple.ple_embedding.ngram_embedding.shard_0.weight", l.layer);
        let t = headers.get(&name).ok_or_else(|| PlanError::MissingTensor { name: name.clone() })?;
        let rows = t.shape.first().copied().ok_or_else(|| PlanError::TensorMismatch {
            name: name.clone(), expected: vec![0, l.row_dim], actual: t.shape.clone() })?;
        if rows == 0 || rows > MAX_POSITIONS { return Err(invalid("lexical part_rows", "zero or above its limit")); }
        mul(rows, l.parts)?;
        l.part_rows = rows;
    }
    let mut inventory = expected(&plan)?;
    // Served code reads singular shared expert names. Refuse the plural public names here.
    // The deferred adapter maps the plural names itself.
    for e in inventory.values().filter(|e| matches!(e.role.part, Part::Shared(_))) {
        let plural = e.name.replace(".mlp.shared_expert.", ".mlp.shared_experts.");
        if plural != e.name && headers.contains_key(&plural) {
            return Err(PlanError::Unsupported { field: "shared expert tensor name",
                value: format!("{plural}: the supported plan reads {}", e.name) });
        }
    }
    if let Format::Mixed(q) = &plan.format {
        crate::flash::mixed::inventory(q, &mut inventory, headers)
            .map_err(|e| super::read_error("mixed tensor inventory", e))?;
    }
    for (name, entry) in &mut inventory {
        let t = headers.get(name).ok_or_else(|| PlanError::MissingTensor { name: name.clone() })?;
        if t.shape != entry.shape { return Err(PlanError::TensorMismatch {
            name: name.clone(), expected: entry.shape.clone(), actual: t.shape.clone() }); }
        if plan.schema.family == Family::Hybrid && t.dtype == Dtype::F32 && (name.ends_with(".linear_attn.A_log")
            || name.ends_with(".linear_attn.dt_bias") || name.ends_with(".linear_attn.norm.weight")) { entry.dtype = t.dtype; }
        if t.dtype != entry.dtype { return Err(PlanError::Unsupported { field: "tensor dtype",
            value: format!("{name}: expected {:?}, actual {:?}", entry.dtype, t.dtype) }); }
    }
    for (name, t) in headers {
        if inventory.contains_key(name) { continue; }
        if !name.starts_with("model.visual.") { return Err(PlanError::UnexpectedTensor { name: name.clone() }); }
        let store = if name.ends_with("_scale_inv") { Store::BlockScale }
            else if name.ends_with("weight_scale_2") || name.ends_with("input_scale") { Store::Scalar } else { Store::Weight };
        inventory.insert(name.clone(), TensorEntry { name: name.clone(), dtype: t.dtype, shape: t.shape.clone(),
            role: Role { place: Place::Unused, part: Part::Whole, store } });
    }
    plan.shape.tensors = inventory.into_values().collect();
    Ok(plan)
}
