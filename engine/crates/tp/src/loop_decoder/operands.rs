//! Each weight of a visit by its name, as a value of the compute backend.
//!
//! `Operands<V>` holds one value for each tensor of the load plan (`engine_model::plan::loop_load`).
//! `V` is the value type of the owner of the weights: a device vector, or host bytes in a test.
//! A caller asks by role and place, never by a stored name: `model` gives the weights that every
//! token reads, `layer` gives the weights of one stored layer, and `expert` gives the part of an
//! expert stack that belongs to one picked expert. A visit reads the weights of its stored layer.
//!
//! A weight that the description asks for and the checkpoint does not hold is a refusal that names
//! the layer and the role. No weight is made here and none is changed.
//!
//! The stored form of the experts is a fact of the load plan. Here an expert stack is one value of
//! `count` matrices, and matrix `i` starts `i` matrices into the value. A projection can have two
//! stacks: the widened one (`expert`) and the packed one (`packed_expert`), a unit a matrix.
//!
//! Layout of a value: the little-endian bytes of the tensor, row-major, in whole 32-bit words. A
//! BF16 matrix holds two values a word, the layout that the BF16 product of the device reads.
use crate::Result;
use engine_model::{plan::{description::{ConvTaps, LoopTextDescription, Visit},
    inkling_index::{ExpertPart, GatePart, MessageGate, StoredRole, Stream}, loop_load::{DEVICE_WORD, LoadTensor, LoopLoadPlan}, loop_packed_unit::UnitShape},
    shapes::{Place, Proj}};
use std::{collections::BTreeMap, ops::Range};

pub mod load;
pub mod weights;
#[cfg(target_os = "linux")]
pub mod device;
#[cfg(test)]
mod tests;

/// One tensor that the owner of the weights holds.
#[derive(Debug)]
pub struct Resident<V> { pub tensor: LoadTensor, pub value: V }
impl<V> Resident<V> {
    /// 32-bit words of the value.
    pub fn words(&self) -> Result<usize> { words(self.tensor.device_bytes().map_err(|e| e.to_string())?) }
}
fn words(bytes: u64) -> Result<usize> { usize::try_from(bytes / DEVICE_WORD).map_err(|_| "loop operands: a tensor exceeds the address space".into()) }

/// The part of an expert stack that belongs to one expert: `rows` by `columns` BF16 values.
#[derive(Debug)]
pub struct ExpertMatrix<'a, V> { pub value: &'a V, pub words: Range<usize>, pub rows: u64, pub columns: u64 }
/// The unit of one expert inside a packed stack: the words that a packed product reads, and the shape of the unit.
#[derive(Debug)]
pub struct PackedExpert<'a, V> { pub value: &'a V, pub words: Range<usize>, pub unit: UnitShape }
/// Taps of a key or value convolution.
#[derive(Debug)]
pub enum ConvOperands<'a, V> {
    /// One stored weight for each channel and tap.
    Fixed(&'a Resident<V>),
    /// Stored taps, and the projection that makes taps from the current token.
    Token { fixed: &'a Resident<V>, projection: &'a Resident<V> },
}
/// Norm weights of the attention inputs.
#[derive(Debug)]
pub enum InputNorms<'a, V> {
    One(&'a Resident<V>),
    Separate { query: &'a Resident<V>, key: &'a Resident<V>, value: &'a Resident<V> },
}
/// The three matrices of one message gate.
#[derive(Debug)]
pub struct GateOperands<'a, V> { pub down: &'a Resident<V>, pub forget: &'a Resident<V>, pub gain: &'a Resident<V> }
/// The four message gates of a layer in the loop range.
#[derive(Debug)]
pub struct MessageGates<'a, V> { pub query: GateOperands<'a, V>, pub key: GateOperands<'a, V>, pub value: GateOperands<'a, V>, pub expert: GateOperands<'a, V> }
/// Weights that every token reads one time.
#[derive(Debug)]
pub struct ModelOperands<'a, V> {
    pub embedding: &'a Resident<V>, pub embedding_norm: &'a Resident<V>,
    /// The used rows of the output head.
    pub head: &'a Resident<V>,
    pub final_norm: &'a Resident<V>, pub mix_output: &'a Resident<V>,
    /// Rows `2j` and `2j + 1`: the query and the expert message query of loop layer `j`. None without a loop range.
    pub message_queries: Option<&'a Resident<V>>,
    /// Rows `2j` and `2j + 1`: the key and the value message query of loop layer `j`.
    pub message_kv_queries: Option<&'a Resident<V>>,
}
/// Every weight of one stored layer.
#[derive(Debug)]
pub struct LayerOperands<'a, V> {
    pub layer: u32,
    pub input_norms: InputNorms<'a, V>,
    pub query: &'a Resident<V>, pub key: &'a Resident<V>, pub value: &'a Resident<V>,
    pub output: &'a Resident<V>, pub output_gate: &'a Resident<V>, pub sink: &'a Resident<V>, pub query_clip: &'a Resident<V>,
    pub relative_projection: &'a Resident<V>, pub relative_table: &'a Resident<V>, pub relative_clip: &'a Resident<V>,
    pub key_conv: ConvOperands<'a, V>, pub value_conv: ConvOperands<'a, V>,
    pub conv_after_attention: &'a Resident<V>, pub conv_after_experts: &'a Resident<V>,
    /// Block-mix query of the attention input: one row, or three rows (query, key, value). None in a layer that stores none.
    pub mix_attention: Option<&'a Resident<V>>,
    pub mix_experts: &'a Resident<V>,
    pub expert_norm: &'a Resident<V>, pub router: &'a Resident<V>, pub router_bias: &'a Resident<V>,
    /// All gate rows of an expert, then all its up rows.
    pub gate_up: &'a Resident<V>,
    pub down: &'a Resident<V>,
    /// The same two stacks in their stored form, a unit an expert. None in a load plan that holds no packed stack.
    pub gate_up_packed: Option<&'a Resident<V>>, pub down_packed: Option<&'a Resident<V>>,
    /// Only a layer in the loop range has message gates.
    pub gates: Option<MessageGates<'a, V>>,
}
impl<V> LayerOperands<'_, V> {
    /// Every weight of the layer, for a caller that counts or lists them.
    pub fn all(&self) -> Vec<&Resident<V>> {
        let mut out = vec![self.query, self.key, self.value, self.output, self.output_gate, self.sink, self.query_clip,
            self.relative_projection, self.relative_table, self.relative_clip, self.conv_after_attention, self.conv_after_experts,
            self.mix_experts, self.expert_norm, self.router, self.router_bias, self.gate_up, self.down];
        match &self.input_norms { InputNorms::One(n) => out.push(n), InputNorms::Separate { query, key, value } => out.extend([*query, *key, *value]) }
        for conv in [&self.key_conv, &self.value_conv] {
            match conv { ConvOperands::Fixed(w) => out.push(w), ConvOperands::Token { fixed, projection } => out.extend([*fixed, *projection]) }
        }
        out.extend(self.mix_attention);
        out.extend(self.gate_up_packed); out.extend(self.down_packed);
        if let Some(g) = &self.gates { for gate in [&g.query, &g.key, &g.value, &g.expert] { out.extend([gate.down, gate.forget, gate.gain]); } }
        out
    }
}

/// Matrices that one value holds one after another (`load::STACKED`). `members` gives each stored name and its words
/// in the value, in the order of the value. The value of each member is a view of those words: no word is held twice.
#[derive(Debug)]
pub struct Stack<V> { pub members: Vec<(String, Range<usize>)>, pub value: V }
#[derive(Debug)]
pub struct Operands<V> { held: BTreeMap<String, Resident<V>>, stacks: Vec<Stack<V>> }
impl<V> Operands<V> {
    /// One value for each tensor of the load plan. `value` makes or finds the value of one tensor.
    pub fn new(plan: &LoopLoadPlan, mut value: impl FnMut(&LoadTensor) -> Result<V>) -> Result<Self> {
        let mut held = BTreeMap::new();
        for (name, tensor) in &plan.tensors { held.insert(name.clone(), Resident { tensor: tensor.clone(), value: value(tensor)? }); }
        Ok(Self { held, stacks: Vec::new() })
    }
    /// The values that hold several tensors one after another. Empty where the owner of the weights keeps no views.
    pub fn stacks(&self) -> &[Stack<V>] { &self.stacks }
    pub fn len(&self) -> usize { self.held.len() }
    pub fn is_empty(&self) -> bool { self.held.is_empty() }
    /// Every held tensor, in name order.
    pub fn all(&self) -> impl Iterator<Item = &Resident<V>> { self.held.values() }
    pub(crate) fn all_mut(&mut self) -> impl Iterator<Item = &mut Resident<V>> { self.held.values_mut() }
    /// The tensor with this stored name.
    pub fn named(&self, name: &str) -> Result<&Resident<V>> {
        self.held.get(name).ok_or_else(|| format!("loop operands: the checkpoint holds no weight named {name}"))
    }
    fn find(&self, place: Place, role: StoredRole) -> Result<Option<&Resident<V>>> {
        let mut found = self.held.values().filter(|r| r.tensor.place == place && r.tensor.role == role && !r.tensor.form.is_packed());
        match (found.next(), found.next()) {
            (found, None) => Ok(found),
            _ => Err(format!("loop operands: {} holds more than one {role:?} weight", at(place))),
        }
    }
    /// The packed stack of one projection of one layer. None when the load plan holds none.
    fn packed_stack(&self, layer: u32, proj: Proj) -> Result<Option<&Resident<V>>> {
        let role = StoredRole::Expert { proj, part: ExpertPart::Codes };
        let mut found = self.held.values().filter(|r| r.tensor.place == Place::Layer(layer) && r.tensor.role == role && r.tensor.form.is_packed());
        match (found.next(), found.next()) {
            (found, None) => Ok(found),
            _ => Err(format!("loop operands: layer {layer} holds more than one packed {role:?} stack")),
        }
    }
    /// The one tensor with this role at this place.
    pub fn weight(&self, place: Place, role: StoredRole) -> Result<&Resident<V>> {
        self.find(place, role)?.ok_or_else(|| format!("loop operands: {} has no {role:?} weight: the checkpoint does not hold it", at(place)))
    }
    /// Weights that every token reads one time.
    pub fn model(&self, d: &LoopTextDescription) -> Result<ModelOperands<'_, V>> {
        use StoredRole::*;
        let m = |role| self.weight(Place::Model, role);
        let looped = d.layers.iter().any(|l| l.message_gates);
        Ok(ModelOperands { embedding: m(Embedding)?, embedding_norm: m(EmbeddingNorm)?, head: m(Head)?, final_norm: m(FinalNorm)?, mix_output: m(MixOutput)?,
            message_queries: looped.then(|| m(MessageQueries)).transpose()?, message_kv_queries: looped.then(|| m(MessageKvQueries)).transpose()? })
    }
    /// Every weight of one stored layer. The description says which weights the layer has.
    pub fn layer(&self, d: &LoopTextDescription, layer: u64) -> Result<LayerOperands<'_, V>> {
        use StoredRole::*;
        let description = usize::try_from(layer).ok().and_then(|i| d.layers.get(i)).ok_or_else(|| format!("loop operands: the model has no layer {layer}"))?;
        let layer = u32::try_from(layer).map_err(|_| "loop operands: layer number exceeds its range".to_owned())?;
        let w = |role| self.weight(Place::Layer(layer), role);
        let conv = |stream| Ok::<_, String>(match description.attention.key_value_conv.taps {
            ConvTaps::Fixed { .. } => ConvOperands::Fixed(w(ConvFixed(stream))?),
            ConvTaps::FixedAndToken { .. } => ConvOperands::Token { fixed: w(ConvStatic(stream))?, projection: w(ConvProjection(stream))? },
        });
        let gate = |gate| Ok::<_, String>(GateOperands { down: w(Gate { gate, part: GatePart::Down })?, forget: w(Gate { gate, part: GatePart::Forget })?, gain: w(Gate { gate, part: GatePart::Gain })? });
        // The widened stack of a projection, or its packed stack in a load of the packed stacks only
        // (`LoopLoadPlan::packed_experts_only`).
        let stack = |proj| match self.find(Place::Layer(layer), Expert { proj, part: ExpertPart::Codes })? {
            Some(found) => Ok(found),
            None => self.packed_stack(layer, proj)?.ok_or_else(|| format!("loop operands: layer {layer} has no {:?} weight: the checkpoint does not hold it", Expert { proj, part: ExpertPart::Codes })),
        };
        Ok(LayerOperands { layer,
            input_norms: if description.attention.separate_inputs { InputNorms::Separate { query: w(QueryInputNorm)?, key: w(KeyInputNorm)?, value: w(ValueInputNorm)? } }
                else { InputNorms::One(w(AttentionNorm)?) },
            query: w(Query)?, key: w(Key)?, value: w(Value)?, output: w(Output)?, output_gate: w(OutputGate)?, sink: w(Sink)?, query_clip: w(QueryClip)?,
            relative_projection: w(RelativeProjection)?, relative_table: w(RelativeTable)?, relative_clip: w(RelativeClip)?,
            key_conv: conv(Stream::Key)?, value_conv: conv(Stream::Value)?,
            conv_after_attention: w(ConvAfterAttention)?, conv_after_experts: w(ConvAfterExperts)?,
            mix_attention: (description.attention_queries > 0).then(|| w(MixAttention)).transpose()?, mix_experts: w(MixExperts)?,
            expert_norm: w(ExpertNorm)?, router: w(Router)?, router_bias: w(RouterBias)?, gate_up: stack(Proj::GateUp)?, down: stack(Proj::Down)?,
            gate_up_packed: self.packed_stack(layer, Proj::GateUp)?, down_packed: self.packed_stack(layer, Proj::Down)?,
            gates: description.message_gates.then(|| Ok::<_, String>(MessageGates { query: gate(MessageGate::Query)?, key: gate(MessageGate::Key)?,
                value: gate(MessageGate::Value)?, expert: gate(MessageGate::Expert)? })).transpose()? })
    }
    /// The weights of one visit: those of its stored layer. A visit that reads messages needs the message gates.
    pub fn visit(&self, d: &LoopTextDescription, visit: &Visit) -> Result<LayerOperands<'_, V>> {
        let weights = self.layer(d, visit.layer)?;
        if visit.messages && weights.gates.is_none() { return Err(format!("loop operands: visit of layer {} reads messages and the layer stores no message gates", visit.layer)); }
        Ok(weights)
    }
    /// The whole widened BF16 expert stack of one projection of one layer: `(value, experts, rows, columns)`.
    pub fn expert_stack(&self, layer: u32, proj: Proj) -> Result<(&V, usize, usize, usize)> {
        let stack = self.weight(Place::Layer(layer), StoredRole::Expert { proj, part: ExpertPart::Codes })?;
        let [count, rows, columns] = stack.tensor.stack().ok_or_else(|| format!("loop operands: {} is not an expert stack", stack.tensor.name))?;
        let to_usize = |n: u64| usize::try_from(n).map_err(|_| "loop operands: expert stack dimension exceeds address space".to_owned());
        Ok((&stack.value, to_usize(count)?, to_usize(rows)?, to_usize(columns)?))
    }
    /// The matrix of one expert inside the stack of one projection of one layer.
    pub fn expert(&self, layer: u32, proj: Proj, expert: u64) -> Result<ExpertMatrix<'_, V>> {
        let role = StoredRole::Expert { proj, part: ExpertPart::Codes };
        if self.find(Place::Layer(layer), role)?.is_none() && self.packed_stack(layer, proj)?.is_some() {
            return Err(format!("loop operands: layer {layer} holds its {proj:?} experts packed only (the widened stack is not loaded): this product reads the packed stack, and the compute gave no packed product for it"));
        }
        let stack = self.weight(Place::Layer(layer), role)?;
        let [count, rows, columns] = stack.tensor.stack().ok_or_else(|| format!("loop operands: {} is not an expert stack", stack.tensor.name))?;
        if expert >= count { return Err(format!("loop operands: layer {layer} has {count} experts and no expert {expert}")); }
        let bytes = stack.tensor.matrix_bytes().map_err(|e| e.to_string())?;
        if !bytes.is_multiple_of(DEVICE_WORD) { return Err(format!("loop operands: one matrix of {} does not fill whole device words", stack.tensor.name)); }
        let each = words(bytes)?;
        let start = each.checked_mul(usize::try_from(expert).map_err(|_| "loop operands: expert number exceeds its range".to_owned())?).ok_or("loop operands: expert offset overflow")?;
        Ok(ExpertMatrix { value: &stack.value, words: start..start + each, rows, columns })
    }
    /// The whole packed stack of one projection of one layer: its value, its count of experts and the unit shape of
    /// each, every unit at the same stride (`packed_expert`). None when the load plan holds no packed stack for it.
    pub fn packed_experts(&self, layer: u32, proj: Proj) -> Result<Option<(&V, usize, UnitShape)>> {
        let Some(stack) = self.packed_stack(layer, proj)? else { return Ok(None); };
        let (Some([count, ..]), Some(unit)) = (stack.tensor.stack(), stack.tensor.unit()) else { return Err(format!("loop operands: {} is not a packed expert stack", stack.tensor.name)); };
        let count = usize::try_from(count).map_err(|_| "loop operands: the expert count exceeds the address space".to_owned())?;
        Ok(Some((&stack.value, count, unit)))
    }
    /// The unit of one expert inside the packed stack of one projection of one layer. None when the load plan
    /// holds no packed stack for it.
    pub fn packed_expert(&self, layer: u32, proj: Proj, expert: u64) -> Result<Option<PackedExpert<'_, V>>> {
        let Some(stack) = self.packed_stack(layer, proj)? else { return Ok(None); };
        let (Some([count, ..]), Some(unit)) = (stack.tensor.stack(), stack.tensor.unit()) else { return Err(format!("loop operands: {} is not a packed expert stack", stack.tensor.name)); };
        if expert >= count { return Err(format!("loop operands: layer {layer} has {count} experts and no expert {expert}")); }
        let bytes = stack.tensor.matrix_bytes().map_err(|e| e.to_string())?;
        if !bytes.is_multiple_of(DEVICE_WORD) { return Err(format!("loop operands: one unit of {} does not fill whole device words", stack.tensor.name)); }
        let each = words(bytes)?;
        let start = each.checked_mul(usize::try_from(expert).map_err(|_| "loop operands: expert number exceeds its range".to_owned())?).ok_or("loop operands: expert offset overflow")?;
        Ok(Some(PackedExpert { value: &stack.value, words: start..start + each, unit }))
    }
}
fn at(place: Place) -> String { match place { Place::Layer(i) => format!("layer {i}"), Place::Model => "the model".into(), other => format!("{other:?}") } }
