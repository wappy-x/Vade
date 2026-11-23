//! The load plan of the looped text family: what each device tensor is made from. Host metadata only.
//! This module reads headers and no payload. It allocates nothing on a device and admits no execution.
//!
//! Each device tensor has one of four forms:
//! - `Stored`: the bytes of one stored tensor, unchanged (the BF16 and F32 tensors).
//! - `LeadingRows`: the first rows of a stored matrix. The output head stores padded rows that no
//!   token uses (`unpadded_vocab_size` in the config).
//! - `WidenedExperts`: one stack of expert matrices as BF16. It is made from the stored codes, the
//!   block scales and, for 4-bit storage, the factor of each expert. A matrix that holds gate rows
//!   and up rows is un-mixed: all gate rows, then all up rows.
//! - `PackedExperts`: one stack of expert matrices in their stored 4-bit or 8-bit form, a unit a
//!   matrix (`loop_packed_unit.rs`): the stored codes, the stored block scales and a weight table.
//!   The rows keep their stored order.
//!
//! `expert_tensor` is the one place that decides the device forms of the experts. The reference code
//! holds them as BF16 (`loop_widen.rs`). Today the function gives both forms of each stack: the
//! product of one input row reads the packed stack, and the product of many input rows reads the
//! widened stack until it has a packed kernel. The packed stack has the name of the widened stack
//! with `PACKED_NAME` after it; a lookup by role gives the widened stack, and `packed_experts`
//! gives the packed one.
//!
//! Reference: transformers v5.24.0. This file follows these lines and holds its own code.
//! - `core_model_loading.py:772-816`: the widened expert stack and its sources.
//! - `core_model_loading.py:856-878`: the head keeps the first `unpadded_vocab_size` rows.
//! - `conversion_mapping.py:2658-2667`: gate rows first, then up rows.
use super::{PlanError, Result, description::LoopTextDescription, read_error,
    inkling_index::{self, BoundIndex, BoundTensor, ExpertPart, StoredRole, TensorState},
    loop_packed::{PackedCodes, PackedMatrix}, loop_packed_unit::UnitShape, loop_widen, storage_groups::{ExpertRows, GATE_UP_ROWS, NVFP4_NIBBLES}};
use crate::{safetensors::{self, Dtype, Header, Index, Limits},
    shapes::{CONFIG_BYTES, HEADER_BYTES, MAX_SHARDS, MAX_TENSORS, MAX_TENSOR_RANK, Place, Proj}};
use std::{collections::BTreeMap, path::Path};

/// A device value is made of 32-bit words. A tensor with another byte count takes the next whole word.
pub const DEVICE_WORD: u64 = 4;
fn mul(a: u64, b: u64) -> Result<u64> { super::mul(a, b) }
fn add(a: u64, b: u64) -> Result<u64> { super::add(a, b) }
/// Bytes that a tensor of `bytes` payload bytes takes on the device.
pub fn device_extent(bytes: u64) -> Result<u64> { mul(add(bytes, DEVICE_WORD - 1)? / DEVICE_WORD, DEVICE_WORD) }

/// The stored form of an expert stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertCodes {
    /// Two 4-bit codes a byte, one 8-bit float scale for 16 values of a row, one factor for each expert.
    Nvfp4,
    /// One 8-bit float a weight, one power-of-two scale for 32 values of a row.
    Mxfp8,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadForm {
    Stored,
    LeadingRows,
    /// `rows` is `Some` for a matrix whose stored rows mix gate rows and up rows.
    WidenedExperts { codes: ExpertCodes, rows: Option<ExpertRows> },
    /// The stored form, a unit a matrix. `rows` is `Some` for a matrix whose stored rows mix gate rows and up rows.
    PackedExperts { codes: ExpertCodes, rows: Option<ExpertRows> },
}
impl LoadForm {
    pub fn is_packed(&self) -> bool { matches!(self, LoadForm::PackedExperts { .. }) }
}
/// The name of a packed expert stack is the name of its code tensor and this text. No stored name ends with it.
pub const PACKED_NAME: &str = "#packed";
/// What a file part is to its device tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartKind { Whole, Codes, BlockScale, ExpertScale, ScaleDivisor }
/// A byte range of one weight file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilePart {
    pub kind: PartKind,
    /// Stored tensor name.
    pub name: String,
    pub shard: String,
    pub dtype: Dtype,
    /// Shape of the part that is read. For `LeadingRows` this is the cut shape.
    pub shape: Vec<u64>,
    /// Byte offset from the start of the file.
    pub offset: u64,
    pub bytes: u64,
}
/// One tensor on the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadTensor {
    /// The stored name of the tensor, or of the code tensor of an expert stack.
    pub name: String,
    pub role: StoredRole,
    pub place: Place,
    pub form: LoadForm,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Payload bytes on the device.
    pub bytes: u64,
    pub parts: Vec<FilePart>,
}
impl LoadTensor {
    /// Bytes this tensor takes on the device.
    pub fn device_bytes(&self) -> Result<u64> { device_extent(self.bytes) }
    /// Bytes this tensor reads from the weight files.
    pub fn file_bytes(&self) -> u64 { self.parts.iter().map(|p| p.bytes).sum() }
    pub fn part(&self, kind: PartKind) -> Option<&FilePart> { self.parts.iter().find(|p| p.kind == kind) }
    /// Matrices, rows and columns of an expert stack, widened or packed.
    pub fn stack(&self) -> Option<[u64; 3]> {
        match (self.form, self.shape.as_slice()) { (LoadForm::WidenedExperts { .. } | LoadForm::PackedExperts { .. }, &[n, r, c]) => Some([n, r, c]), _ => None }
    }
    /// One matrix of a packed stack: its stored form, rows, columns and row order. None for any other tensor.
    pub fn unit(&self) -> Option<UnitShape> {
        let (LoadForm::PackedExperts { codes, rows: order }, Some([_, rows, columns])) = (self.form, self.stack()) else { return None; };
        Some(UnitShape { codes, rows: usize::try_from(rows).ok()?, columns: usize::try_from(columns).ok()?, order })
    }
    /// Device bytes of one matrix of an expert stack. Matrix `i` starts at byte `i` times this value.
    pub fn matrix_bytes(&self) -> Result<u64> {
        let [_, rows, columns] = self.stack().ok_or(PlanError::Invalid { field: "expert storage", reason: "the tensor is not an expert stack" })?;
        if !self.form.is_packed() { return mul(mul(rows, columns)?, Dtype::Bf16.bytes()); }
        let unit = self.unit().ok_or(PlanError::Invalid { field: "expert storage", reason: "a packed matrix exceeds the address space" })?;
        Ok(unit.layout().map_err(|_| PlanError::Invalid { field: "expert storage", reason: "the rows and columns are not whole blocks of a packed matrix" })?.bytes as u64)
    }
    /// The file bytes of one part that belong to matrix `matrix` of a stack, as (byte offset in the file, bytes).
    /// A part with one value for the whole stack gives its whole range.
    pub fn matrix_part(&self, kind: PartKind, matrix: u64) -> Result<(u64, u64)> {
        let [count, ..] = self.stack().ok_or(PlanError::Invalid { field: "expert storage", reason: "the tensor is not an expert stack" })?;
        let part = self.part(kind).ok_or(PlanError::Invalid { field: "expert storage", reason: "the stack has no such part" })?;
        if matrix >= count { return Err(PlanError::Invalid { field: "expert", reason: "the stack has no matrix with this number" }); }
        if part.shape.first() != Some(&count) { return Ok((part.offset, part.bytes)); }
        let each = part.bytes / count;
        Ok((add(part.offset, mul(each, matrix)?)?, each))
    }
    /// Make the device bytes of one matrix of a widened stack from the stored bytes of its parts.
    /// `expert_scale` and `divisor` are the 32-bit values of a 4-bit stack; an 8-bit stack has none.
    pub fn widen(&self, codes: &[u8], scales: &[u8], expert_scale: Option<f32>, divisor: Option<f32>, out: &mut [u8]) -> crate::Result<()> {
        let (LoadForm::WidenedExperts { codes: form, rows: order }, Some([_, rows, columns])) = (self.form, self.stack()) else {
            return Err(crate::invalid("expert widen: the tensor is not an expert stack"));
        };
        let size = |n: u64| usize::try_from(n).map_err(|_| crate::invalid("expert widen: a size exceeds the address space"));
        let (rows, columns) = (size(rows)?, size(columns)?);
        match (form, expert_scale, divisor) {
            (ExpertCodes::Nvfp4, Some(amax), Some(divisor)) =>
                loop_widen::widen_nvfp4(codes, scales, loop_widen::expert_factor(amax, divisor)?, rows, columns, order, NVFP4_NIBBLES, out),
            (ExpertCodes::Mxfp8, None, None) => loop_widen::widen_mxfp8(codes, scales, rows, columns, order, out),
            _ => Err(crate::invalid("expert widen: the scales given differ from the stored form of the stack")),
        }
    }
    /// Make the device bytes of one matrix of a packed stack, its unit, from the stored bytes of its parts.
    /// The arguments are those of `widen`.
    pub fn fill_packed(&self, codes: &[u8], scales: &[u8], expert_scale: Option<f32>, divisor: Option<f32>, out: &mut [u8]) -> crate::Result<()> {
        let unit = self.unit().ok_or_else(|| crate::invalid("packed expert: the tensor is not a packed expert stack"))?;
        let factor = match (unit.codes, expert_scale, divisor) {
            (ExpertCodes::Nvfp4, Some(amax), Some(divisor)) => Some(loop_widen::expert_factor(amax, divisor)?),
            (ExpertCodes::Mxfp8, None, None) => None,
            _ => return Err(crate::invalid("packed expert: the scales given differ from the stored form of the stack")),
        };
        unit.fill(codes, scales, factor, out)
    }
    /// The stored parts of one matrix of the stack as the bytes that a product of form A2 reads (`loop_packed`).
    /// The arguments are those of `widen`.
    pub fn packed<'a>(&self, codes: &'a [u8], scales: &'a [u8], expert_scale: Option<f32>, divisor: Option<f32>) -> crate::Result<PackedMatrix<'a>> {
        let (LoadForm::WidenedExperts { codes: form, rows: order }, Some([_, rows, columns])) = (self.form, self.stack()) else {
            return Err(crate::invalid("packed expert: the tensor is not an expert stack"));
        };
        let size = |n: u64| usize::try_from(n).map_err(|_| crate::invalid("packed expert: a size exceeds the address space"));
        let form = match (form, expert_scale, divisor) {
            (ExpertCodes::Nvfp4, Some(amax), Some(divisor)) => PackedCodes::Nvfp4 { factor: loop_widen::expert_factor(amax, divisor)?, nibbles: NVFP4_NIBBLES },
            (ExpertCodes::Mxfp8, None, None) => PackedCodes::Mxfp8,
            _ => return Err(crate::invalid("packed expert: the scales given differ from the stored form of the stack")),
        };
        Ok(PackedMatrix { codes, scales, rows: size(rows)?, columns: size(columns)?, form, order })
    }
}
/// Byte sums of a load plan. Every value is a sum over `LoopLoadPlan::tensors`, or over the names that are not loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoadBytes {
    /// Tensors loaded as stored, and their device bytes.
    pub stored_tensors: u64, pub stored: u64,
    /// The head: device bytes of its used rows, and the stored bytes of the rows that are not loaded.
    pub head: u64, pub head_unused: u64,
    /// Widened expert stacks: their count, their device bytes and the stored bytes that they are made from.
    pub expert_stacks: u64, pub experts: u64, pub expert_sources: u64,
    /// Packed expert stacks: their count and their device bytes. They are made from the same stored bytes.
    pub packed_stacks: u64, pub packed: u64,
    /// All device bytes of the weights.
    pub device: u64,
    /// The bytes of the weight files that the load reads, each stored byte counted one time. A packed stack reads
    /// the stored bytes of its widened stack again.
    pub file: u64,
    /// Stored bytes of the image tower. Text serving does not load them.
    pub unused: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopLoadPlan {
    /// Device tensors by name.
    pub tensors: BTreeMap<String, LoadTensor>,
    pub bytes: LoadBytes,
}

fn part(kind: PartKind, name: &str, t: &BoundTensor, at: &(String, u64)) -> FilePart {
    FilePart { kind, name: name.into(), shard: at.0.clone(), dtype: t.dtype, shape: t.shape.clone(), offset: at.1, bytes: t.bytes }
}
/// The device forms of one expert stack: the widened stack and the packed stack. This is the one place that
/// decides them.
fn expert_tensor(name: &str, codes: &BoundTensor, proj: Proj, parts: Vec<FilePart>) -> Result<(LoadTensor, LoadTensor)> {
    let wrong = || PlanError::Unsupported { field: "expert storage", value: format!("{name}: the stored parts are not a 4-bit or an 8-bit expert stack") };
    let has = |kind| parts.iter().any(|p: &FilePart| p.kind == kind);
    let &[count, rows, stored] = codes.shape.as_slice() else { return Err(wrong()); };
    let (form, columns) = match codes.dtype {
        Dtype::U8 if has(PartKind::BlockScale) && has(PartKind::ExpertScale) && has(PartKind::ScaleDivisor) && parts.len() == 4 => (ExpertCodes::Nvfp4, mul(stored, 2)?),
        Dtype::F8E4M3 if has(PartKind::BlockScale) && parts.len() == 2 => (ExpertCodes::Mxfp8, stored),
        _ => return Err(wrong()),
    };
    let unmix = match proj { Proj::GateUp => Some(GATE_UP_ROWS), Proj::Down => None, _ => return Err(wrong()) };
    let widened = LoadTensor { name: name.into(), role: codes.role, place: codes.place, form: LoadForm::WidenedExperts { codes: form, rows: unmix },
        dtype: Dtype::Bf16, shape: vec![count, rows, columns], bytes: mul(mul(mul(count, rows)?, columns)?, Dtype::Bf16.bytes())?, parts };
    // The same matrices in their stored form. The shape is the shape of the weights; the bytes are whole units.
    let mut packed = LoadTensor { name: format!("{name}{PACKED_NAME}"), form: LoadForm::PackedExperts { codes: form, rows: unmix }, dtype: codes.dtype, bytes: 0, ..widened.clone() };
    packed.bytes = mul(count, packed.matrix_bytes()?)?;
    Ok((widened, packed))
}

impl LoopLoadPlan {
    /// `bound` is the checked index of the checkpoint. `shards` holds the header of each weight file by file name.
    pub fn new(d: &LoopTextDescription, bound: &BoundIndex, shards: &BTreeMap<String, Header>) -> Result<Self> {
        // File and absolute byte offset of every stored name.
        let mut places = BTreeMap::new();
        for (shard, header) in shards {
            for (name, t) in &header.tensors {
                if places.insert(name.as_str(), (shard.clone(), add(header.data_start, t.start)?)).is_some() {
                    return Err(PlanError::Invalid { field: "tensors", reason: "a stored name is in two weight files" });
                }
            }
        }
        let place_of = |name: &str| places.get(name).ok_or_else(|| PlanError::MissingTensor { name: name.into() });
        let mut tensors = BTreeMap::new();
        let mut bytes = LoadBytes::default();
        // The parts of each expert stack, by layer and projection.
        let mut stacks: BTreeMap<(u32, bool), Vec<(&String, &BoundTensor, PartKind)>> = BTreeMap::new();
        for (name, t) in &bound.tensors {
            match (t.state, t.role) {
                (TensorState::Unused, _) => bytes.unused = add(bytes.unused, t.bytes)?,
                (TensorState::NotRun, _) => return Err(PlanError::Invalid { field: "tensors", reason: "a draft tensor is in the main weight file" }),
                (TensorState::Text, StoredRole::Expert { proj, part: p }) => {
                    let kind = match p { ExpertPart::Codes => PartKind::Codes, ExpertPart::BlockScale => PartKind::BlockScale,
                        ExpertPart::ExpertScale => PartKind::ExpertScale, ExpertPart::ScaleDivisor => PartKind::ScaleDivisor };
                    let Place::Layer(layer) = t.place else { return Err(PlanError::Invalid { field: "expert storage", reason: "an expert tensor is not in a layer" }); };
                    stacks.entry((layer, proj == Proj::Down)).or_default().push((name, t, kind));
                }
                (TensorState::Text, StoredRole::Head) => {
                    let &[rows, hidden] = t.shape.as_slice() else { return Err(PlanError::TensorMismatch { name: name.clone(), expected: vec![d.vocab, d.hidden], actual: t.shape.clone() }); };
                    if d.head_rows == 0 || d.head_rows > rows { return Err(PlanError::Invalid { field: "unpadded_vocab_size", reason: "the head stores fewer rows than the tokens that it must score" }); }
                    let used = mul(mul(d.head_rows, hidden)?, t.dtype.bytes())?;
                    let mut cut = part(PartKind::Whole, name, t, place_of(name)?);
                    cut.shape = vec![d.head_rows, hidden]; cut.bytes = used;
                    bytes.head = add(bytes.head, device_extent(used)?)?; bytes.head_unused = add(bytes.head_unused, t.bytes - used)?;
                    tensors.insert(name.clone(), LoadTensor { name: name.clone(), role: t.role, place: t.place, form: LoadForm::LeadingRows,
                        dtype: t.dtype, shape: vec![d.head_rows, hidden], bytes: used, parts: vec![cut] });
                }
                (TensorState::Text, _) => {
                    bytes.stored_tensors += 1; bytes.stored = add(bytes.stored, device_extent(t.bytes)?)?;
                    tensors.insert(name.clone(), LoadTensor { name: name.clone(), role: t.role, place: t.place, form: LoadForm::Stored,
                        dtype: t.dtype, shape: t.shape.clone(), bytes: t.bytes, parts: vec![part(PartKind::Whole, name, t, place_of(name)?)] });
                }
            }
        }
        for stack in stacks.into_values() {
            let mut codes = stack.iter().filter(|(_, _, kind)| *kind == PartKind::Codes);
            let (Some(&(name, t, _)), None) = (codes.next(), codes.next()) else { return Err(PlanError::Invalid { field: "expert storage", reason: "an expert stack does not have exactly one code tensor" }); };
            let StoredRole::Expert { proj, .. } = t.role else { unreachable!() };
            let parts = stack.iter().map(|&(name, t, kind)| Ok(part(kind, name, t, place_of(name)?))).collect::<Result<Vec<_>>>()?;
            let (tensor, packed) = expert_tensor(name, t, proj, parts)?;
            bytes.expert_stacks += 1; bytes.experts = add(bytes.experts, tensor.device_bytes()?)?;
            bytes.expert_sources = add(bytes.expert_sources, tensor.file_bytes())?;
            bytes.packed_stacks += 1; bytes.packed = add(bytes.packed, packed.device_bytes()?)?;
            tensors.insert(name.clone(), tensor);
            if tensors.insert(packed.name.clone(), packed).is_some() { return Err(PlanError::Invalid { field: "tensors", reason: "a stored name is the name of a packed expert stack" }); }
        }
        bytes.device = add(add(add(bytes.stored, bytes.head)?, bytes.experts)?, bytes.packed)?;
        for t in tensors.values().filter(|t| !t.form.is_packed()) { bytes.file = add(bytes.file, t.file_bytes())?; }
        Ok(Self { tensors, bytes })
    }
    pub fn tensor(&self, name: &str) -> Result<&LoadTensor> {
        self.tensors.get(name).ok_or_else(|| PlanError::MissingTensor { name: name.into() })
    }
    /// The one tensor with this role at this place. A packed expert stack is not found here (`packed_experts`).
    pub fn by_role(&self, place: Place, role: StoredRole) -> Result<&LoadTensor> {
        let mut found = self.tensors.values().filter(|t| t.place == place && t.role == role && !t.form.is_packed());
        match (found.next(), found.next()) {
            (Some(t), None) => Ok(t),
            _ => Err(PlanError::Unsupported { field: "weight role", value: format!("{role:?} at {place:?}: the load plan does not hold exactly one tensor") }),
        }
    }
    /// The widened stack of one projection of one stored layer.
    pub fn experts(&self, layer: u32, proj: Proj) -> Result<&LoadTensor> {
        self.by_role(Place::Layer(layer), StoredRole::Expert { proj, part: ExpertPart::Codes })
    }
    /// The packed stack of one projection of one stored layer.
    pub fn packed_experts(&self, layer: u32, proj: Proj) -> Result<&LoadTensor> {
        let role = StoredRole::Expert { proj, part: ExpertPart::Codes };
        let mut found = self.tensors.values().filter(|t| t.place == Place::Layer(layer) && t.role == role && t.form.is_packed());
        match (found.next(), found.next()) {
            (Some(t), None) => Ok(t),
            _ => Err(PlanError::Unsupported { field: "weight role", value: format!("{role:?} at layer {layer}: the load plan does not hold exactly one packed stack") }),
        }
    }
}

/// SHA-256 of bytes that a load reads or makes, as lower-case hexadecimal text.
pub fn digest(bytes: &[u8]) -> String { engine_hash::digest(bytes) }

/// A checkpoint folder after the header checks, with its load plan. No payload was read.
#[derive(Debug)]
pub struct LoopCheckpoint {
    pub description: LoopTextDescription,
    pub bound: BoundIndex,
    /// The header of each weight file of the main model, by file name.
    pub shards: BTreeMap<String, Header>,
    pub load: LoopLoadPlan,
}
/// Read the config, the storage sidecar, the index and the headers of the checkpoint in `root`, bind every
/// stored name and make the load plan. Any other schema is refused.
pub fn read_folder(root: &Path) -> Result<LoopCheckpoint> {
    let read = |file: &'static str, limit: usize| safetensors::read_bounded(&root.join(file), limit).map_err(|e| read_error(file, e));
    let description = LoopTextDescription::from_config(&read("config.json", CONFIG_BYTES)?)?;
    let bound = inkling_index::inspect(root)?;
    let limits = Limits { header_bytes: HEADER_BYTES, tensors: MAX_TENSORS, rank: MAX_TENSOR_RANK, shards: MAX_SHARDS };
    const INDEX: &str = "model.safetensors.index.json";
    let index = Index::parse(&read(INDEX, HEADER_BYTES)?, limits).map_err(|e| read_error(INDEX, e))?;
    let shards = index.inspect_shards(root, limits).map_err(|e| read_error("safetensors headers", e))?;
    let load = LoopLoadPlan::new(&description, &bound, &shards)?;
    Ok(LoopCheckpoint { description, bound, shards, load })
}
