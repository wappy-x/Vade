//! Stored tensors of the whole-model host form: where the bytes come from,
//! which stored name each role has, and the bounded cache of widened experts.
//!
//! A weight file and a map held in memory give the same thing here: stored
//! bytes by stored name. The names come from the tensor index of the model
//! family (`plan/inkling_index.rs`), never from this file. Each tensor is
//! checked against the type and the shape that the index expects.
//!
//! Experts are not widened at load. One expert is widened when a token picks
//! it, through `capped_moe::Expert::from_stored`, and kept in a cache with a
//! byte limit. The cache changes no value: a widened expert is a fixed
//! function of its stored bytes.
use crate::{Result, invalid, capped_moe::Expert, expert_widen::{Stored, StoredMatrix}, flash::nvfp4::bf16_to_f32,
    plan::{description::LoopTextDescription, inkling_index::{Expected, ExpertPart, IndexInput, StoredRole}, storage_groups},
    safetensors::{self, Dtype, TensorFile}, shapes::{HEADER_BYTES, MAX_SHARDS, MAX_TENSORS, MAX_TENSOR_RANK, Place, Proj}};
use std::{collections::BTreeMap, path::Path, sync::{Arc, Mutex}};

/// Stored tensors by stored name.
pub trait Source: Send + Sync {
    /// The stored type and the shape of one tensor.
    fn tensor(&self, name: &str) -> Result<(Dtype, Vec<u64>)>;
    /// Copy stored bytes of one tensor, starting at `byte_offset` of its payload.
    fn read(&self, name: &str, byte_offset: u64, output: &mut [u8]) -> Result<()>;
}
impl Source for TensorFile {
    fn tensor(&self, name: &str) -> Result<(Dtype, Vec<u64>)> {
        let tensor = self.header().tensors.get(name).ok_or_else(|| invalid(format!("loop host: tensor {name} is absent from the weight file")))?;
        Ok((tensor.dtype, tensor.shape.clone()))
    }
    fn read(&self, name: &str, byte_offset: u64, output: &mut [u8]) -> Result<()> { self.read_range(name, byte_offset, output) }
}
/// The weight files of one model folder, as its shard index names them.
pub struct Folder { files: Vec<TensorFile>, file_of: BTreeMap<String, usize> }
impl Folder {
    /// Open every weight file that `model.safetensors.index.json` in `root` names. Only headers are read.
    pub fn open(root: &Path) -> Result<Self> {
        let limits = safetensors::Limits { header_bytes: HEADER_BYTES, tensors: MAX_TENSORS, rank: MAX_TENSOR_RANK, shards: MAX_SHARDS };
        let root = root.canonicalize()?;
        let index = safetensors::Index::open(&root.join("model.safetensors.index.json"), limits)?;
        let mut shards: Vec<&String> = index.weight_map.values().collect();
        shards.sort_unstable(); shards.dedup();
        let (mut files, mut file_of) = (Vec::with_capacity(shards.len()), BTreeMap::new());
        for shard in shards {
            let path = root.join(shard).canonicalize()?;
            if !path.starts_with(&root) { return Err(invalid("loop host: a weight file link leaves the model folder")); }
            let file = TensorFile::open(&path, limits)?;
            for name in file.header().tensors.keys() {
                if file_of.insert(name.clone(), files.len()).is_some() { return Err(invalid(format!("loop host: tensor {name} is in two weight files"))); }
            }
            files.push(file);
        }
        Ok(Self { files, file_of })
    }
    fn file(&self, name: &str) -> Result<&TensorFile> {
        self.file_of.get(name).map(|&at| &self.files[at]).ok_or_else(|| invalid(format!("loop host: tensor {name} is absent from the weight files")))
    }
}
impl Source for Folder {
    fn tensor(&self, name: &str) -> Result<(Dtype, Vec<u64>)> { Source::tensor(self.file(name)?, name) }
    fn read(&self, name: &str, byte_offset: u64, output: &mut [u8]) -> Result<()> { self.file(name)?.read_range(name, byte_offset, output) }
}
/// Read the description and the tensor index of the model folder `root` and open its weight files.
pub fn open_folder(root: &Path) -> Result<(LoopTextDescription, IndexInput, Folder)> {
    let (description, groups, _) = storage_groups::read_folder(root)?;
    let index = IndexInput::from_description(&description, &groups).map_err(|e| invalid(format!("loop host: tensor index: {e}")))?;
    Ok((description, index, Folder::open(root)?))
}
/// One stored tensor held in memory: little-endian payload bytes, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryTensor { pub dtype: Dtype, pub shape: Vec<u64>, pub bytes: Vec<u8> }
/// Stored tensors held in memory, by stored name. For made-up models in tests.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Memory { pub tensors: BTreeMap<String, MemoryTensor> }
impl Source for Memory {
    fn tensor(&self, name: &str) -> Result<(Dtype, Vec<u64>)> {
        let tensor = self.tensors.get(name).ok_or_else(|| invalid(format!("loop host: tensor {name} is absent from memory")))?;
        Ok((tensor.dtype, tensor.shape.clone()))
    }
    fn read(&self, name: &str, byte_offset: u64, output: &mut [u8]) -> Result<()> {
        let tensor = self.tensors.get(name).ok_or_else(|| invalid(format!("loop host: tensor {name} is absent from memory")))?;
        let start = usize::try_from(byte_offset).map_err(|_| invalid("loop host: read offset exceeds usize"))?;
        let bytes = start.checked_add(output.len()).and_then(|end| tensor.bytes.get(start..end))
            .ok_or_else(|| invalid(format!("loop host: read extends beyond tensor {name}")))?;
        output.copy_from_slice(bytes);
        Ok(())
    }
}

fn count(shape: &[u64]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &d| usize::try_from(d).ok().and_then(|d| n.checked_mul(d)))
        .filter(|n| n.checked_mul(std::mem::size_of::<f32>()).is_some_and(|bytes| bytes <= isize::MAX as usize))
        .ok_or_else(|| invalid("loop host: tensor size exceeds the address range"))
}

/// The expected stored names of one model, with the source that holds them.
pub(super) struct Names<'a> { pub expected: BTreeMap<String, Expected>, pub source: &'a dyn Source }
impl<'a> Names<'a> {
    pub fn new(index: &IndexInput, source: &'a dyn Source) -> Result<Self> {
        let expected = index.expected_main().map_err(|e| invalid(format!("loop host: tensor index: {e}")))?;
        Ok(Self { expected, source })
    }
    /// The one stored name with this role at this place, if the model stores it.
    pub fn find(&self, role: StoredRole, place: Place) -> Option<(&str, &Expected)> {
        self.expected.iter().find(|(_, e)| e.role == role && e.place == place).map(|(name, e)| (name.as_str(), e))
    }
    /// As [`Names::find`]. A role that the model must store is refused when absent.
    pub fn need(&self, role: StoredRole, place: Place) -> Result<(&str, &Expected)> {
        let (name, expected) = self.find(role, place).ok_or_else(|| invalid(format!("loop host: no stored tensor has the role {role:?} at {place:?}")))?;
        let (dtype, shape) = self.source.tensor(name)?;
        if dtype != expected.dtype || shape != expected.shape {
            return Err(invalid(format!("loop host: tensor {name} is {dtype:?} {shape:?}, expected {:?} {:?}", expected.dtype, expected.shape)));
        }
        Ok((name, expected))
    }
    /// The first `values` numbers of a BF16 or FP32 tensor, widened to FP32. Widening changes no value.
    pub fn first(&self, role: StoredRole, place: Place, values: usize) -> Result<Vec<f32>> {
        let (name, expected) = self.need(role, place)?;
        if values > count(&expected.shape)? { return Err(invalid(format!("loop host: tensor {name} holds fewer values than its role reads"))); }
        floats(self.source, name, expected.dtype, 0, values)
    }
    /// A whole BF16 or FP32 tensor, widened to FP32.
    pub fn all(&self, role: StoredRole, place: Place) -> Result<Vec<f32>> {
        let (_, expected) = self.need(role, place)?;
        self.first(role, place, count(&expected.shape)?)
    }
    /// The stored stack of one expert matrix kind of one layer.
    pub fn stack(&self, layer: Place, proj: Proj) -> Result<Stack> {
        let part = |part| StoredRole::Expert { proj, part };
        let (codes, expected) = self.need(part(ExpertPart::Codes), layer)?;
        let (scales, scale) = self.need(part(ExpertPart::BlockScale), layer)?;
        let [experts, rows, stored_columns] = expected.shape[..] else { return Err(invalid(format!("loop host: expert stack {codes} must have three axes"))); };
        let (experts, rows) = (count(&[experts])?, count(&[rows])?);
        if scale.shape.len() != 3 || scale.shape[..2] != expected.shape[..2] { return Err(invalid(format!("loop host: block scales of {codes} differ from its rows"))); }
        let (code_bytes, scale_bytes) = (count(&[stored_columns])?.checked_mul(rows), count(&[scale.shape[2]])?.checked_mul(rows));
        let (Some(code_bytes), Some(scale_bytes)) = (code_bytes, scale_bytes) else { return Err(invalid("loop host: expert stack size overflow")); };
        // The stored type of the codes tells the two formats apart, as in `expert_widen`.
        let (columns, second) = match expected.dtype {
            Dtype::U8 => {
                let amax = self.all(part(ExpertPart::ExpertScale), layer)?;
                let denominator = self.all(part(ExpertPart::ScaleDivisor), layer)?;
                if amax.len() != experts || denominator.len() != 1 { return Err(invalid(format!("loop host: second-level scales of {codes} differ from its expert count"))); }
                (count(&[stored_columns])?.checked_mul(2), Some((amax, denominator[0])))
            }
            Dtype::F8E4M3 => (Some(count(&[stored_columns])?), None),
            other => return Err(invalid(format!("loop host: expert stack {codes} has the unsupported type {other:?}"))),
        };
        let columns = columns.ok_or_else(|| invalid("loop host: expert stack size overflow"))?;
        Ok(Stack { codes: codes.into(), scales: scales.into(), experts, rows, columns, code_bytes, scale_bytes, second })
    }
}

/// Read `values` numbers of a BF16 or FP32 tensor from value `first`, as FP32.
pub(super) fn floats(source: &dyn Source, name: &str, dtype: Dtype, first: usize, values: usize) -> Result<Vec<f32>> {
    let width = match dtype { Dtype::Bf16 => 2, Dtype::F32 => 4, other => return Err(invalid(format!("loop host: tensor {name} has the type {other:?}, not BF16 or FP32"))) };
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(values.checked_mul(width).ok_or_else(|| invalid("loop host: tensor byte count overflow"))?).map_err(|_| invalid(format!("loop host: cannot reserve tensor {name}")))?;
    bytes.resize(values * width, 0u8);
    source.read(name, (first as u64).checked_mul(width as u64).ok_or_else(|| invalid("loop host: tensor offset overflow"))?, &mut bytes)?;
    let out: Vec<f32> = match dtype {
        Dtype::Bf16 => bytes.as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes(*b))).collect(),
        _ => bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
    };
    if out.iter().any(|v| !v.is_finite()) { return Err(invalid(format!("loop host: tensor {name} holds a value that is not finite"))); }
    Ok(out)
}

/// Where the stored bytes of one expert matrix kind of one layer are.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Stack {
    codes: String, scales: String, experts: usize,
    pub rows: usize, pub columns: usize,
    code_bytes: usize, scale_bytes: usize,
    /// `amax` of each expert and the one denominator of the layer. Packed 4-bit only.
    second: Option<(Vec<f32>, f32)>,
}
impl Stack {
    pub fn experts(&self) -> usize { self.experts }
    /// Read the stored bytes of one expert and hand its matrix to `use_matrix`.
    fn with<T>(&self, source: &dyn Source, expert: usize, use_matrix: impl FnOnce(&StoredMatrix<'_>) -> Result<T>) -> Result<T> {
        if expert >= self.experts { return Err(invalid("loop host: expert number is outside the stack")); }
        let (mut codes, mut scales) = (vec![0u8; self.code_bytes], vec![0u8; self.scale_bytes]);
        source.read(&self.codes, (expert * self.code_bytes) as u64, &mut codes)?;
        source.read(&self.scales, (expert * self.scale_bytes) as u64, &mut scales)?;
        let data = match &self.second {
            Some((amax, denominator)) => Stored::Packed4 { codes: &codes, scales: &scales, amax: amax[expert], denominator: *denominator },
            None => Stored::Float8 { values: &codes, scales: &scales },
        };
        use_matrix(&StoredMatrix { rows: self.rows, columns: self.columns, data })
    }
}

/// The two stored stacks of one layer: the gate-and-up matrix and the down matrix.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ExpertStacks { pub gate_up: Stack, pub down: Stack }
impl ExpertStacks {
    /// Widen one expert from its stored bytes. This calls the one seam, `Expert::from_stored`.
    pub fn widen(&self, source: &dyn Source, hidden: usize, width: usize, expert: usize) -> Result<Expert> {
        self.gate_up.with(source, expert, |gate_up| self.down.with(source, expert, |down| Expert::from_stored(hidden, width, gate_up, down)))
    }
}

/// What the expert cache did so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheCounts { pub hits: u64, pub widened: u64, pub held: usize, pub held_bytes: usize }
#[derive(Default)]
struct Held { experts: BTreeMap<(usize, usize), Entry>, tick: u64, bytes: usize, hits: u64, widened: u64 }
struct Entry { expert: Arc<Expert>, used: u64, bytes: usize }
/// Widened experts by layer and expert number, with a byte limit. The entry
/// that was used longest ago leaves first. A limit below one expert keeps none.
pub(super) struct ExpertCache { limit: usize, held: Mutex<Held> }
impl ExpertCache {
    pub fn new(limit: usize) -> Self { Self { limit, held: Mutex::new(Held::default()) } }
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Held>> { self.held.lock().map_err(|_| invalid("loop host: the expert cache lock is poisoned")) }
    pub fn counts(&self) -> Result<CacheCounts> {
        let held = self.lock()?;
        Ok(CacheCounts { hits: held.hits, widened: held.widened, held: held.experts.len(), held_bytes: held.bytes })
    }
    /// The widened expert, from the cache or from `widen`. `bytes` is the size of one widened expert.
    /// Two threads can widen the same expert at the same time. Both get the same numbers.
    pub fn get(&self, key: (usize, usize), bytes: usize, widen: impl FnOnce() -> Result<Expert>) -> Result<Arc<Expert>> {
        {
            let mut held = self.lock()?;
            held.tick += 1;
            let tick = held.tick;
            if let Some(entry) = held.experts.get_mut(&key) {
                entry.used = tick;
                let expert = entry.expert.clone();
                held.hits += 1;
                return Ok(expert);
            }
        }
        let expert = Arc::new(widen()?);
        let mut held = self.lock()?;
        held.tick += 1; held.widened += 1;
        let used = held.tick;
        match held.experts.insert(key, Entry { expert: expert.clone(), used, bytes }) {
            Some(old) => held.bytes = held.bytes - old.bytes + bytes,
            None => held.bytes += bytes,
        }
        while held.bytes > self.limit {
            let Some(oldest) = held.experts.iter().min_by_key(|(_, entry)| entry.used).map(|(key, _)| *key) else { break };
            if let Some(gone) = held.experts.remove(&oldest) { held.bytes -= gone.bytes; }
        }
        Ok(expert)
    }
}
