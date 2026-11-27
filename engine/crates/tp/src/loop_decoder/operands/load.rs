//! Read the weight files, make each device tensor of the load plan, and give it to the owner of the weights.
//!
//! The work is cut into units. A tensor that is stored as it is, and the used rows of the head, are cut into
//! units of at most `UNIT_BYTES` file bytes. One matrix of an expert stack is one unit: its stored parts are
//! read and widened by the load plan (`LoadTensor::widen`), or packed by it for a packed stack
//! (`LoadTensor::fill_packed`). Worker threads make the units; the calling thread
//! alone talks to the owner of the weights. Every unit has the SHA-256 of the bytes that were written, so a
//! later read of the owner can be compared with what the files gave.
use super::{Operands, Resident, Stack};
use crate::Result;
use engine_model::{flash::nvfp4::bf16_to_f32, plan::{inkling_index::{GatePart, MessageGate, StoredRole, Stream}, loop_load::{DEVICE_WORD, LoadForm, LoadTensor, LoopCheckpoint, LoopLoadPlan, PartKind, digest}}, safetensors::Dtype, shapes::Place};
use std::{collections::BTreeMap, fs::File, ops::Range, os::unix::fs::FileExt, path::Path,
    sync::{Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}, mpsc::sync_channel}, time::Instant};

/// The owner of the weights: a device, or host bytes in a test.
pub trait Store {
    type Value;
    /// Storage of `words` 32-bit words.
    fn create(&mut self, words: usize) -> Result<Self::Value>;
    /// Write whole words. `bytes` holds little-endian words and starts at word `word` of the value.
    fn write(&mut self, value: &mut Self::Value, word: usize, bytes: &[u8]) -> Result<()>;
    /// Optionally retain lossless panels for a BF16 matrix loaded by this store.
    fn register_bf16(&mut self, _value: &Self::Value, _rows: usize, _cols: usize, _bytes: &[u8]) -> Result<()> { Ok(()) }
    /// Read whole words back into `out`, from word `word` of the value.
    fn read(&mut self, value: &Self::Value, word: usize, out: &mut [u8]) -> Result<()>;
    /// True where `view` gives a value that is a range of words of another value, the same storage.
    fn views(&self) -> bool { false }
    /// True when 4-bit expert units whose shape admits MMA-lane panels should be packed in-place during load.
    fn nvfp4_panels(&self) -> bool { false }
    /// The words `words` of `value` as a value of its own, the same storage. Only called where `views` is true.
    fn view(&mut self, _value: &Self::Value, _words: Range<usize>) -> Result<Self::Value> { Err("loop load: this owner of the weights keeps no views".into()) }
}
/// The attention matrices of a layer that one value holds one after another, in this order: the four that read the
/// query input, then the key and the projection of its token-made taps, then the value and its projection. With one
/// input for all three, one product reads the six of a layer that makes no taps from the token; on separate inputs
/// the key and its tap projection are one product, and the value and its own.
pub const STACKED: [StoredRole; 8] = [StoredRole::Query, StoredRole::RelativeProjection, StoredRole::Sink, StoredRole::OutputGate,
    StoredRole::Key, StoredRole::ConvProjection(Stream::Key), StoredRole::Value, StoredRole::ConvProjection(Stream::Value)];
/// The members of `STACKED` that a layer may lack: the projections of the token-made taps.
const OPTIONAL: [bool; 8] = [false, false, false, false, false, true, false, true];
/// The forget and the gain matrix of each message gate read the same inner vector: one stack each, in this order, so
/// one product gives the pair that arm `GateBlend` reads.
const GATES: [MessageGate; 4] = [MessageGate::Query, MessageGate::Key, MessageGate::Value, MessageGate::Expert];
/// The groups of a layer that one value holds one after another: the roles in order and whether a layer may lack one.
fn groups() -> Vec<Vec<(StoredRole, bool)>> {
    let mut out = vec![STACKED.iter().copied().zip(OPTIONAL).collect::<Vec<_>>()];
    out.extend(GATES.iter().map(|&gate| vec![(StoredRole::Gate { gate, part: GatePart::Forget }, false), (StoredRole::Gate { gate, part: GatePart::Gain }, false)]));
    out
}
/// The stacks of the plan: for each layer and each group of `groups` whose tensors (the optional ones where it has
/// them) are BF16 matrices as stored, with the same columns and no padding word, their names and words in one value,
/// and the words of the value. The bytes of each tensor do not change, so its units and its hashes are those of a
/// value of its own.
pub(crate) fn stacks(plan: &LoopLoadPlan) -> Result<Vec<(Vec<(String, Range<usize>)>, usize)>> {
    let mut found_all: BTreeMap<(u32, usize), Vec<Option<&LoadTensor>>> = BTreeMap::new();
    let groups = groups();
    for t in plan.tensors.values() {
        let Place::Layer(layer) = t.place else { continue };
        for (g, group) in groups.iter().enumerate() {
            if let Some(i) = group.iter().position(|(role, _)| *role == t.role) { found_all.entry((layer, g)).or_insert_with(|| vec![None; group.len()])[i] = Some(t); }
        }
    }
    let mut out = Vec::new();
    for (&(_, g), found) in &found_all {
        if found.iter().zip(&groups[g]).any(|(t, (_, optional))| t.is_none() && !optional) { continue; }
        let all = found.iter().flatten().copied().collect::<Vec<_>>();
        let columns = all[0].shape.get(1).copied();
        let fits = |t: &LoadTensor| t.form == LoadForm::Stored && t.dtype == Dtype::Bf16 && t.shape.len() == 2 && t.shape.get(1).copied() == columns
            && t.shape[0].checked_mul(t.shape[1]).and_then(|n| n.checked_mul(2)) == Some(t.bytes) && t.bytes.is_multiple_of(DEVICE_WORD);
        if !all.iter().all(|t| fits(t)) { continue; }
        let (mut start, mut members) = (0usize, Vec::with_capacity(all.len()));
        for t in all {
            let end = start.checked_add(size(t.bytes / DEVICE_WORD)?).ok_or("loop load: a stack exceeds the address space")?;
            members.push((t.name.clone(), start..end)); start = end;
        }
        out.push((members, start));
    }
    Ok(out)
}
/// Largest unit of a tensor that is loaded as stored: whole words, and small against the memory of a host.
pub const UNIT_BYTES: usize = 16 << 20;

/// One written unit and the hash of its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    /// Name of the device tensor.
    pub name: String,
    /// The matrix of an expert stack that this unit is. None for a unit of a tensor that is loaded as stored.
    pub matrix: Option<u64>,
    /// First word of the unit in its value, and its bytes (whole words).
    pub word: usize, pub bytes: usize,
    /// SHA-256 of the written bytes, as lower-case hexadecimal text.
    pub sha256: String,
}
/// Times and byte counts of one load.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LoadReport { pub tensors: usize, pub units: usize, pub device_bytes: u64, pub file_bytes: u64, pub allocate_seconds: f64, pub fill_seconds: f64 }
#[derive(Debug)]
pub struct Loaded<V> { pub operands: Operands<V>, pub units: Vec<Unit>, pub report: LoadReport }
impl<V> Loaded<V> {
    /// One hash for the whole load: the SHA-256 of the unit lines (name, matrix, word, bytes, hash) in name and word order.
    pub fn fingerprint(&self) -> String { digest(unit_lines(&self.units).as_bytes()) }
}
/// One line a unit: tensor name, matrix number or `-`, first word, bytes, SHA-256. Tab-separated.
pub fn unit_lines(units: &[Unit]) -> String {
    units.iter().map(|u| format!("{}\t{}\t{}\t{}\t{}\n", u.name, u.matrix.map_or("-".into(), |m| m.to_string()), u.word, u.bytes, u.sha256)).collect()
}

/// What a worker must make: unit `matrix` of an expert stack, or the bytes `offset..offset + bytes` of a stored tensor.
struct Job<'a> { tensor: &'a LoadTensor, matrix: Option<u64>, offset: u64, bytes: usize, word: usize }
fn plan_error(e: impl std::fmt::Display) -> String { e.to_string() }
fn size(n: u64) -> Result<usize> { usize::try_from(n).map_err(|_| "loop load: a size exceeds the address space".into()) }
fn jobs(checkpoint: &LoopCheckpoint) -> Result<Vec<Job<'_>>> {
    let word = DEVICE_WORD as usize;
    let mut jobs = Vec::new();
    for tensor in checkpoint.load.tensors.values() {
        match tensor.form {
            LoadForm::Stored | LoadForm::LeadingRows => {
                let total = size(tensor.bytes)?;
                let mut offset = 0;
                while offset < total {
                    let bytes = (total - offset).min(UNIT_BYTES);
                    jobs.push(Job { tensor, matrix: None, offset: offset as u64, bytes, word: offset / word });
                    offset += bytes;
                }
            }
            LoadForm::WidenedExperts { .. } | LoadForm::PackedExperts { .. } => {
                let [count, ..] = tensor.stack().ok_or("loop load: an expert tensor is not a stack")?;
                let each = size(tensor.matrix_bytes().map_err(plan_error)?)?;
                if !each.is_multiple_of(word) { return Err(format!("loop load: one matrix of {} does not fill whole device words", tensor.name)); }
                for matrix in 0..count { jobs.push(Job { tensor, matrix: Some(matrix), offset: 0, bytes: each, word: size(matrix)? * (each / word) }); }
            }
        }
    }
    Ok(jobs)
}
fn read_at(files: &BTreeMap<String, File>, shard: &str, offset: u64, bytes: usize) -> Result<Vec<u8>> {
    let file = files.get(shard).ok_or_else(|| format!("loop load: weight file {shard} is not open"))?;
    let mut out = vec![0u8; bytes];
    file.read_exact_at(&mut out, offset).map_err(|e| format!("loop load: read of {bytes} bytes at {offset} of {shard}: {e}"))?;
    Ok(out)
}
/// The bytes of one unit, in whole words.
fn make(job: &Job<'_>, files: &BTreeMap<String, File>, pack_nvfp4: bool) -> Result<Vec<u8>> {
    let t = job.tensor;
    let Some(matrix) = job.matrix else {
        let part = t.part(PartKind::Whole).ok_or_else(|| format!("loop load: {} has no file range", t.name))?;
        let mut bytes = read_at(files, &part.shard, part.offset + job.offset, job.bytes)?;
        // The last unit of a tensor ends on a whole word. The fill bytes are zero.
        bytes.resize(job.bytes.div_ceil(DEVICE_WORD as usize) * DEVICE_WORD as usize, 0);
        return Ok(bytes);
    };
    let read = |kind| -> Result<Option<Vec<u8>>> {
        let Some(part) = t.part(kind) else { return Ok(None); };
        let (offset, bytes) = t.matrix_part(kind, matrix).map_err(plan_error)?;
        Ok(Some(read_at(files, &part.shard, offset, size(bytes)?)?))
    };
    let need = |bytes: Option<Vec<u8>>| bytes.ok_or_else(|| format!("loop load: expert stack {} has no codes or no block scales", t.name));
    let float = |bytes: Option<Vec<u8>>| -> Result<Option<f32>> {
        bytes.map(|b| Ok(f32::from_le_bytes(b.as_slice().try_into().map_err(|_| format!("loop load: a scale of {} is not one 32-bit value", t.name))?))).transpose()
    };
    let (codes, scales) = (need(read(PartKind::Codes)?)?, need(read(PartKind::BlockScale)?)?);
    let mut out = vec![0u8; job.bytes];
    let (expert_scale, divisor) = (float(read(PartKind::ExpertScale)?)?, float(read(PartKind::ScaleDivisor)?)?);
    if t.form.is_packed() {
        t.fill_packed(&codes, &scales, expert_scale, divisor, &mut out)
            .map_err(|e| format!("loop load: {} matrix {matrix}: {e}", t.name))?;
        if pack_nvfp4
            && let Some(unit) = t.unit()
            && unit.codes == engine_model::plan::loop_load::ExpertCodes::Nvfp4
            && unit.rows.is_multiple_of(64)
            && unit.columns.is_multiple_of(64)
        {
            out = spark_gpu_probe::decode_panels::pack_expert_tagged(&out, unit.rows, unit.columns);
        }
    } else {
        t.widen(&codes, &scales, expert_scale, divisor, &mut out)
            .map_err(|e| format!("loop load: {} matrix {matrix}: {e}", t.name))?;
    }
    Ok(out)
}

/// The open weight files of a checkpoint folder, by file name.
pub struct Files(BTreeMap<String, File>);
impl Files {
    pub fn open(checkpoint: &LoopCheckpoint, root: &Path) -> Result<Self> {
        let root = root.canonicalize().map_err(|e| format!("loop load: checkpoint folder: {e}"))?;
        let mut files = BTreeMap::new();
        for shard in checkpoint.shards.keys() {
            let path = root.join(shard).canonicalize().map_err(|e| format!("loop load: {shard}: {e}"))?;
            if !path.starts_with(&root) { return Err(format!("loop load: {shard}: the file is outside the checkpoint folder")); }
            files.insert(shard.clone(), File::open(&path).map_err(|e| format!("loop load: {shard}: {e}"))?);
        }
        Ok(Self(files))
    }
    #[cfg(target_os = "linux")]
    pub(crate) fn append_stored(&self, tensor: &LoadTensor, out: &mut Vec<u8>) -> Result<()> {
        let part = tensor.part(PartKind::Whole).ok_or("BF16 panels: matrix has no stored file range")?;
        let file = self.0.get(&part.shard).ok_or("BF16 panels: matrix file is not open")?;
        let start = out.len();
        let end = start.checked_add(size(tensor.bytes)?).ok_or("BF16 panels: matrix size overflow")?;
        out.resize(end, 0);
        file.read_exact_at(&mut out[start..], part.offset).map_err(|e| e.to_string())
    }
    /// The values of a tensor that the plan keeps as stored, as 32-bit words in stored order. A BF16 value is
    /// widened; that changes no value.
    pub fn words(&self, tensor: &LoadTensor) -> Result<Vec<f32>> {
        let part = tensor.part(PartKind::Whole).ok_or_else(|| format!("loop load: {} has no file range", tensor.name))?;
        let bytes = read_at(&self.0, &part.shard, part.offset, size(tensor.bytes)?)?;
        match tensor.dtype {
            Dtype::Bf16 => Ok(bytes.as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes(*b))).collect()),
            Dtype::F32 => Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()),
            other => Err(format!("loop load: {} is {other:?}: only 16-bit and 32-bit floats are read as host words", tensor.name)),
        }
    }
}

/// Load every tensor of the plan into `store`. `root` is the checkpoint folder; `workers` threads read and widen.
pub fn load<S: Store>(checkpoint: &LoopCheckpoint, root: &Path, store: &mut S, workers: usize) -> Result<Loaded<S::Value>> {
    let files = Files::open(checkpoint, root)?.0;
    let started = Instant::now();
    let mut device_bytes = 0u64;
    // A tensor of a stack is a view of the value of its stack, made with its first member.
    let stacks = if store.views() { stacks(&checkpoint.load)? } else { Vec::new() };
    let member: BTreeMap<&str, (usize, Range<usize>)> = stacks.iter().enumerate().flat_map(|(i, (members, _))| members.iter().map(move |(name, words)| (name.as_str(), (i, words.clone())))).collect();
    let mut made: Vec<Option<S::Value>> = stacks.iter().map(|_| None).collect();
    let mut operands = Operands::new(&checkpoint.load, |tensor| {
        let bytes = tensor.device_bytes().map_err(plan_error)?;
        device_bytes += bytes;
        if let Some((i, words)) = member.get(tensor.name.as_str()) {
            if made[*i].is_none() { made[*i] = Some(store.create(stacks[*i].1)?); }
            return store.view(made[*i].as_ref().ok_or("loop load: a stack has no value")?, words.clone());
        }
        store.create(size(bytes / DEVICE_WORD)?)
    })?;
    operands.stacks = stacks.into_iter().zip(made).filter_map(|((members, _), value)| value.map(|value| Stack { members, value })).collect();
    let allocate_seconds = started.elapsed().as_secs_f64();
    let jobs = jobs(checkpoint)?;
    let filled = Instant::now();
    let (next, stop) = (AtomicUsize::new(0), AtomicBool::new(false));
    let mut units: Vec<Option<Unit>> = jobs.iter().map(|_| None).collect();
    let mut values: BTreeMap<String, &mut Resident<S::Value>> = operands.all_mut().map(|r| (r.tensor.name.clone(), r)).collect();
    let pack_nvfp4 = store.nvfp4_panels();
    std::thread::scope(|scope| -> Result<()> {
        let (sender, receiver) = sync_channel::<Result<(usize, Vec<u8>, String)>>(workers.max(1));
        for _ in 0..workers.max(1) {
            let (sender, jobs, files, next, stop) = (sender.clone(), &jobs, &files, &next, &stop);
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= jobs.len() || stop.load(Ordering::Relaxed) { break; }
                let made = make(&jobs[index], files, pack_nvfp4).map(|bytes| { let hash = digest(&bytes); (index, bytes, hash) });
                let failed = made.is_err();
                if sender.send(made).is_err() || failed { break; }
            });
        }
        drop(sender);
        let mut result = Ok(());
        for made in receiver {
            if result.is_err() { continue; }
            result = made.and_then(|(index, bytes, sha256)| {
                let job = &jobs[index];
                let resident = values.get_mut(job.tensor.name.as_str()).ok_or("loop load: a unit has no value")?;
                store.write(&mut resident.value, job.word, &bytes)?;
                units[index] = Some(Unit { name: job.tensor.name.clone(), matrix: job.matrix, word: job.word, bytes: bytes.len(), sha256 });
                Ok(())
            });
            if result.is_err() { stop.store(true, Ordering::Relaxed); }
        }
        result
    })?;
    drop(values);
    let units = units.into_iter().collect::<Option<Vec<_>>>().ok_or("loop load: a unit was not written")?;
    let report = LoadReport { tensors: operands.len(), units: units.len(), device_bytes, file_bytes: checkpoint.load.bytes.file,
        allocate_seconds, fill_seconds: filled.elapsed().as_secs_f64() };
    Ok(Loaded { operands, units, report })
}

/// Result of a read-back check.
#[derive(Debug, Clone, PartialEq)]
pub struct Verified { pub units: usize, pub bytes: u64, pub seconds: f64,
    /// Units whose bytes in the store differ from the bytes that were written.
    pub different: Vec<String> }
/// Read every unit back from `store` and compare its SHA-256 with the hash of the written bytes.
pub fn verify<S: Store>(loaded: &Loaded<S::Value>, store: &mut S, workers: usize) -> Result<Verified> {
    let started = Instant::now();
    let different = Mutex::new(Vec::new());
    let mut bytes = 0u64;
    let (sender, receiver) = sync_channel::<(usize, Vec<u8>)>(workers.max(1));
    let receiver = Mutex::new(receiver);
    std::thread::scope(|scope| -> Result<()> {
        for _ in 0..workers.max(1) {
            let (receiver, units, different) = (&receiver, &loaded.units, &different);
            scope.spawn(move || loop {
                let next = receiver.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).recv();
                let Ok((index, read)) = next else { break; };
                let unit = &units[index];
                if digest(&read) != unit.sha256 {
                    different.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(format!("{} matrix {} word {}", unit.name, unit.matrix.map_or("-".into(), |m| m.to_string()), unit.word));
                }
            });
        }
        let result = loaded.units.iter().enumerate().try_for_each(|(index, unit)| {
            let resident = loaded.operands.named(&unit.name)?;
            let mut out = vec![0u8; unit.bytes];
            store.read(&resident.value, unit.word, &mut out)?;
            bytes += unit.bytes as u64;
            sender.send((index, out)).map_err(|_| "loop load: the hash workers stopped".to_owned())
        });
        drop(sender);
        result
    })?;
    let mut different = different.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner());
    different.sort();
    Ok(Verified { units: loaded.units.len(), bytes, seconds: started.elapsed().as_secs_f64(), different })
}
