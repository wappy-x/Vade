//! The device as the owner of the weights: one device vector for each tensor of the load plan.
//! A vector holds the bytes of its tensor in whole 32-bit words. A unit of a load is written through a
//! view of its words, so no host copy of a whole tensor is kept and the pinned staging stays small.
use super::load::Store;
use crate::Result;
use spark_gpu_probe::driver::{Backend, DeviceVector};

pub struct DeviceStore<'b, 'd> { backend: &'b Backend<'d>, panel_room: usize }
impl<'b, 'd> DeviceStore<'b, 'd> {
    pub fn set_panel_room(&mut self, bytes: usize) { self.panel_room = bytes; }
    pub fn new(backend: &'b Backend<'d>) -> Self { Self { backend, panel_room: 0 } }
}
impl<'d> Store for DeviceStore<'_, 'd> {
    type Value = DeviceVector<'d>;
    fn create(&mut self, words: usize) -> Result<Self::Value> { self.backend.vector(words)?.into_resident_shared() }
    fn write(&mut self, value: &mut Self::Value, word: usize, bytes: &[u8]) -> Result<()> {
        if !bytes.len().is_multiple_of(4) { return Err("device store: a write must hold whole words".into()); }
        let end = word.checked_add(bytes.len() / 4).ok_or("device store: write range overflow")?;
        let mut part = value.resident_range(word..end)?;
        self.backend.write_vector_bytes(&mut part, bytes)
    }
    fn register_bf16(&mut self, value: &Self::Value, rows: usize, cols: usize, bytes: &[u8]) -> Result<()> {
        let bytes_needed = (cols >= 256).then(|| storage_words(rows, cols).and_then(|words| words.checked_mul(4))).flatten();
        if let Some(size) = bytes_needed.filter(|&size| size <= self.panel_room) {
            let allocated = self.backend.register_bf16_panels(value, rows, cols, bytes)?;
            if allocated > size { return Err("BF16 panels: draft bytes exceed admission".into()); }
            self.panel_room -= allocated;
        }
        Ok(())
    }
    fn views(&self) -> bool { true }
    fn nvfp4_panels(&self) -> bool { true }
    fn view(&mut self, value: &Self::Value, words: std::ops::Range<usize>) -> Result<Self::Value> { value.resident_range(words) }
    fn read(&mut self, value: &Self::Value, word: usize, out: &mut [u8]) -> Result<()> {
        if !out.len().is_multiple_of(4) { return Err("device store: a read must hold whole words".into()); }
        let end = word.checked_add(out.len() / 4).ok_or("device store: read range overflow")?;
        let words = self.backend.read_vector_values(&value.resident_range(word..end)?)?;
        // A word keeps its bit pattern: the bytes of a weight are not numbers to this store.
        for (out, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) { *out = word.to_bits().to_le_bytes(); }
        Ok(())
    }
}

use engine_model::{plan::{inkling_index::{StoredRole, Stream}, loop_load::{LoadBytes, LoadForm, LoadTensor, LoopLoadPlan}}, safetensors::Dtype};
use spark_gpu_probe::bf16_panels::storage_words;
use std::{collections::BTreeSet, ops::Range};

struct PanelSpec {
    names: Vec<String>,
    rows: usize,
    cols: usize,
    /// When this panel covers a sub-view inside a larger stack (e.g. Key or Value starting past a non-16-aligned Sink),
    /// the parent stack member names and the word range within that stack.
    parent_stack: Option<(Vec<String>, Range<usize>)>,
}

/// The main model's panel allocations, counted before service admission.
pub struct PanelPlan {
    specs: Vec<PanelSpec>,
    bytes: u64,
}

fn bf16_shape(t: &LoadTensor) -> Option<(usize, usize)> {
    let &[rows, cols] = t.shape.as_slice() else { return None; };
    let (rows, cols) = (usize::try_from(rows).ok()?, usize::try_from(cols).ok()?);
    (t.dtype == Dtype::Bf16 && matches!(t.form, LoadForm::Stored | LoadForm::LeadingRows)
        && rows.checked_mul(cols)?.checked_mul(2)? as u64 == t.bytes
        && t.role != StoredRole::Embedding).then_some((rows, cols))
}

impl PanelPlan {
    pub fn new(load: &LoopLoadPlan) -> Result<Self> {
        let mut specs = Vec::new();
        let mut stacked = BTreeSet::new();
        for (members, _) in super::load::stacks(load)? {
            let cols = bf16_shape(&load.tensors[&members[0].0]).ok_or("BF16 panels: stack shape differs")?.1;
            let mut rows = 0usize;
            let mut names = Vec::with_capacity(members.len());
            let mut entries = Vec::with_capacity(members.len());
            for (name, range) in members {
                let r = range.len() * 2 / cols;
                let row_start = rows;
                rows = rows.checked_add(r).ok_or("BF16 panels: stack extent overflow")?;
                stacked.insert(name.clone());
                names.push(name.clone());
                entries.push((name, row_start, r, range));
            }
            // Keep small latency-bound GateBlend (`cols == 64`) raw `cp.async` (`4,352 B` SMEM, 12+ blk/SM).
            if cols >= 256 && storage_words(rows, cols).is_some() {
                specs.push(PanelSpec { names: names.clone(), rows, cols, parent_stack: None });
            }
            // Register 16-row-aligned sub-view panels for Key(+Tap) and Value(+Tap) when a preceding Sink
            // shifts their row offset off a 16-row band boundary, so separate-input passes still use panels.
            if cols >= 256 {
                for (i, (name, row_start, r, range)) in entries.iter().enumerate() {
                    let stream = match load.tensors[name].role {
                        StoredRole::Key => Stream::Key,
                        StoredRole::Value => Stream::Value,
                        _ => continue,
                    };
                    if row_start.is_multiple_of(16) { continue; }
                    let (mut sub_names, mut sub_rows, mut end_word) = (vec![name.clone()], *r, range.end);
                    if let Some((next_name, _, next_r, next_range)) = entries.get(i + 1)
                        && load.tensors[next_name].role == StoredRole::ConvProjection(stream)
                    {
                        sub_names.push(next_name.clone());
                        sub_rows += *next_r;
                        end_word = next_range.end;
                    }
                    if storage_words(sub_rows, cols).is_some() {
                        specs.push(PanelSpec {
                            names: sub_names, rows: sub_rows, cols,
                            parent_stack: Some((names.clone(), range.start..end_word)),
                        });
                    }
                }
            }
        }
        for tensor in load.tensors.values().filter(|t| !stacked.contains(&t.name)) {
            if let Some((rows, cols)) = bf16_shape(tensor).filter(|&(rows, cols)| cols >= 256 && storage_words(rows, cols).is_some()) {
                specs.push(PanelSpec { names: vec![tensor.name.clone()], rows, cols, parent_stack: None });
            }
        }
        let bytes = specs.iter().try_fold(0u64, |total, spec| {
            let words = storage_words(spec.rows, spec.cols)?;
            total.checked_add(u64::try_from(words).ok()?.checked_mul(4)?)
        }).ok_or("BF16 panels: device byte count overflow")?;
        Ok(Self { specs, bytes })
    }

    pub fn bytes(&self) -> u64 { self.bytes }

    pub fn charge(&self, mut load: LoadBytes) -> Result<LoadBytes> {
        load.device = load.device.checked_add(self.bytes).ok_or("BF16 panels: weight byte count overflow")?;
        Ok(load)
    }

    pub fn upload<'d>(&self, backend: &Backend<'d>, files: &super::load::Files, operands: &super::Operands<DeviceVector<'d>>) -> Result<()> {
        let before = backend.allocated_bytes();
        for spec in &self.specs {
            let mut bytes = Vec::new();
            for name in &spec.names {
                files.append_stored(&operands.named(name)?.tensor, &mut bytes)?;
            }
            if let Some((parent, range)) = &spec.parent_stack {
                let stack = operands.stacks().iter().find(|stack| stack.members.iter().map(|m| &m.0).eq(parent))
                    .ok_or("BF16 panels: resident stack missing")?;
                let view = stack.value.resident_range(range.clone())?;
                backend.register_bf16_panels(&view, spec.rows, spec.cols, &bytes)?;
            } else {
                let weight = if spec.names.len() == 1 {
                    &operands.named(&spec.names[0])?.value
                } else {
                    &operands.stacks().iter().find(|stack| stack.members.iter().map(|m| &m.0).eq(&spec.names))
                        .ok_or("BF16 panels: resident stack missing")?.value
                };
                backend.register_bf16_panels(weight, spec.rows, spec.cols, &bytes)?;
            }
        }
        if (backend.allocated_bytes() - before) as u64 > self.bytes {
            return Err("BF16 panels: device bytes exceed admission".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panels_charge_exactly_their_device_bytes() {
        let bytes = (storage_words(17, 128).unwrap() * 4) as u64;
        let plan = PanelPlan { specs: Vec::new(), bytes };
        let raw = LoadBytes { device: 17 * 128 * 2, head: 128, ..LoadBytes::default() };
        let charged = plan.charge(raw).unwrap();
        assert_eq!(charged.device - raw.device, bytes);
        assert_eq!(charged.head, raw.head);
        assert!(plan.charge(LoadBytes { device: u64::MAX, ..raw }).is_err());
    }
}
