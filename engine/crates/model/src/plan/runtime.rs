//! Hybrid runtime metadata. Geometry/roles include draft layers; addresses belong to the loader.
use super::*;
use crate::safetensors::Dtype;
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
pub enum ProjectionStorage<'a> { Bf16, Fp8 { scales: &'a TensorEntry, block: [u64; 2] } }
#[derive(Debug, Clone, Copy)]
pub struct Projection<'a> {
    pub weight: &'a TensorEntry,
    pub rows: u64,
    pub cols: u64,
    pub storage: ProjectionStorage<'a>,
}
impl Projection<'_> {
    /// Original row-major file coordinates, before packing; empty ranges are legal.
    pub fn row_bytes(&self, rows: Range<u64>) -> Result<Range<u64>> {
        if rows.start > rows.end || rows.end > self.rows { return Err(invalid("projection rows", "outside weight")); }
        let stride = mul(self.cols, self.weight.dtype.bytes())?;
        Ok(mul(rows.start, stride)?..mul(rows.end, stride)?)
    }
    /// Flat BF16 multiplier index for an original weight coordinate; never invert/repack scales.
    pub fn scale_index(&self, row: u64, col: u64) -> Result<u64> {
        if row >= self.rows || col >= self.cols { return Err(invalid("scale coordinate", "outside weight")); }
        let ProjectionStorage::Fp8 { scales, block } = self.storage else { return Err(invalid("scale coordinate", "BF16 has no block scale")); };
        if block != [FP8_BLOCK; 2] || scales.shape.len() != 2 { return Err(invalid("scale layout", "requires K128/N128 grid")); }
        add(mul(row / block[0], scales.shape[1])?, col / block[1])
    }
}
impl ModelPlan {
    /// Contract only, not execution qualification. LayerShape is the capability record.
    /// FP8 uses per-token K128 groups and fixed ascending K blocks in every phase (scout Q3).
    pub fn hybrid_projections(&self) -> Result<Vec<Projection<'_>>> {
        if self.schema.family != config::Family::Hybrid || self.shape.residual != Residual::Single || self.shape.lexical.is_some() {
            return Err(invalid("hybrid capabilities", "requires single residual without lexical/indexed attention"));
        }
        if self.shape.tensors.is_empty() { return Err(invalid("hybrid inventory", "headers have not been reconciled")); }
        if self.shape.layers.iter().chain(&self.shape.draft).any(|l| matches!(l.mixer, Mixer::Attention(a) if a.index.is_some())
            || matches!(l.ffn, Ffn::Experts(e) if e.storage == ExpertStorage::Stacked && matches!(self.format, Format::Fp8 { .. }))) {
            return Err(invalid("hybrid layout", "indexed attention or stacked FP8 experts are unsupported"));
        }
        let inventory = self.shape.tensors.iter().map(|t| (t.name.as_str(), t)).collect::<BTreeMap<_, _>>();
        self.shape.tensors.iter().filter(|t| t.role.place != Place::Unused && t.role.store == Store::Weight && t.shape.len() == 2).map(|weight| {
            let (rows, cols) = (weight.shape[0], weight.shape[1]);
            if rows == 0 || cols == 0 { return Err(invalid("projection shape", "zero dimension")); }
            mul(mul(rows, cols)?, weight.dtype.bytes())?;
            let storage = match weight.dtype {
                Dtype::Bf16 => ProjectionStorage::Bf16,
                Dtype::F8E4M3 => {
                    let Format::Fp8 { block, .. } = self.format else { return Err(invalid("projection format", "FP8 without block format")); };
                    if block != [FP8_BLOCK; 2] || rows % FP8_BLOCK != 0 || cols % FP8_BLOCK != 0 { return Err(invalid("projection shape", "FP8 requires K128/N128 blocks")); }
                    let name = format!("{}_scale_inv", weight.name);
                    let scales = *inventory.get(name.as_str()).ok_or(PlanError::MissingTensor { name })?;
                    if scales.dtype != Dtype::Bf16 || scales.shape != [rows / block[0], cols / block[1]]
                        || scales.role != (Role { store: Store::BlockScale, ..weight.role }) { return Err(invalid("projection scales", "requires paired BF16 multiplier grid and role")); }
                    mul(mul(scales.shape[0], scales.shape[1])?, scales.dtype.bytes())?;
                    ProjectionStorage::Fp8 { scales, block }
                }
                _ => return Err(invalid("projection dtype", "requires BF16 or E4M3")),
            };
            Ok(Projection { weight, rows, cols, storage })
        }).collect()
    }
}
