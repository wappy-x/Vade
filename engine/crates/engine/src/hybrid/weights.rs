//! Physical rank-local text storage. Arithmetic/forward admission remains in hybrid::plan.
use engine_model::{plan::ModelPlan, safetensors::{Dtype, Header, Index, Limits, TensorFile},
    shapes::{Role, Store, TensorEntry, HEADER_BYTES, MAX_TENSORS, MAX_TENSOR_RANK, MAX_SHARDS}, split::{Owners, Selection, SplitPlan}};
use std::{collections::BTreeMap, path::Path};
type Result<T> = std::result::Result<T, String>;

pub struct TensorView<'p> {
    pub tensor: &'p TensorEntry, pub offset: usize, pub bytes: usize,
    pub file_offset: u64, pub rows: usize, pub cols: usize,
    pub selection: Selection, pub spans: Vec<ByteSpan>,
}
/// File offsets and strides remain in original coordinates; local offsets concatenate row groups.
#[derive(Debug, Clone)]
pub struct ByteSpan { pub file_offset: u64, pub local_offset: usize, pub bytes: usize, pub row_bytes: usize, pub row_stride: usize }
fn selected(t: &TensorEntry, selection: Selection, base: u64) -> Result<(Selection, Vec<ByteSpan>, usize, usize)> {
    let elements = t.shape.iter().try_fold(1u64, |n, d| n.checked_mul(*d)).ok_or("weight shape overflow")?;
    let stride = t.shape.iter().skip(1).try_fold(t.dtype.bytes(), |n, d| n.checked_mul(*d)).ok_or("row stride overflow")?;
    let mut spans = Vec::new(); let mut local = 0usize;
    let mut append = |start: u64, bytes: u64, row_bytes: u64, row_stride: u64| -> Result<()> {
        if bytes == 0 { return Ok(()); }
        spans.push(ByteSpan { file_offset: base.checked_add(start).ok_or("selected file offset overflow")?,
            local_offset: local, bytes: size(bytes)?, row_bytes: size(row_bytes)?, row_stride: size(row_stride)? });
        local = local.checked_add(size(bytes)?).ok_or("selected payload overflow")?; Ok(())
    };
    let (selection, rows, cols) = match selection {
        Selection::All => { append(0, elements.checked_mul(t.dtype.bytes()).ok_or("weight bytes overflow")?, 0, 0)?;
            (Selection::All, if t.shape.len() == 2 { t.shape[0] } else { 1 }, if t.shape.len() == 2 { t.shape[1] } else { elements }) }
        Selection::Rows(ranges) => {
            let mut merged: Vec<std::ops::Range<u64>> = Vec::new(); let mut count = 0u64;
            for r in ranges {
                if r.start > r.end || t.shape.first().is_none_or(|n| r.end > *n) || merged.last().is_some_and(|p| r.start < p.end) {
                    return Err("invalid original row selection".into());
                }
                if r.is_empty() { continue; }
                count = count.checked_add(r.end-r.start).ok_or("selected rows overflow")?;
                if let Some(p) = merged.last_mut().filter(|p| p.end == r.start) { p.end = r.end; } else { merged.push(r); }
            }
            for r in &merged { append(r.start.checked_mul(stride).ok_or("selected row offset overflow")?,
                (r.end-r.start).checked_mul(stride).ok_or("selected row bytes overflow")?, 0, 0)?; }
            (Selection::Rows(merged), if t.shape.len() == 2 { count } else { 1 },
                if t.shape.len() == 2 { t.shape[1] } else { count.checked_mul(stride/t.dtype.bytes()).ok_or("selected elements overflow")? })
        }
        Selection::Columns(r) => {
            if t.shape.len() != 2 || r.start > r.end || r.end > t.shape[1] { return Err("invalid original column selection".into()); }
            let width = (r.end-r.start).checked_mul(t.dtype.bytes()).ok_or("selected width overflow")?;
            append(r.start.checked_mul(t.dtype.bytes()).ok_or("selected column offset overflow")?,
                t.shape[0].checked_mul(width).ok_or("selected column bytes overflow")?, width, stride)?;
            (Selection::Columns(r.clone()), t.shape[0], r.end-r.start)
        }
        Selection::Absent => return Ok((Selection::Absent, spans, 0, 0)),
    };
    Ok((selection, spans, size(rows)?, size(cols)?))
}
/// A local contiguous FP8 segment and its intersecting original BF16 block grid.
pub struct Fp8Slice { pub rows: std::ops::Range<u64>, pub cols: std::ops::Range<u64>, pub weight_offset: usize, pub scale_offset: usize }
impl TensorView<'_> {
    pub fn fp8_slices(&self, scales: &Self) -> Result<Vec<Fp8Slice>> {
        if self.tensor.shape.len() != 2 || scales.tensor.shape.len() != 2 || self.tensor.dtype != Dtype::F8E4M3 || scales.tensor.dtype != Dtype::Bf16 { return Err("invalid selected FP8 pair".into()); }
        if scales.tensor.name != format!("{}_scale_inv", self.tensor.name) || scales.tensor.role != (Role { store: Store::BlockScale, ..self.tensor.role }) { return Err("selected scale identity differs from FP8 weight".into()); }
        let blocks = |r: &std::ops::Range<u64>| r.start/128..r.end.div_ceil(128);
        let (rows, cols, expected) = match &self.selection {
            Selection::All => (vec![0..self.tensor.shape[0]], 0..self.tensor.shape[1], Selection::All),
            Selection::Rows(rs) => (rs.clone(), 0..self.tensor.shape[1], Selection::Rows(rs.iter().map(blocks).collect())),
            Selection::Columns(c) => (vec![0..self.tensor.shape[0]], c.clone(), Selection::Columns(blocks(c))),
            Selection::Absent => return Err("absent FP8 pair".into()),
        };
        if selected(scales.tensor, expected, 0)?.0 != scales.selection { return Err("scales do not cover original FP8 selection".into()); }
        let mut local_row = 0u64; let mut scale_row = 0u64; let mut slices = Vec::new();
        for r in rows {
            let scale_rows = r.end.div_ceil(128)-r.start/128;
            slices.push(Fp8Slice { weight_offset: size(local_row.checked_mul(self.cols as u64).ok_or("FP8 offset overflow")?)?,
                scale_offset: size(scale_row.checked_mul(scales.cols as u64).and_then(|n| n.checked_mul(2)).ok_or("scale offset overflow")?)?, rows: r.clone(), cols: cols.clone() });
            local_row += r.end-r.start; scale_row += scale_rows;
        }
        if local_row != self.rows as u64 || scale_row != scales.rows as u64 { return Err("selected FP8 row groups disagree with scales".into()); }
        Ok(slices)
    }
}
/// Arena and payload totals differ only by alignment, never expert expansion.
pub struct LoadPlan<'p> {
    pub rank: u32, pub owners: Owners,
    bytes: usize, payload_bytes: usize, tensors: BTreeMap<String, TensorView<'p>>,
    shards: BTreeMap<String, Vec<String>>, headers: &'p BTreeMap<String, Header>,
}
fn size(n: u64) -> Result<usize> { usize::try_from(n).map_err(|_| "weight size exceeds address space".into()) }
impl<'p> LoadPlan<'p> {
    pub fn bytes(&self) -> usize { self.bytes }
    pub fn payload_bytes(&self) -> usize { self.payload_bytes }
    pub fn tensors(&self) -> &BTreeMap<String, TensorView<'p>> { &self.tensors }
    pub fn new(plan: &'p ModelPlan, index: &Index, headers: &'p BTreeMap<String, Header>, split: &SplitPlan) -> Result<Self> {
        if split.ledgers.len() != 1 { return Err("typed hybrid loader requires one rank".into()); }
        Self::for_rank(plan, index, headers, split, 0)
    }
    pub fn for_rank(plan: &'p ModelPlan, index: &Index, headers: &'p BTreeMap<String, Header>, split: &SplitPlan, rank: u32) -> Result<Self> {
        let ledger = split.ledgers.get(rank as usize).ok_or("invalid resident rank")?;
        let owners = split.owners.get(rank as usize).ok_or("invalid resident owner")?.clone();
        if usize::try_from(split.ranks).ok() != Some(split.owners.len()) || split.owners.len() != split.ledgers.len() {
            return Err("inconsistent rank ownership".into());
        }
        plan.hybrid_projections().map_err(|e| e.to_string())?;
        let arms = crate::arms::Table::new(plan,split.ranks as usize,split.draft_enabled());
        let mut out = Self { rank, owners, bytes: 0, payload_bytes: 0, tensors: BTreeMap::new(), shards: BTreeMap::new(), headers };
        for t in &plan.shape.tensors {
            let selection = split.selection(rank, t).map_err(|e| e.to_string())?;
            if selection == Selection::Absent || !arms.reads(t) { continue; }
            if !matches!(t.dtype, Dtype::Bf16 | Dtype::F8E4M3 | Dtype::F32) { return Err("unsupported resident dtype".into()); }
            let shard = index.weight_map.get(&t.name).ok_or("weight absent from index")?;
            let header = headers.get(shard).ok_or("weight shard header absent")?;
            let physical = header.tensors.get(&t.name).ok_or("weight absent from header")?;
            let elements = t.shape.iter().try_fold(1u64, |n, d| n.checked_mul(*d)).ok_or("weight shape overflow")?;
            let bytes = elements.checked_mul(t.dtype.bytes()).ok_or("weight byte overflow")?;
            if bytes == 0 || physical.shape != t.shape || physical.dtype != t.dtype
                || physical.end.checked_sub(physical.start) != Some(bytes) {
                return Err(format!("{}: resident header differs from reconciled plan", t.name));
            }
            let file_offset = header.data_start.checked_add(physical.start).ok_or("weight file offset overflow")?;
            if file_offset.checked_add(bytes).is_none_or(|n| n > header.file_bytes) { return Err("weight exceeds shard".into()); }
            let (selection, spans, rows, cols) = selected(t, selection, file_offset)?;
            let bytes = spans.iter().try_fold(0usize, |n, s| n.checked_add(s.bytes)).ok_or("selected bytes overflow")?;
            if bytes == 0 { continue; }
            if rows > u32::MAX as usize || cols > u32::MAX as usize { return Err("weight dimensions exceed kernel ABI".into()); }
            let file_offset = spans[0].file_offset;
            out.tensors.insert(t.name.clone(), TensorView { tensor: t, offset: 0, bytes, file_offset, rows, cols, selection, spans });
            out.shards.entry(shard.clone()).or_default().push(t.name.clone());
        }
        // Stable file order; 16-byte starts support shared FP8/BF16 vector loads.
        for names in out.shards.values_mut() {
            names.sort_by_key(|n| out.tensors[n].file_offset);
        }
        for name in out.shards.values().flatten() {
            let v = out.tensors.get_mut(name).unwrap();
            v.offset = out.bytes.checked_add(15).ok_or("weight alignment overflow")? / 16 * 16;
            out.bytes = v.offset.checked_add(v.bytes).ok_or("weight arena overflow")?;
            out.payload_bytes = out.payload_bytes.checked_add(v.bytes).ok_or("weight payload overflow")?;
        }
        for (name, v) in &out.tensors {
            if v.tensor.dtype == Dtype::F8E4M3 { v.fp8_slices(out.tensors.get(&format!("{name}_scale_inv")).ok_or("selected FP8 scales absent")?)?; }
        }
        if out.bytes == 0 { return Err("empty resident weight plan".into()); }
        if ledger.weights != out.payload_bytes as u64 {
            return Err("resident payload differs from SplitPlan ledger".into());
        }
        Ok(out)
    }
    /// Reopen all selected shards before allocation; retain these descriptors through upload.
    pub fn open(&self, root: &Path) -> Result<Vec<TensorFile>> {
        self.shards.keys().map(|name| {
            let limits = Limits { header_bytes: HEADER_BYTES, tensors: MAX_TENSORS, rank: MAX_TENSOR_RANK, shards: MAX_SHARDS };
            let file = TensorFile::open(&root.join(name), limits).map_err(|e| e.to_string())?;
            let (a, b) = (&self.headers[name], file.header());
            if a.tensors != b.tensors || a.metadata != b.metadata || a.data_start != b.data_start
                || a.header_bytes != b.header_bytes || a.file_bytes != b.file_bytes {
                return Err("checkpoint changed after text inspection".into());
            }
            Ok(file)
        }).collect()
    }
}

#[cfg(target_os = "linux")]
pub struct Resident<'a> {
    pub arena: spark_gpu_probe::driver::WeightArena<'a>,
    pub matrices: BTreeMap<String, spark_gpu_probe::driver::Matrix<'a>>,
    /// Contiguous block-aligned projections use the ExpertProjection ABI.
    pub fp8: BTreeMap<String, spark_gpu_probe::fp8_abi::ExpertProjection>,
    /// Disjoint original row groups have separate descriptors in local row order.
    pub fp8_segments: BTreeMap<String, Vec<spark_gpu_probe::fp8_abi::ExpertProjection>>,
    pub rank: u32, pub owners: Owners,
}
#[cfg(target_os = "linux")]
impl LoadPlan<'_> {
    pub fn load<'a>(&self, backend: &spark_gpu_probe::driver::Backend<'a>, root: &Path) -> Result<Resident<'a>> {
        use spark_gpu_probe::{load::{WeightDtype, WeightFile, WeightRange}, fp8_abi::ExpertProjection};
        let files = self.open(root)?;
        let ranges = self.shards.values().map(|names| names.iter().flat_map(|n| {
            let v = &self.tensors[n]; v.spans.iter().map(move |s| WeightRange { file_offset: s.file_offset,
                destination: v.offset + s.local_offset, bytes: s.bytes, row_bytes: s.row_bytes, row_stride: s.row_stride })
        }).collect::<Vec<_>>()).collect::<Vec<_>>();
        let shards = files.iter().zip(&ranges).map(|(f, ranges)| WeightFile { file: f.file(), ranges }).collect::<Vec<_>>();
        let arena = backend.load_weight_arena(self.bytes, &shards)?;
        let mut matrices = BTreeMap::new(); let mut fp8 = BTreeMap::new(); let mut fp8_segments = BTreeMap::new();
        for (name, v) in &self.tensors {
            let dtype = match v.tensor.dtype { Dtype::Bf16 => WeightDtype::Bf16, Dtype::F8E4M3 => WeightDtype::Fp8,
                Dtype::F32 => WeightDtype::F32, _ => return Err("unsupported resident dtype".into()) };
            matrices.insert(name.clone(), backend.weight_matrix(&arena, v.offset, v.rows, v.cols, dtype)?);
        }
        for (name, v) in &self.tensors {
            if v.tensor.dtype != Dtype::F8E4M3 { continue; }
            let matrix = matrices.remove(name).ok_or("FP8 matrix absent")?;
            let scales = matrices.remove(&format!("{name}_scale_inv")).ok_or("BF16 scales absent")?;
            let slices = v.fp8_slices(&self.tensors[&format!("{name}_scale_inv")])?;
            let mut descriptors = Vec::new();
            for s in &slices {
                if s.rows.start % 128 == 0 && (s.rows.end-s.rows.start) % 128 == 0 && s.cols.start % 128 == 0 && v.cols % 128 == 0 {
                    descriptors.push(ExpertProjection::checked(matrix.device_address() + s.weight_offset as u64,
                        scales.device_address() + s.scale_offset as u64, s.rows.end-s.rows.start, v.cols as u64,
                        s.rows.start/128, s.cols.start/128).ok_or("FP8 descriptor overflow")?);
                }
            }
            if descriptors.len() == 1 && slices.len() == 1 { fp8.insert(name.clone(), descriptors[0]); }
            if descriptors.len() == slices.len() { fp8_segments.insert(name.clone(), descriptors); }
            matrices.insert(name.clone(), matrix.with_selected_fp8_scales(scales, [size(slices[0].rows.start)?, size(slices[0].cols.start)?])?);
        }
        Ok(Resident { arena, matrices, fp8, fp8_segments, rank: self.rank, owners: self.owners.clone() })
    }
}
