//! Selected official checkpoint bytes in one allocation, with global coordinates.
use engine_model::{
    flash::{
        Checkpoint, WeightFormat,
        plan::{RankPlan, Selection},
    },
    safetensors::{Dtype, Tensor},
};
use spark_gpu_probe::{
    driver::{Backend, PromptBuffer, WeightArena},
    kernel_params::{
        FlashBf16Pack, HybridCopy, HybridCopyRow, flash_bf16_element, flash_bf16_packed, shard_ranges,
    },
    load::{WeightFile, WeightRange},
};
use std::{collections::BTreeMap, fs::File, path::Path};

type Result<T> = std::result::Result<T, String>;

/// Local packing never changes the source tensor or its scale-grid coordinates.
pub struct TensorSlice {
    pub tensor: Tensor,
    pub selection: Selection,
    pub offset: usize,
    pub bytes: usize,
    packed_cols: usize,
}
impl TensorSlice {
    pub fn global_byte_offset(&self, local: u64) -> Result<u64> {
        self.selection
            .global_byte_offset(&self.tensor, local)
            .map_err(|e| e.to_string())
    }
    /// Inverse mapping for a requested global element (including scale cells).
    pub fn local_byte_offset(&self, global: u64) -> Result<usize> {
        let local = self
            .selection
            .local_byte_offset(&self.tensor, global)
            .ok_or("global tensor coordinate is not owned by this rank")?;
        usize::try_from(local).map_err(|e| e.to_string())
    }
}

pub struct Weights<'a> {
    pub format: WeightFormat,
    pub mixed: bool,
    pub plan: RankPlan,
    pub tensors: BTreeMap<String, TensorSlice>,
    pub payload_bytes: usize,
    /// K-split shard ranges; declared before the arena so they unregister first.
    _split: Vec<crate::flash_split_ranges::Registration>,
    arena: WeightArena<'a>,
}
struct Entry<'a> {
    shard: &'a str,
    name: &'a str,
    tensor: &'a Tensor,
    selection: Selection,
    offset: usize,
    bytes: usize,
}

impl<'a> Entry<'a> {
    fn layout(
        checkpoint: &'a Checkpoint,
        plan: &RankPlan,
        select: impl Fn(&str) -> bool,
    ) -> Result<(Vec<Self>, usize, usize)> {
        let mut entries = Vec::new();
        for (name, tensor) in checkpoint.tensors() {
            if !select(name) {
                continue;
            }
            let selection = if checkpoint.config.mixed.is_some() {
                plan.mixed_selection(name)
            } else {
                plan.selection_for(name, checkpoint.config.format)
            }
            .map_err(|e| e.to_string())?;
            let bytes = usize::try_from(selection.bytes(tensor)).map_err(|e| e.to_string())?;
            if bytes == 0 {
                continue;
            }
            entries.push(Entry {
                shard: &checkpoint.index.weight_map[name],
                name,
                tensor,
                selection,
                offset: 0,
                bytes,
            });
        }
        if entries.is_empty() {
            return Err("no resident tensors match the requested block".into());
        }
        // File order gives the shared reader pool sequential windows within each shard.
        entries.sort_unstable_by(|a, b| (a.shard, a.tensor.start).cmp(&(b.shard, b.tensor.start)));
        let mut bytes = 0usize;
        let mut payload_bytes = 0usize;
        for entry in &mut entries {
            let alignment = checkpoint.config.format.arena_alignment(entry.tensor.dtype);
            bytes = bytes
                .checked_add(alignment - 1)
                .ok_or("weight arena overflow")?
                & !(alignment - 1);
            entry.offset = bytes;
            bytes = bytes
                .checked_add(entry.bytes)
                .ok_or("weight arena overflow")?;
            payload_bytes = payload_bytes
                .checked_add(entry.bytes)
                .ok_or("weight payload overflow")?;
        }
        Ok((entries, bytes, payload_bytes))
    }
}

impl<'a> Weights<'a> {
    /// Stream only the reconciled rank's resident selections into one arena.
    /// CSF containers are expanded separately and never retained on the GPU.
    pub fn load_shard(
        b: &Backend<'a>,
        cp: &engine_model::flash::shard::ShardCheckpoint,
    ) -> Result<Self> {
        use engine_model::flash::shard::Placement;
        b.require_nvfp4_device()?;
        let mut shards = BTreeMap::<String, (File, Vec<WeightRange>)>::new();
        let (mut tensors, mut bytes, mut payload_bytes) = (BTreeMap::new(), 0usize, 0usize);
        for (name, t) in cp.resident_order() {
            if t.placement != Placement::Resident || name.contains(".nvfp4_csf_") || t.bytes() == 0
            {
                continue;
            }
            let (file, base) = cp.source(name).map_err(|e| e.to_string())?;
            if !shards.contains_key(&t.file) {
                shards.insert(
                    t.file.clone(),
                    (file.try_clone().map_err(|e| e.to_string())?, Vec::new()),
                );
            }
            let size = usize::try_from(t.bytes()).map_err(|e| e.to_string())?;
            // NVFP4 matrices (U8) and BF16 start on 256 bytes, everything else on 16:
            // fp4_bw streams 64-byte row chunks with 256-byte L2 fills, and the file's
            // 4-byte weight_scale_2 scalars leave most matrices 16-32 bytes off a 64-byte
            // boundary, which cost fp4_bw_down +48-55 % (kbench KBENCH_MISALIGN). Same bytes.
            let align = if matches!(t.tensor.dtype, Dtype::U8 | Dtype::Bf16) {
                255
            } else {
                15
            };
            bytes = bytes.checked_add(align).ok_or("Next arena overflow")? & !align;
            let offset = bytes;
            bytes = bytes.checked_add(size).ok_or("Next arena overflow")?;
            payload_bytes += size;
            let ranges = &mut shards.get_mut(&t.file).unwrap().1;
            if let Selection::Columns(r) = &t.selection {
                ranges.push(WeightRange {
                    file_offset: base + r.start * t.tensor.dtype.bytes(),
                    destination: offset,
                    bytes: size,
                    row_bytes: ((r.end - r.start) * t.tensor.dtype.bytes()) as usize,
                    row_stride: (t.tensor.shape[1] * t.tensor.dtype.bytes()) as usize,
                });
            } else {
                let mut destination = offset;
                for (at, len) in t.selection.spans(&t.tensor) {
                    ranges.push(WeightRange {
                        file_offset: base + at,
                        destination,
                        bytes: len as usize,
                        row_bytes: 0,
                        row_stride: 0,
                    });
                    destination += len as usize;
                }
            }
            tensors.insert(
                name.clone(),
                TensorSlice {
                    tensor: t.tensor.clone(),
                    selection: t.selection.clone(),
                    offset,
                    bytes: size,
                    packed_cols: 0,
                },
            );
        }
        let files = shards
            .values()
            .map(|(file, ranges)| WeightFile { file, ranges })
            .collect::<Vec<_>>();
        bytes = bytes.checked_add(15).ok_or("Next arena overflow")? & !15;
        let arena = b.load_weight_arena(bytes, &files)?;
        let staging = b.vector(64 * 10240 / 2)?;
        for t in tensors.values_mut() {
            if t.tensor.dtype != Dtype::Bf16 || t.tensor.shape.len() != 2 {
                continue;
            }
            let cols = match &t.selection {
                Selection::Columns(r) => (r.end - r.start) as usize,
                _ => t.tensor.shape[1] as usize,
            };
            let rows = t.bytes / (2 * cols);
            if flash_bf16_packed(rows, cols) {
                super::next::pack_bf16(
                    b,
                    arena.device_address() + t.offset as u64,
                    rows,
                    cols,
                    &staging,
                )?;
                t.packed_cols = cols;
            }
        }
        b.synchronize()?;
        Ok(Self {
            format: WeightFormat::Nvfp4,
            mixed: true,
            plan: cp.plan.execution_plan().map_err(|e| e.to_string())?,
            tensors,
            payload_bytes,
            _split: Vec::new(),
            arena,
        })
    }
    /// Exact device allocation size, including scalar alignment, without reading payloads.
    pub fn planned_bytes(checkpoint: &Checkpoint, plan: RankPlan) -> Result<usize> {
        Ok(Entry::layout(checkpoint, &plan, |_| true)?.1)
    }
    pub fn load(
        backend: &Backend<'a>,
        root: &Path,
        checkpoint: &Checkpoint,
        plan: RankPlan,
    ) -> Result<Self> {
        Self::load_prefix(backend, root, checkpoint, plan, "")
    }
    /// Load an independently executable checkpoint block using the same rank
    /// ownership and single-allocation layout as a complete resident model.
    pub fn load_prefix(
        backend: &Backend<'a>,
        root: &Path,
        checkpoint: &Checkpoint,
        plan: RankPlan,
        prefix: &str,
    ) -> Result<Self> {
        Self::load_selected(backend, root, checkpoint, plan, |name| {
            name.starts_with(prefix)
        })
    }
    /// Keep independently streamed blocks and persistent input/output weights
    /// disjoint, while retaining the same official rank ownership rules.
    pub fn load_selected(
        backend: &Backend<'a>,
        root: &Path,
        checkpoint: &Checkpoint,
        plan: RankPlan,
        select: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        if checkpoint.config.format == WeightFormat::Nvfp4 {
            checkpoint
                .config
                .format
                .require_opt_in(WeightFormat::opt_in().map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            backend.require_nvfp4_device()?;
        }
        let started = std::time::Instant::now();
        checkpoint.prepare_load(root).map_err(|e| e.to_string())?;
        let (entries, bytes, payload_bytes) = Entry::layout(checkpoint, &plan, select)?;
        let mut shards = BTreeMap::<&str, (File, Vec<WeightRange>)>::new();
        for entry in &entries {
            if !shards.contains_key(entry.shard) {
                let file = File::open(root.join(entry.shard)).map_err(|e| e.to_string())?;
                if file.metadata().map_err(|e| e.to_string())?.len()
                    != checkpoint.shards[entry.shard].file_bytes
                {
                    return Err("checkpoint shard changed after inspection".into());
                }
                shards.insert(entry.shard, (file, Vec::new()));
            }
            let ranges = &mut shards.get_mut(entry.shard).unwrap().1;
            let base = checkpoint.shards[entry.shard].data_start + entry.tensor.start;
            if let Selection::Columns(columns) = &entry.selection {
                let scalar = entry.tensor.dtype.bytes();
                ranges.push(WeightRange {
                    file_offset: base + columns.start * scalar,
                    destination: entry.offset,
                    bytes: entry.bytes,
                    row_bytes: ((columns.end - columns.start) * scalar) as usize,
                    row_stride: (entry.tensor.shape[1] * scalar) as usize,
                });
            } else {
                let mut destination = entry.offset;
                for (offset, count) in entry.selection.spans(entry.tensor) {
                    let count = usize::try_from(count).map_err(|e| e.to_string())?;
                    ranges.push(WeightRange {
                        file_offset: base + offset,
                        destination,
                        bytes: count,
                        row_bytes: 0,
                        row_stride: 0,
                    });
                    destination += count;
                }
            }
        }
        let files = shards
            .values()
            .map(|(file, ranges)| WeightFile { file, ranges })
            .collect::<Vec<_>>();
        let planned = started.elapsed();
        let arena = backend.load_weight_arena(bytes, &files)?;
        let streamed = started.elapsed();
        drop(files);
        drop(shards);
        let mut tensors = BTreeMap::new();
        // One bounded tile stages the in-place canonical permutation. No full
        // matrix is duplicated, and the tile is released before inference.
        let staging = {
            let width = entries
                .iter()
                .filter(|e| e.tensor.dtype == Dtype::Bf16 && e.tensor.shape.len() == 2)
                .map(|e| e.tensor.shape[1] as usize)
                .max()
                .unwrap_or(0);
            backend.vector(super::next::PACK_TILES * 64 * width / 2)?
        };
        for e in entries {
            let mut packed_cols = 0;
            if e.tensor.dtype == Dtype::Bf16
                && e.tensor.shape.len() == 2
                && e.name != "model.language_model.embed_tokens.weight"
            {
                let cols = match &e.selection {
                    Selection::Columns(columns) => (columns.end - columns.start) as usize,
                    _ => e.tensor.shape[1] as usize,
                };
                let rows = e.bytes / (cols * 2);
                if flash_bf16_packed(rows, cols) {
                    let mut params = FlashBf16Pack {
                        source: 0,
                        weight: arena.device_address() + e.offset as u64,
                        rows: rows as u32,
                        cols: cols as u32,
                    };
                    if rows % 64 == 0 {
                        // Consecutive 64-row tiles per step; same bytes as one tile per step.
                        let source = staging.device_address();
                        let tiles = (staging.len() * 4 / (64 * cols * 2)).clamp(1, super::next::PACK_TILES);
                        for row in (0..rows).step_by(64 * tiles) {
                            let n = (rows - row).min(64 * tiles);
                            let words = n * cols / 2;
                            let destination =
                                arena.device_address() + (e.offset + row * cols * 2) as u64;
                            let mut copy = HybridCopy {
                                source: destination as *const u32,
                                destination: source as *mut u32,
                                words: words as u32,
                                padding: 0,
                                rows: 0,
                            };
                            params.source = source;
                            params.weight = destination;
                            params.rows = n as u32;
                            unsafe {
                                backend.launch_flash_rows(
                                    "flash_hybrid_copy",
                                    words.div_ceil(128) as u32,
                                    1,
                                    128,
                                    0,
                                    &mut [(&mut copy as *mut HybridCopy).cast()],
                                )?;
                                backend.launch_flash_shared(
                                    "flash_bf16_pack",
                                    (n / 64 * (cols / 8)) as u32,
                                    128,
                                    0,
                                    &mut [(&mut params as *mut FlashBf16Pack).cast()],
                                )?;
                            }
                        }
                    } else {
                        unsafe {
                            backend.launch_flash_shared(
                                "flash_bf16_pack",
                                (rows / 4) as u32,
                                128,
                                (cols * 8) as u32,
                                &mut [(&mut params as *mut FlashBf16Pack).cast()],
                            )?;
                        }
                    }
                    packed_cols = cols;
                }
            }
            tensors.insert(
                e.name.to_owned(),
                TensorSlice {
                    tensor: e.tensor.clone(),
                    selection: e.selection,
                    offset: e.offset,
                    bytes: e.bytes,
                    packed_cols,
                },
            );
        }
        backend.synchronize()?;
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        eprintln!("flash_weights load plan_ms={:.0} arena_ms={:.0} pack_ms={:.0} bytes={bytes}",
            ms(planned), ms(streamed - planned), ms(started.elapsed() - streamed));
        // A BF16 matrix stored by a column (K) range is a K-split shard.
        let mut split = Vec::new();
        for t in tensors.values() {
            if let Selection::Columns(r) = &t.selection && t.tensor.dtype == Dtype::Bf16 && t.tensor.shape.len() == 2 {
                split.extend(crate::flash_split_ranges::register(arena.device_address() + t.offset as u64,
                    shard_ranges((r.end - r.start) as usize, t.tensor.shape[1] as usize))?);
            }
        }
        Ok(Self {
            format: checkpoint.config.format,
            mixed: checkpoint.config.mixed.is_some(),
            plan,
            tensors,
            payload_bytes,
            _split: split,
            arena,
        })
    }
    pub fn resident_bytes(&self) -> usize {
        self.arena.bytes()
    }
    /// Stored arena bytes after all prior device work (load digest receipts).
    pub fn read_arena(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        self.arena.read(offset, out)
    }
    /// Bytes of every owned tensor whose name starts with `prefix`; visits only that key range.
    pub fn prefix_bytes(&self, prefix: &str) -> usize {
        use std::ops::Bound;
        self.tensors
            .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
            .take_while(|(name, _)| name.starts_with(prefix))
            .map(|(_, t)| t.bytes)
            .sum()
    }
    /// The rank's packed attention projections (GDN qkv, z, out; QSA q/gate, o; the MTP layer's
    /// too) with K 2560 or 3072 (the native decode entries) as (address, rows, cols) of the stored
    /// packed matrix, for blocks::Lbf12Twins.
    pub(super) fn attention_matrices(&self) -> Vec<(u64, usize, usize)> {
        const PARTS: [&str; 5] = [".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
            ".linear_attn.out_proj.weight", ".self_attn.q_proj.weight", ".self_attn.o_proj.weight"];
        self.tensors.iter()
            .filter(|(name, t)| matches!(t.packed_cols, 2560 | 3072) && t.tensor.dtype == Dtype::Bf16 && PARTS.iter().any(|p| name.ends_with(p)))
            .map(|(_, t)| (self.arena.device_address() + t.offset as u64, t.bytes / 2 / t.packed_cols, t.packed_cols))
            .collect()
    }
    pub fn tensor(&self, name: &str) -> Result<&TensorSlice> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("rank does not own {name}"))
    }
    pub fn device_address(&self, name: &str) -> Result<u64> {
        Ok(self.arena.device_address() + self.tensor(name)?.offset as u64)
    }
    pub(super) fn row_address(&self, name: &str, offset: usize, bytes: usize) -> Result<u64> {
        let tensor = self.tensor(name)?;
        if tensor.packed_cols != 0
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > tensor.bytes)
        {
            return Err("batch lookup requires a bounded raw checkpoint row".into());
        }
        Ok(self.arena.device_address() + (tensor.offset + offset) as u64)
    }
    /// Every BF16 element of a local [rows, cols] matrix, element (r, c) written to
    /// out[2 * place(r, c)..]: one device read of the resident bytes, whatever their layout.
    pub fn read_bf16_matrix(&self, name: &str, cols: usize, place: impl Fn(usize, usize) -> usize, out: &mut [u8]) -> Result<()> {
        let tensor = self.tensor(name)?;
        if tensor.tensor.dtype != Dtype::Bf16 || cols == 0 || tensor.bytes % (2 * cols) != 0 || out.len() != tensor.bytes
            || tensor.packed_cols != 0 && tensor.packed_cols != cols {
            return Err(format!("{name}: BF16 matrix read shape"));
        }
        let rows = tensor.bytes / (2 * cols);
        let mut raw = vec![0u8; tensor.bytes];
        self.arena.read(tensor.offset, &mut raw)?;
        for r in 0..rows {
            for c in 0..cols {
                let from = if tensor.packed_cols == 0 { r * cols + c } else { flash_bf16_element(r, c, cols, rows) };
                let to = place(r, c);
                if to >= rows * cols { return Err(format!("{name}: BF16 matrix placement")); }
                out[2 * to..2 * to + 2].copy_from_slice(&raw[2 * from..2 * from + 2]);
            }
        }
        Ok(())
    }
    pub fn read(&self, name: &str, local_offset: usize, out: &mut [u8]) -> Result<()> {
        let tensor = self.tensor(name)?;
        if local_offset
            .checked_add(out.len())
            .is_none_or(|n| n > tensor.bytes)
        {
            return Err("tensor read outside local slice".into());
        }
        if tensor.packed_cols == 0 {
            return self.arena.read(tensor.offset + local_offset, out);
        }
        // Preserve logical checkpoint coordinates for bounded inspection;
        // inference reads packed fragments directly without host copies.
        let (mut at, mut out) = (local_offset, out);
        while !out.is_empty() {
            let element = at / 2;
            let physical = flash_bf16_element(
                element / tensor.packed_cols,
                element % tensor.packed_cols,
                tensor.packed_cols,
                tensor.bytes / (2 * tensor.packed_cols),
            );
            let count = out.len().min(2 - at % 2);
            self.arena
                .read(tensor.offset + physical * 2 + at % 2, &mut out[..count])?;
            at += count;
            out = &mut out[count..];
        }
        Ok(())
    }
    pub(super) fn stage_prompt_rows(
        &self,
        b: &Backend<'_>,
        slot: &mut PromptBuffer<'_>,
        copies: &[HybridCopyRow],
    ) -> Result<()> {
        let base = self.arena.device_address();
        let rows = copies
            .iter()
            .map(|row| {
                Ok((
                    usize::try_from(
                        row.source
                            .checked_sub(base)
                            .ok_or("prompt row outside weight arena")?,
                    )
                    .map_err(|e| e.to_string())?,
                    usize::try_from(row.destination).map_err(|e| e.to_string())?,
                    row.words as usize * 4,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        b.stage_prompt_rows(&self.arena, slot, &rows)
    }
    /// All owned values of a small BF16/F32 vector in one device read, with the
    /// same conversion as value_f32 (one synchronizing copy, not one per value).
    pub fn values_f32(&self, name: &str) -> Result<Vec<f32>> {
        let slice = self.tensor(name)?;
        let mut raw = vec![0u8; slice.bytes];
        self.read(name, 0, &mut raw)?;
        match slice.tensor.dtype {
            Dtype::Bf16 => Ok(raw
                .as_chunks::<2>().0.iter()
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect()),
            Dtype::F32 => Ok(raw
                .as_chunks::<4>().0.iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            _ => Err(format!("{name}: bulk read supports BF16/F32 vectors only")),
        }
    }
    /// Read one owned numeric weight using the official FP32 conversion. This
    /// bounded inspection path does not expand or retain any tensor on the host.
    pub fn value_f32(&self, name: &str, local_element: usize) -> Result<f32> {
        let slice = self.tensor(name)?;
        if self.format == WeightFormat::Nvfp4 && slice.tensor.dtype == Dtype::U8 {
            return super::nvfp4::Matrix::load(
                self,
                name.strip_suffix(".weight").ok_or("NVFP4 tensor suffix")?,
            )?
            .value_f32(local_element);
        }
        let width = slice.tensor.dtype.bytes() as usize;
        let local = local_element
            .checked_mul(width)
            .ok_or("element offset overflow")?;
        let mut raw = [0u8; 8];
        self.read(name, local, &mut raw[..width])?;
        let value = match slice.tensor.dtype {
            Dtype::Bf16 => {
                f32::from_bits((u16::from_le_bytes(raw[..2].try_into().unwrap()) as u32) << 16)
            }
            Dtype::F32 => f32::from_le_bytes(raw[..4].try_into().unwrap()),
            Dtype::F8E4M3 => {
                let weight = fp8_e4m3fn(raw[0]);
                let (scale_name, global_cell) =
                    if let Some((prefix, _)) = name.split_once(".shard_") {
                        (format!("{prefix}.weight_scale"), 0)
                    } else {
                        let global = slice.global_byte_offset(local as u64)?;
                        let cols = slice.tensor.shape[1];
                        (
                            format!(
                                "{}.weight_scale_inv",
                                name.strip_suffix(".weight").ok_or("FP8 tensor suffix")?
                            ),
                            (global / cols / 128) * cols.div_ceil(128) + (global % cols) / 128,
                        )
                    };
                let scale = self.tensor(&scale_name)?;
                let local_scale = scale.local_byte_offset(global_cell * 2)?;
                let mut bytes = [0u8; 2];
                self.read(&scale_name, local_scale, &mut bytes)?;
                let value = weight * f32::from_bits((u16::from_le_bytes(bytes) as u32) << 16);
                if name.contains(".mlp.experts.") {
                    let bits = value.to_bits();
                    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
                } else {
                    value
                }
            }
            _ => return Err("numeric inspection supports BF16, FP32 and E4M3FN weights".into()),
        };
        Ok(value)
    }
}

/// E4M3FN conversion, including subnormals and its sole NaN mantissa.
pub fn fp8_e4m3fn(byte: u8) -> f32 {
    let exponent = (byte >> 3) & 15;
    let mantissa = byte & 7;
    let value = if exponent == 0 {
        mantissa as f32 / 512.0
    } else if exponent == 15 && mantissa == 7 {
        f32::NAN
    } else {
        (8 + mantissa) as f32 * 2.0f32.powi(exponent as i32 - 10)
    };
    if byte & 128 != 0 { -value } else { value }
}
