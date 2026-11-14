//! Mixed Next ownership: whole routed experts, head/channel TP elsewhere.
//! CSF containers follow their expert intact. PLE/embedding rows stay on disk.
//! The map is data: reconcile every file, dtype and shape before payload I/O.
use super::{mixed, partition, plan::Selection};
use crate::{
    Result, invalid,
    safetensors::{self, Header, Index, Tensor},
    shapes::*,
};
use std::{
    collections::BTreeMap,
    fs::File,
    ops::Range,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Resident,
    Disk,
    Absent,
}
#[derive(Debug, Clone)]
pub struct TensorShard {
    pub file: String,
    pub tensor: Tensor,
    pub role: Role,
    pub selection: Selection,
    pub placement: Placement,
}
impl TensorShard {
    pub fn bytes(&self) -> u64 {
        self.selection.bytes(&self.tensor)
    }
}
#[derive(Debug, Clone, Copy, Default)]
pub struct Budget {
    pub resident: u64,
    pub disk: u64,
    pub experts: u64,
}
#[derive(Debug)]
pub struct ShardPlan {
    pub rank: usize,
    pub world: usize,
    pub expert_tp: usize,
    /// W2/W4 whole experts with replicated attention/GDN/shared expert/HC.
    pub ep2: bool,
    /// Typed geometry without the duplicated full tensor inventory.
    pub shape: ModelShape,
    pub tensors: BTreeMap<String, TensorShard>,
    pub budget: Budget,
}
fn require(ok: bool, why: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(why)) }
}
fn scaled(r: Range<u64>, unit: u64) -> Range<u64> {
    r.start * unit..r.end * unit
}
fn grouped(r: Range<u64>, group: u64) -> Range<u64> {
    if r.is_empty() {
        0..0
    } else {
        r.start / group..r.end.div_ceil(group)
    }
}
impl ShardPlan {
    /// Only the explicitly qualified target worlds are accepted. This does not
    /// relax the existing GPU loader's world-1 mixed-storage restriction.
    pub fn new(
        shape: &ModelShape,
        files: &BTreeMap<String, String>,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        require(
            matches!(world, 1 | 2 | 4) && rank < world,
            "Next shard rank/world must be 1, 2 or 4",
        )?;
        require(
            shape.hidden > 0
                && shape.hidden <= MAX_HIDDEN
                && shape.vocab > 0
                && shape.vocab <= MAX_VOCAB
                && !shape.layers.is_empty()
                && shape.layers.len() <= MAX_LAYERS as usize,
            "invalid Next shard geometry",
        )?;
        require(
            shape.max_positions > 0 && shape.max_positions <= MAX_POSITIONS,
            "invalid Next context geometry",
        )?;
        for layer in &shape.layers {
            let Ffn::Experts(e) = layer.ffn else {
                return Err(invalid("Next shard requires MoE layers"));
            };
            require(
                e.count > 0
                    && e.count <= MAX_EXPERTS
                    && e.selected > 0
                    && e.selected <= e.count
                    && e.width > 0
                    && e.width <= MAX_WIDTH
                    && e.shared_width <= MAX_WIDTH,
                "invalid Next expert geometry",
            )?;
            match layer.mixer {
                Mixer::Attention(a) => require(
                    a.kv_heads > 0
                        && a.query_heads <= MAX_HEADS
                        && a.query_heads >= a.kv_heads
                        && a.query_heads % a.kv_heads == 0
                        && a.head_dim > 0
                        && a.head_dim <= MAX_HEAD_DIM
                        && a.index
                            .is_none_or(|i| i.kv_heads <= MAX_HEADS && i.dim <= MAX_HEAD_DIM),
                    "invalid Next attention geometry",
                )?,
                Mixer::Linear(l) => require(
                    l.key_heads > 0
                        && l.value_heads <= MAX_HEADS
                        && l.value_heads >= l.key_heads
                        && l.value_heads % l.key_heads == 0
                        && l.key_dim > 0
                        && l.key_dim <= MAX_HEAD_DIM
                        && l.value_dim > 0
                        && l.value_dim <= MAX_HEAD_DIM
                        && (1..=MAX_WIDTH).contains(&l.conv),
                    "invalid Next GDN geometry",
                )?,
            }
        }
        require(
            files.len() == shape.tensors.len() && !files.is_empty(),
            "incomplete Next tensor file map",
        )?;
        let mut geometry = shape.clone();
        geometry.tensors.clear();
        let mut out = Self {
            rank,
            world,
            expert_tp: 1,
            ep2: false,
            shape: geometry,
            tensors: BTreeMap::new(),
            budget: Budget::default(),
        };
        for entry in &shape.tensors {
            let file = files
                .get(&entry.name)
                .ok_or_else(|| invalid("missing Next tensor file"))?;
            safetensors::validate_mixed_shard_name(file)?;
            let end = entry.shape.iter().try_fold(entry.dtype.bytes(), |n, &d| {
                n.checked_mul(d)
                    .ok_or_else(|| invalid("Next tensor bytes overflow"))
            })?;
            let tensor = Tensor {
                dtype: entry.dtype,
                shape: entry.shape.clone(),
                start: 0,
                end,
            };
            let selection = out.select(entry)?;
            validate_selection(&selection, &tensor)?;
            let placement = if matches!(selection, Selection::Absent) {
                Placement::Absent
            } else if matches!(entry.role.part, Part::Embedding | Part::Lexical { .. }) {
                Placement::Disk
            } else {
                Placement::Resident
            };
            let shard = TensorShard {
                file: file.clone(),
                tensor,
                role: entry.role,
                selection,
                placement,
            };
            let bytes = shard.bytes();
            if placement != Placement::Absent {
                let total = if placement == Placement::Resident {
                    &mut out.budget.resident
                } else {
                    &mut out.budget.disk
                };
                *total = total
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("Next shard budget overflow"))?;
            }
            if matches!(entry.role.part, Part::Expert { .. }) {
                out.budget.experts = out
                    .budget
                    .experts
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("expert bytes overflow"))?;
            }
            require(
                out.tensors.insert(entry.name.clone(), shard).is_none(),
                "duplicate Next tensor",
            )?;
        }
        Ok(out)
    }
    /// W4 groups {0,1}/{2,3}: disjoint expert groups; full-K row TP within each.
    /// CSF coordinate containers remain intact until expansion, then select rows.
    pub fn ep2tp2(mut self) -> Result<Self> {
        require(self.world == 4, "EP2xTP2 requires world four")?;
        for t in self.tensors.values_mut() {
            if matches!(t.role.place, Place::Unused | Place::Draft { .. }) {
                continue;
            }
            let Part::Expert { id: Some(id), proj } = t.role.part else {
                continue;
            };
            let Place::Layer(li) = t.role.place else {
                return Err(invalid("expert outside layer"));
            };
            let Ffn::Experts(e) = self.shape.layers[li as usize].ffn else {
                return Err(invalid("missing experts"));
            };
            let owned = partition(e.count, (self.rank / 2) as u64, 2)?;
            t.selection = if !owned.contains(&u64::from(id)) {
                Selection::Absent
            } else if t.role.store == Store::Weight {
                let rows = if proj == Proj::Down {
                    self.shape.hidden
                } else {
                    e.width
                };
                require(rows % 2 == 0, "hybrid expert rows must divide by two")?;
                Selection::Rows(vec![partition(rows, (self.rank % 2) as u64, 2)?])
            } else {
                Selection::All
            };
            t.placement = if t.selection == Selection::Absent {
                Placement::Absent
            } else {
                Placement::Resident
            };
        }
        self.expert_tp = 2;
        self.budget = Budget::default();
        for t in self.tensors.values() {
            match t.placement {
                Placement::Resident => self.budget.resident += t.bytes(),
                Placement::Disk => self.budget.disk += t.bytes(),
                _ => (),
            }
            if matches!(t.role.part, Part::Expert { .. }) {
                self.budget.experts += t.bytes();
            }
        }
        Ok(self)
    }
    pub fn vocabulary(&self) -> Range<u64> {
        self.part(self.shape.vocab)
    }
    /// Contiguous balanced expert ownership remains as in new(); only the
    /// non-expert compute replicates. Disk rows and the vocabulary stay sharded.
    pub fn with_ep2(mut self) -> Result<Self> {
        require(
            matches!(self.world, 2 | 4) && self.expert_tp == 1,
            "replicated EP requires world two or four without expert TP",
        )?;
        for t in self.tensors.values_mut() {
            if matches!(t.role.place, Place::Unused | Place::Draft { .. })
                || matches!(
                    t.role.part,
                    Part::Expert { .. } | Part::Embedding | Part::Lexical { .. } | Part::Head
                )
            {
                continue;
            }
            t.selection = Selection::All;
            t.placement = Placement::Resident;
        }
        self.ep2 = true;
        self.budget = Budget::default();
        for t in self.tensors.values() {
            match t.placement {
                Placement::Resident => self.budget.resident += t.bytes(),
                Placement::Disk => self.budget.disk += t.bytes(),
                _ => (),
            }
            if matches!(t.role.part, Part::Expert { .. }) {
                self.budget.experts += t.bytes();
            }
        }
        Ok(self)
    }
    /// Kernel geometry can replicate while the output vocabulary stays local.
    pub fn execution_plan(&self) -> Result<super::plan::RankPlan> {
        if !self.ep2 {
            return super::plan::RankPlan::new(self.rank as u64, self.world as u64);
        }
        let mut plan = super::plan::RankPlan::new(0, 1)?;
        plan.vocabulary = self.vocabulary();
        Ok(plan)
    }
    /// Current FP32 caches, including replicated QSA index history and the KV
    /// head overlap at world four. No session scratch or prefix snapshots.
    pub fn cache_bytes(&self, tokens: u64) -> Result<(u64, u64)> {
        require(
            tokens <= self.shape.max_positions,
            "Next context exceeds config",
        )?;
        let (mut kv, mut state) = (0u64, 0u64);
        for layer in &self.shape.layers {
            match layer.mixer {
                Mixer::Attention(a) => {
                    require(
                        a.kv_heads > 0 && a.query_heads >= a.kv_heads,
                        "invalid QSA cache geometry",
                    )?;
                    let r = grouped(
                        if self.ep2 {
                            0..a.query_heads
                        } else {
                            self.part(a.query_heads)
                        },
                        a.query_heads / a.kv_heads,
                    );
                    kv += (2 * (r.end - r.start) * a.head_dim
                        + a.index.map_or(0, |i| i.kv_heads * i.dim))
                        * 4;
                }
                Mixer::Linear(l) => {
                    require(
                        l.key_heads > 0 && l.value_heads >= l.key_heads && l.conv > 0,
                        "invalid GDN cache geometry",
                    )?;
                    let v = if self.ep2 {
                        0..l.value_heads
                    } else {
                        self.part(l.value_heads)
                    };
                    let k = grouped(v.clone(), l.value_heads / l.key_heads);
                    let (v, k) = (v.end - v.start, k.end - k.start);
                    state += (v * l.value_dim * l.key_dim
                        + (2 * k * l.key_dim + v * l.value_dim) * l.conv)
                        * 4
                        + super::gdn_chunk_tail_bytes(v, k, l.key_dim, l.value_dim);
                }
            }
        }
        Ok((
            kv.checked_mul(tokens)
                .ok_or_else(|| invalid("Next cache bytes overflow"))?,
            state,
        ))
    }
    fn part(&self, total: u64) -> Range<u64> {
        partition(total, self.rank as u64, self.world as u64).expect("validated rank/world")
    }
    fn layer(&self, role: Role) -> Result<LayerShape> {
        let Place::Layer(id) = role.place else {
            return Err(invalid("expected decoder layer role"));
        };
        self.shape
            .layers
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid("Next layer outside config"))
    }
    fn select(&self, t: &TensorEntry) -> Result<Selection> {
        let role = t.role;
        if matches!(role.place, Place::Unused | Place::Draft { .. }) {
            return Ok(Selection::Absent);
        }
        let rows = |r| Selection::Rows(vec![r]);
        // Route ownership is independent of projection and storage, so packed
        // values, CSF exceptions/fixed bytes and both scalars always colocate.
        if let Part::Expert { id, .. } = role.part {
            let Ffn::Experts(e) = self.layer(role)?.ffn else {
                return Err(invalid("expert role without MoE"));
            };
            let id = id.ok_or_else(|| invalid("Next EP requires per-expert tensors"))? as u64;
            require(id < e.count, "expert outside config")?;
            return Ok(if self.part(e.count).contains(&id) {
                Selection::All
            } else {
                Selection::Absent
            });
        }
        if let Part::Lexical { part } = role.part {
            let l = self
                .shape
                .lexical
                .ok_or_else(|| invalid("lexical role without table"))?;
            require(
                l.parts > 0 && l.part_rows > 0 && u64::from(part) < l.parts,
                "invalid lexical part",
            )?;
            let total = l
                .parts
                .checked_mul(l.part_rows)
                .ok_or_else(|| invalid("lexical rows overflow"))?;
            let owned = self.part(total);
            let start = u64::from(part) * l.part_rows;
            let (from, to) = (owned.start.max(start), owned.end.min(start + l.part_rows));
            return Ok(if from < to {
                rows(from - start..to - start)
            } else {
                Selection::Absent
            });
        }
        let selection = match role.part {
            Part::Embedding | Part::Head => rows(self.vocabulary()),
            // Norms/index projection replicate, matching the existing FP8
            // indexer's complete reductions without per-step norm gathers.
            Part::Whole
            | Part::QueryNorm
            | Part::KeyNorm
            | Part::IndexNorm
            | Part::LinearNorm
            | Part::Index => Selection::All,
            Part::Shared(proj) => {
                let Ffn::Experts(e) = self.layer(role)?.ffn else {
                    return Err(invalid("shared role without MoE"));
                };
                require(
                    e.shared_width > 0 && e.shared_width % 32 == 0,
                    "shared width must align to MXFP8 groups",
                )?;
                projection(proj, scaled(self.part(e.shared_width / 32), 32))?
            }
            Part::Query | Part::Key | Part::Value | Part::Output => {
                let Mixer::Attention(a) = self.layer(role)?.mixer else {
                    return Err(invalid("attention role without attention"));
                };
                require(
                    a.kv_heads > 0
                        && a.query_heads > 0
                        && a.query_heads <= MAX_HEADS
                        && a.query_heads % a.kv_heads == 0
                        && a.head_dim > 0
                        && a.head_dim <= MAX_HEAD_DIM,
                    "invalid attention geometry",
                )?;
                let q = self.part(a.query_heads);
                let kv = grouped(q.clone(), a.query_heads / a.kv_heads);
                match role.part {
                    Part::Query => rows(scaled(q, a.head_dim * if a.gated { 2 } else { 1 })),
                    Part::Key | Part::Value => rows(scaled(kv, a.head_dim)),
                    _ => Selection::Columns(scaled(q, a.head_dim)),
                }
            }
            Part::LinearQkv
            | Part::LinearConv
            | Part::LinearZ
            | Part::LinearOut
            | Part::LinearHeads => {
                let Mixer::Linear(l) = self.layer(role)?.mixer else {
                    return Err(invalid("linear role without GDN"));
                };
                require(
                    l.key_heads > 0
                        && l.value_heads > 0
                        && l.value_heads <= MAX_HEADS
                        && l.value_heads % l.key_heads == 0
                        && l.key_dim > 0
                        && l.key_dim <= MAX_HEAD_DIM
                        && l.value_dim > 0
                        && l.value_dim <= MAX_HEAD_DIM,
                    "invalid GDN geometry",
                )?;
                let heads = self.part(l.value_heads);
                let k = scaled(
                    grouped(heads.clone(), l.value_heads / l.key_heads),
                    l.key_dim,
                );
                let v = scaled(heads.clone(), l.value_dim);
                let offset = l.key_heads * l.key_dim;
                match role.part {
                    Part::LinearHeads => rows(heads),
                    Part::LinearZ => rows(v),
                    Part::LinearOut => Selection::Columns(v),
                    _ => Selection::Rows(vec![
                        k.clone(),
                        k.start + offset..k.end + offset,
                        v.start + 2 * offset..v.end + 2 * offset,
                    ]),
                }
            }
            _ => return Err(invalid("unsupported Next tensor role")),
        };
        if role.store == Store::BlockScale && !matches!(role.part, Part::Whole) {
            require(
                t.shape.len() == 2 && t.dtype == safetensors::Dtype::U8,
                "expected MXFP8 UE8M0 scale matrix",
            )?;
            return Ok(match selection {
                Selection::Columns(r) => {
                    require(
                        r.start % 32 == 0 && r.end % 32 == 0,
                        "MXFP8 shard cuts a scale group",
                    )?;
                    Selection::Columns(r.start / 32..r.end / 32)
                }
                other => other,
            });
        }
        Ok(selection)
    }
}
fn projection(proj: Proj, r: Range<u64>) -> Result<Selection> {
    Ok(match proj {
        Proj::Gate | Proj::Up => Selection::Rows(vec![r]),
        Proj::Down => Selection::Columns(r),
        _ => return Err(invalid("unsupported fused Next projection")),
    })
}
fn validate_selection(s: &Selection, t: &Tensor) -> Result<()> {
    match s {
        Selection::Rows(ranges) => {
            let rows = t
                .shape
                .first()
                .ok_or_else(|| invalid("row selection of scalar"))?;
            let mut end = 0;
            for r in ranges {
                require(
                    r.start >= end && r.end >= r.start && r.end <= *rows,
                    "invalid Next row selection",
                )?;
                end = r.end;
            }
        }
        Selection::Columns(r) => require(
            t.shape.len() == 2 && r.start <= r.end && r.end <= t.shape[1],
            "invalid Next column selection",
        )?,
        _ => (),
    }
    Ok(())
}

/// Header binding owns open file descriptors, not weight payloads. Files must
/// remain immutable during a run, as with Checkpoint::read_range. This verifies
/// the map against actual headers/index, not a cryptographic payload digest.
pub struct ShardCheckpoint {
    pub plan: ShardPlan,
    files: BTreeMap<String, (File, Header)>,
    disk: BTreeMap<String, Vec<(u64, u64)>>,
    root: PathBuf,
}
impl ShardCheckpoint {
    pub fn open(root: &Path, tensor_map: &Path, rank: usize, world: usize) -> Result<Self> {
        Self::open_mode(root, tensor_map, rank, world, false)
    }
    pub fn open_mode(
        root: &Path,
        tensor_map: &Path,
        rank: usize,
        world: usize,
        hybrid: bool,
    ) -> Result<Self> {
        let metadata = mixed::Plan::open(&root.join("config.json"), tensor_map, 1)?;
        let mut plan = ShardPlan::new(&metadata.model.shape, &metadata.files, rank, world)?;
        if hybrid {
            plan = plan.ep2tp2()?;
        }
        Self::bind(root, plan)
    }
    /// Bind a reconciled typed plan. Public for host fixtures and metadata tools;
    /// production callers use open so ModelPlan validates the mixed inventory.
    pub fn bind(root: &Path, plan: ShardPlan) -> Result<Self> {
        let root = root.canonicalize()?;
        let limits = super::checkpoint_limits(super::WeightFormat::Nvfp4, true);
        let index = Index::parse_mixed(
            &safetensors::read_bounded(
                &root.join("model.safetensors.index.json"),
                limits.header_bytes,
            )?,
            limits,
        )?;
        require(
            index.weight_map.len() == plan.tensors.len(),
            "tensor-map/index count mismatch",
        )?;
        let mut files = BTreeMap::new();
        for (name, t) in &plan.tensors {
            require(
                index.weight_map.get(name) == Some(&t.file),
                "tensor-map/index file mismatch",
            )?;
            if !files.contains_key(&t.file) {
                safetensors::validate_model_path(&root, &t.file)?;
                let path = root.join(&t.file).canonicalize()?;
                require(path.starts_with(&root), "Next shard escapes checkpoint")?;
                let mut file = File::open(path)?;
                let header = Header::read(&mut file, limits)?;
                files.insert(t.file.clone(), (file, header));
            }
            let header = &files[&t.file].1;
            let actual = header
                .tensors
                .get(name)
                .ok_or_else(|| invalid("tensor-map name missing from shard header"))?;
            require(
                actual.dtype == t.tensor.dtype
                    && actual.shape == t.tensor.shape
                    && actual.byte_len() == t.tensor.byte_len(),
                "tensor-map/header dtype or shape mismatch",
            )?;
        }
        let actual = files.values().map(|(_, h)| h.tensors.len()).sum::<usize>();
        require(
            actual == plan.tensors.len(),
            "unexpected tensors in Next shard headers",
        )?;
        let disk = plan
            .tensors
            .iter()
            .filter(|(_, t)| t.placement == Placement::Disk)
            .map(|(name, t)| {
                let h = &files[&t.file].1;
                (
                    name.clone(),
                    t.selection
                        .spans(&t.tensor)
                        .map(|(start, bytes)| (h.data_start + h.tensors[name].start + start, bytes))
                        .collect(),
                )
            })
            .collect();
        Ok(Self {
            plan,
            files,
            root,
            disk,
        })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// File order must also be destination order for the shared GPU uploader.
    pub fn resident_order(&self) -> Vec<(&String, &TensorShard)> {
        let mut entries = self
            .plan
            .tensors
            .iter()
            .filter(|(_, t)| t.placement == Placement::Resident)
            .collect::<Vec<_>>();
        entries.sort_by_key(|(name, t)| (&t.file, self.files[&t.file].1.tensors[*name].start));
        entries
    }
    /// Already validated file and absolute tensor start for a bounded GPU loader.
    pub fn source(&self, name: &str) -> Result<(&File, u64)> {
        let t = self
            .plan
            .tensors
            .get(name)
            .ok_or_else(|| invalid("unknown Next tensor"))?;
        require(
            t.placement != Placement::Absent,
            "source outside owned Next shard",
        )?;
        let (file, header) = &self.files[&t.file];
        require(
            file.metadata()?.len() == header.file_bytes,
            "Next shard changed after inspection",
        )?;
        Ok((file, header.data_start + header.tensors[name].start))
    }
    /// Packed LOCAL coordinates; even disk reads cannot reach a peer's rows.
    pub fn read_local(&self, name: &str, offset: u64, output: &mut [u8]) -> Result<()> {
        let t = self
            .plan
            .tensors
            .get(name)
            .ok_or_else(|| invalid("unknown Next tensor"))?;
        require(
            t.placement != Placement::Absent
                && offset
                    .checked_add(output.len() as u64)
                    .is_some_and(|end| end <= t.bytes()),
            "read outside owned Next shard",
        )?;
        let (file, header) = &self.files[&t.file];
        // Disk descriptors and file ownership are fixed at bind, like prompt Reader.
        // Retain load-time validation; avoid stat and span allocation per lexical row.
        let temporary;
        let spans = if let Some(spans) = self.disk.get(name) {
            spans
        } else {
            require(
                file.metadata()?.len() == header.file_bytes,
                "Next shard changed after inspection",
            )?;
            temporary = t
                .selection
                .spans(&t.tensor)
                .map(|(start, bytes)| {
                    (
                        header.data_start + header.tensors[name].start + start,
                        bytes,
                    )
                })
                .collect::<Vec<_>>();
            &temporary
        };
        let mut skip = offset;
        let mut done = 0;
        for &(start, bytes) in spans {
            if skip >= bytes {
                skip -= bytes;
                continue;
            }
            let count = (bytes - skip).min((output.len() - done) as u64) as usize;
            file.read_exact_at(&mut output[done..done + count], start + skip)?;
            done += count;
            skip = 0;
            if done == output.len() {
                return Ok(());
            }
        }
        require(done == output.len(), "incomplete Next shard read")
    }
    /// Stream only resident selections, with bounded staging. The sink can
    /// upload directly to a rank's GPU arena; no full tensor/shard mirror exists.
    pub fn stream_resident(
        &self,
        staging_bytes: usize,
        mut sink: impl FnMut(&str, u64, &[u8]) -> Result<()>,
    ) -> Result<u64> {
        require(
            (1..=8 << 20).contains(&staging_bytes),
            "Next staging must be 1..=8 MiB",
        )?;
        let mut staging = vec![0; staging_bytes];
        let mut total = 0;
        for (name, t) in &self.plan.tensors {
            if t.placement != Placement::Resident {
                continue;
            }
            let mut offset = 0;
            while offset < t.bytes() {
                let count = (t.bytes() - offset).min(staging.len() as u64) as usize;
                self.read_local(name, offset, &mut staging[..count])?;
                sink(name, offset, &staging[..count])?;
                offset += count as u64;
            }
            total += offset;
        }
        Ok(total)
    }
}
