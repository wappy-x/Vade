//! First GPU decode path for the FP4-Local mixed checkpoint (qwen4_exp_text),
//! world 1, batch 1, greedy, mirroring the FP32 CPU reference (`refcpu`):
//! embeddings x4 streams, PLE at layer 1, HC mix -> attention -> combine,
//! HC mix -> MoE -> combine, final mixer + lm_head. Load-time options
//! default OFF (see next-fast.md).
//! Routed experts run the bit-exact W4A16 kernels
//! against CSF-expanded scales; full attention is dense FP32 behind a typed
//! indexer hook. Prompt tokens take the same single-token steps as decode.
#[path = "next_frontend.rs"]
pub(super) mod frontend;
use super::{
    Weights,
    blocks::{self, H, Hc, HcScratch, Ple, W},
    gdn::{self, Gdn, GdnScratch, GdnState},
    head::Head,
    next_native::{self as native, Matrix},
    qsa::{Qsa, QsaScratch, QsaState},
};
use crate::flash_next_projection::text_mxfp8;
use engine_model::{
    flash::{
        Checkpoint, PLE_PART_ROWS, PLE_PREFIX, PLE_ROWS, dump,
        mixed::{Plan, ScaleView, Storage},
        mxfp8, nvfp4,
        plan::RankPlan,
        refcpu::{self, PleHash},
    },
    plan::Format,
    safetensors::Dtype,
};
use spark_gpu_probe::{
    driver::{Backend, DeviceVector},
    kernel_params::{FlashBf16Pack, HybridCopy, flash_bf16_packed},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, String>;
pub(super) const LAYERS: usize = 48;
pub(super) const EXPERTS: usize = 512;
pub(super) const SCALE_BYTES: usize = 640 * 2560 / 16; // rows*cols/16 of every expert projection
pub(super) const EMBED: &str = "model.language_model.embed_tokens.weight";
pub(super) fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}
pub(super) fn read_all(cp: &Checkpoint, root: &Path, name: &str) -> Result<Vec<u8>> {
    let mut out = vec![0u8; cp.tensor(name).map_err(err)?.byte_len() as usize];
    cp.read_range(root, name, 0, &mut out).map_err(err)?;
    Ok(out)
}
pub(super) fn module_of(name: &str) -> Option<&str> {
    name.strip_suffix(".weight_scale")
        .or_else(|| name.strip_suffix(".weight"))
}
/// 64-row tiles per in-place pack step when the staging vector holds them. A pack of
/// consecutive 64-row tiles writes the same bytes as one pack per tile: every 64-row
/// block of the packed layout is contiguous and depends only on rows within the block.
pub(super) const PACK_TILES: usize = 16;
pub(super) fn pack_bf16(
    b: &Backend<'_>,
    weight: u64,
    rows: usize,
    cols: usize,
    staging: &DeviceVector<'_>,
) -> Result<()> {
    if !flash_bf16_packed(rows, cols) {
        return Err(format!("BF16 pack shape [{rows},{cols}] unsupported"));
    }
    let mut params = FlashBf16Pack {
        source: 0,
        weight,
        rows: rows as u32,
        cols: cols as u32,
    };
    if rows % 64 == 0 {
        let source = staging.device_address();
        let tiles = (staging.len() * 4 / (64 * cols * 2)).clamp(1, PACK_TILES);
        for row in (0..rows).step_by(64 * tiles) {
            let n = (rows - row).min(64 * tiles);
            let words = n * cols / 2;
            let destination = weight + (row * cols * 2) as u64;
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
                b.launch_flash_rows(
                    "flash_hybrid_copy",
                    words.div_ceil(128) as u32,
                    1,
                    128,
                    0,
                    &mut [(&mut copy as *mut HybridCopy).cast()],
                )?;
                b.launch_flash_shared(
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
            b.launch_flash_shared(
                "flash_bf16_pack",
                (rows / 4) as u32,
                128,
                (cols * 8) as u32,
                &mut [(&mut params as *mut FlashBf16Pack).cast()],
            )?;
        }
    }
    Ok(())
}
/// Sparse-indexer hook for full-attention layers (lane flashnext-indexer
/// owns the sparse selection itself). `Dense` attends every visible
/// position, matching `refcpu::DenseIndexer`; `Sparse` runs the QSA path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexSelection {
    Dense,
    Sparse,
}
pub trait IndexerHook {
    fn select(&self, layer: usize, pos: usize, kv_len: usize) -> IndexSelection;
}
pub struct DenseIndexer;
impl IndexerHook for DenseIndexer {
    fn select(&self, _l: usize, _p: usize, _n: usize) -> IndexSelection {
        IndexSelection::Dense
    }
}
pub struct SparseIndexer;
impl IndexerHook for SparseIndexer {
    fn select(&self, _l: usize, _p: usize, _n: usize) -> IndexSelection {
        IndexSelection::Sparse
    }
}
pub(super) struct MoeAddr {
    pub router: u64,
    pub shared_gate: u64,
    pub shared: [u64; 3],
    pub native: [Matrix; 3],
}
pub struct MemoryPlan {
    pub resident_weights: u64,
    pub expanded_bf16: u64,
    pub expanded_scales: u64,
    pub native_mxfp8: u64,
    pub kv: u64,
    pub recurrent: u64,
    pub total: u64,
}
/// Resident world-1 weights: checkpoint arena (BF16/plain tensors and packed
/// expert nibbles; PLE shards and CSF containers stay on NVMe), MXFP8
/// matrices native or expanded to packed BF16, and CSF scales expanded to E4M3 bytes.
pub struct NextModel<'a> {
    tables: Option<frontend::Tables>,
    root: PathBuf,
    checkpoint: Checkpoint,
    plan: Plan,
    pub(super) weights: Weights<'a>,
    expanded: BTreeMap<String, DeviceVector<'a>>,
    native: BTreeMap<String, native::Weight<'a>>,
    pub(super) fast: crate::flash_next_options::Options,
    pub(super) scales: Vec<DeviceVector<'a>>,
    pub(super) globals: Vec<f32>,
    gdn_native: Vec<[Matrix; 5]>,
    qsa_native: Vec<[Matrix; 5]>,
    gdn_proj: Vec<[u64; 5]>,
    qsa_proj: Vec<[u64; 5]>,
    pub(super) moe: Vec<MoeAddr>,
    ple_global: f32,
    hash: (Vec<i64>, Vec<i64>, Vec<i64>, i64),
}
impl<'a> NextModel<'a> {
    pub fn load(b: &Backend<'a>, root: &Path) -> Result<Self> {
        Self::load_with_options(b, root, crate::flash_next_options::Options::load()?)
    }
    pub fn load_with_options(
        b: &Backend<'a>,
        root: &Path,
        fast: crate::flash_next_options::Options,
    ) -> Result<Self> {
        let checkpoint = Checkpoint::inspect(root).map_err(err)?;
        if checkpoint.config.mixed.is_none() {
            return Err("flashnext GPU path requires the mixed FP4-Local checkpoint".into());
        }
        let plan = checkpoint.mixed_plan(1).map_err(err)?;
        let Format::Mixed(q) = &plan.model.format else {
            return Err("expected mixed storage plan".into());
        };
        let mxfp8 = |name: &str| module_of(name).is_some_and(|m| q.storage(m) == Storage::Mxfp8);
        let weights = Weights::load_selected(
            b,
            root,
            &checkpoint,
            RankPlan::new(0, 1).map_err(err)?,
            |name| {
                !name.starts_with("mtp.")
                    && !name.contains(".ngram_embedding.shard_")
                    && !name.contains(".nvfp4_csf_")
                    && !mxfp8(name)
            },
        )?;
        let staging = if fast.proj {
            None
        } else {
            Some(b.vector(64 * 6144 / 2)?)
        };
        let mut expanded = BTreeMap::new();
        let mut native = BTreeMap::new();
        for (name, t) in checkpoint.tensors() {
            let Some(module) = name.strip_suffix(".weight") else {
                continue;
            };
            if t.dtype != Dtype::F8E4M3 || !text_mxfp8(q, module) {
                continue;
            }
            if t.shape.len() != 2 {
                return Err(format!("{name}: MXFP8 target is not a matrix"));
            }
            let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
            let (w, s) = (
                read_all(&checkpoint, root, name)?,
                read_all(&checkpoint, root, &format!("{module}.weight_scale"))?,
            );
            if fast.proj {
                native.insert(
                    module.to_owned(),
                    native::Weight::load(b, &w, &s, rows, cols)?,
                );
                continue;
            }
            let mut bf = vec![0u16; rows * cols];
            mxfp8::expand_to_bf16(&w, &s, &mut bf).map_err(err)?;
            let mut v = b.vector(rows * cols / 2)?;
            b.write_vector_bytes(&mut v, unsafe {
                std::slice::from_raw_parts(bf.as_ptr().cast::<u8>(), bf.len() * 2)
            })?;
            pack_bf16(b, v.device_address(), rows, cols, staging.as_ref().unwrap())?;
            expanded.insert(module.to_owned(), v);
        }
        if expanded.len() + native.len() != 384 {
            return Err("expected 384 MXFP8 projections".into());
        }
        let (mut scales, mut globals) = (
            Vec::with_capacity(LAYERS),
            Vec::with_capacity(LAYERS * EXPERTS * 3),
        );
        for li in 0..LAYERS {
            let mut arena = vec![0u8; EXPERTS * 3 * SCALE_BYTES];
            for id in 0..EXPERTS {
                for (pi, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                    let m = format!("model.language_model.layers.{li}.mlp.experts.{id}.{proj}");
                    let (rows, cols) = if pi == 2 { (2560, 640) } else { (640, 2560) };
                    if weights.tensor(&format!("{m}.weight"))?.bytes != rows * cols / 2 {
                        return Err(format!("{m}: packed expert length disagree"));
                    }
                    let fixed = read_all(
                        &checkpoint,
                        root,
                        &format!("{m}.weight_scale.nvfp4_csf_fixed"),
                    )?;
                    let raw = read_all(
                        &checkpoint,
                        root,
                        &format!("{m}.weight_scale.nvfp4_csf_exceptions"),
                    )?;
                    if raw.len() % 4 != 0 {
                        return Err(format!("{m}: CSF exceptions are not U32 words"));
                    }
                    let exceptions: Vec<u32> = raw
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    let expanded_scales =
                        nvfp4::csf_expand(&fixed, &exceptions, rows, cols).map_err(err)?;
                    if expanded_scales.iter().any(|&s| s >= 127) {
                        return Err(format!(
                            "{m}: expanded scale is not a nonnegative finite E4M3 code"
                        ));
                    }
                    arena[(id * 3 + pi) * SCALE_BYTES..(id * 3 + pi + 1) * SCALE_BYTES]
                        .copy_from_slice(&expanded_scales);
                    let g = read_all(&checkpoint, root, &format!("{m}.weight_scale_2"))?;
                    let global = f32::from_le_bytes(
                        g.try_into().map_err(|_| format!("{m}: bad global scale"))?,
                    );
                    if !global.is_finite() || global <= 0.0 {
                        return Err(format!("{m}: invalid global scale"));
                    }
                    globals.push(global);
                }
            }
            let mut v = b.vector(arena.len() / 4)?;
            b.write_vector_bytes(&mut v, &arena)?;
            scales.push(v);
        }
        b.synchronize()?;
        let addr = |module: &str| {
            expanded
                .get(module)
                .map(|v| v.device_address())
                .or_else(|| native.get(module).map(|v| v.matrix.weights))
                .ok_or_else(|| format!("missing projection {module}"))
        };
        let project = |base: String, names: [&str; 5]| -> Result<[u64; 5]> {
            Ok(names
                .map(|n| addr(&format!("{base}.{n}")))
                .into_iter()
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .unwrap())
        };
        let mx = |base: String, names: [&str; 5]| -> Result<[Matrix; 5]> {
            if !fast.proj {
                return Ok([Matrix::default(); 5]);
            }
            Ok(names
                .map(|n| {
                    native
                        .get(&format!("{base}.{n}"))
                        .map(|v| v.matrix)
                        .ok_or("missing native projection".into())
                })
                .into_iter()
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .unwrap())
        };
        let (mut gdn_native, mut qsa_native) = (Vec::new(), Vec::new());
        let (mut gdn_proj, mut qsa_proj, mut moe) = (Vec::new(), Vec::new(), Vec::new());
        for li in 0..LAYERS {
            let p = format!("model.language_model.layers.{li}");
            gdn_proj.push(if li % 4 == 3 {
                [0; 5]
            } else {
                project(
                    format!("{p}.linear_attn"),
                    [
                        "in_proj_qkv",
                        "in_proj_z",
                        "in_proj_a",
                        "in_proj_b",
                        "out_proj",
                    ],
                )?
            });
            qsa_proj.push(if li % 4 != 3 {
                [0; 5]
            } else {
                project(
                    format!("{p}.self_attn"),
                    [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "indexer.index_qk_proj",
                        "o_proj",
                    ],
                )?
            });
            gdn_native.push(if li % 4 == 3 {
                [Matrix::default(); 5]
            } else {
                mx(
                    format!("{p}.linear_attn"),
                    [
                        "in_proj_qkv",
                        "in_proj_z",
                        "in_proj_a",
                        "in_proj_b",
                        "out_proj",
                    ],
                )?
            });
            qsa_native.push(if li % 4 != 3 {
                [Matrix::default(); 5]
            } else {
                mx(
                    format!("{p}.self_attn"),
                    [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "indexer.index_qk_proj",
                        "o_proj",
                    ],
                )?
            });
            moe.push(MoeAddr {
                native: mx(
                    format!("{p}.mlp.shared_expert"),
                    ["gate_proj", "up_proj", "down_proj", "gate_proj", "up_proj"],
                )?[..3]
                    .try_into()
                    .unwrap(),
                router: gdn::bf16(
                    &weights,
                    &format!("{p}.mlp.gate.weight"),
                    &[512, 2560],
                    EXPERTS * H,
                )?,
                shared_gate: gdn::bf16(
                    &weights,
                    &format!("{p}.mlp.shared_expert_gate.weight"),
                    &[1, 2560],
                    H,
                )?,
                shared: ["gate_proj", "up_proj", "down_proj"]
                    .map(|n| addr(&format!("{p}.mlp.shared_expert.{n}")))
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?
                    .try_into()
                    .unwrap(),
            });
        }
        let integers = |suffix: &str, n: usize| -> Result<Vec<i64>> {
            let bytes = read_all(&checkpoint, root, &format!("{PLE_PREFIX}.{suffix}"))?;
            if bytes.len() != n * 8 {
                return Err(format!("unexpected n-gram buffer {suffix}"));
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };
        let (mult, sizes, offsets) = (
            integers("layer_multipliers", 3)?,
            integers("ngram_heads_vocab_sizes", 16)?,
            integers("ngram_heads_offsets", 16)?,
        );
        if sizes
            .iter()
            .zip(&offsets)
            .any(|(&s, &o)| s <= 0 || o < 0 || o.checked_add(s).is_none_or(|e| e as u64 > PLE_ROWS))
        {
            return Err("invalid checkpoint n-gram head ranges".into());
        }
        let probe = plan.ple_row(0, 0).map_err(err)?;
        let g = read_all(&checkpoint, root, &probe.matrix.global.name)?;
        let ple_global = f32::from_le_bytes(g.try_into().map_err(|_| "bad PLE global scale")?);
        if !ple_global.is_finite() || ple_global <= 0.0 {
            return Err("invalid PLE global scale".into());
        }
        let tables = if fast.frontend {
            Some(frontend::Tables::load(b, &checkpoint, &plan, root)?)
        } else {
            None
        };
        Ok(Self {
            tables,
            root: root.to_owned(),
            checkpoint,
            plan,
            weights,
            expanded,
            native,
            fast,
            scales,
            globals,
            gdn_native,
            qsa_native,
            gdn_proj,
            qsa_proj,
            moe,
            ple_global,
            hash: (mult, sizes, offsets, 248044),
        })
    }
    /// Device-byte estimate at `capacity` (loaded weights; FP32 KV + index
    /// vectors and GDN state from the loader plan; session scratch excluded).
    pub fn memory_plan(&self, capacity: usize) -> Result<MemoryPlan> {
        let resident_weights = self.weights.resident_bytes() as u64;
        let expanded_bf16 = self.expanded.values().map(|v| v.len() as u64 * 4).sum();
        let native_mxfp8 = self
            .native
            .values()
            .map(|v| (v.weights.len() + v.scales.len()) as u64 * 4)
            .sum::<u64>();
        let expanded_scales = self.scales.iter().map(|v| v.len() as u64 * 4).sum();
        let (kv, recurrent) = self.plan.cache_bytes(capacity as u64, false).map_err(err)?;
        Ok(MemoryPlan {
            resident_weights,
            expanded_bf16,
            expanded_scales,
            native_mxfp8,
            kv,
            recurrent,
            total: resident_weights
                + expanded_bf16
                + expanded_scales
                + native_mxfp8
                + kv
                + recurrent,
        })
    }
    pub fn session<'w>(&'w self, b: &Backend<'a>, capacity: usize) -> Result<NextSession<'a, 'w>> {
        NextSession::new(b, self, capacity)
    }
}

#[derive(PartialEq, Eq)]
enum NextOutput {
    Prompt,
    Logits,
    Greedy,
}

/// One batch-1 sequence: FP32 residual streams, per-layer GDN/KV state and
/// PLE history. Every step runs the same kernels in the same order, so a
/// token produces identical bytes alone, after a prompt, or after a reset.
pub struct NextSession<'a, 'w> {
    model: &'w NextModel<'a>,
    capacity: usize,
    pos: usize,
    hash: PleHash,
    feed: Option<frontend::Feed<'a>>,
    attn_hc: Vec<Hc<'w>>,
    mlp_hc: Vec<Hc<'w>>,
    mixer: Hc<'w>,
    ple: Ple<'a, 'w>,
    head: Head<'a, 'w>,
    hc_scratch: HcScratch<'a>,
    gdn_scratch: GdnScratch<'a>,
    qsa_scratch: QsaScratch<'a>,
    gdn_state: Vec<Option<GdnState<'a>>>,
    qsa_state: Vec<Option<QsaState<'a>>>,
    gdn_norms: Vec<Option<DeviceVector<'a>>>,
    qsa_norms: Vec<Option<DeviceVector<'a>>>,
    residual: DeviceVector<'a>,
    block: DeviceVector<'a>,
    packed_in: DeviceVector<'a>,
    gate: DeviceVector<'a>,
    up: DeviceVector<'a>,
    mid: DeviceVector<'a>,
    tmp: DeviceVector<'a>,
    logits: DeviceVector<'a>,
    shared_gate: DeviceVector<'a>,
    lexical: DeviceVector<'a>,
    indexer: Box<dyn IndexerHook>,
    dump: Option<Vec<String>>,
}
impl<'a, 'w> NextSession<'a, 'w> {
    fn new(b: &Backend<'a>, model: &'w NextModel<'a>, capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > 262144 {
            return Err("flashnext capacity must be 1..=262144".into());
        }
        let w = &model.weights;
        let hc = |role: &str| {
            (0..LAYERS)
                .map(|li| {
                    Hc::load_next(
                        w,
                        &format!("model.language_model.layers.{li}.{role}_hyper_connection"),
                    )
                })
                .collect::<Result<Vec<_>>>()
        };
        let (attn_hc, mlp_hc) = (hc("attn")?, hc("mlp")?);
        let [gdn_norms, qsa_norms] =
            crate::flash_next_storage::session_norms(b, |name| w.values_f32(name))?;
        let mut v = b
            .vector_arena(&[W, H, H / 2, 640, 640, 320, H, EXPERTS, 1, H])?
            .into_iter();
        let (mult, sizes, offsets, eos) = &model.hash;
        Ok(Self {
            model,
            capacity,
            pos: 0,
            feed: if model.fast.frontend {
                Some(frontend::Feed::new(b, model)?)
            } else {
                None
            },
            hash: PleHash::new(mult.clone(), sizes.clone(), offsets.clone(), *eos),
            attn_hc,
            mlp_hc,
            mixer: Hc::mixer_next(w, "model.language_model.hyper_connection_mixer")?,
            ple: Ple::load(b, w)?,
            head: Head::load(b, w)?,
            hc_scratch: HcScratch::new(b)?,
            gdn_scratch: GdnScratch::new(b, w, 1, &[0.0; 128])?,
            qsa_scratch: QsaScratch::new(b, &w.plan, 1, capacity)?,
            gdn_state: (0..LAYERS).map(|_| None).collect(),
            qsa_state: (0..LAYERS).map(|_| None).collect(),
            gdn_norms,
            qsa_norms,
            residual: v.next().unwrap(),
            block: v.next().unwrap(),
            packed_in: v.next().unwrap(),
            gate: v.next().unwrap(),
            up: v.next().unwrap(),
            mid: v.next().unwrap(),
            tmp: v.next().unwrap(),
            logits: v.next().unwrap(),
            shared_gate: v.next().unwrap(),
            lexical: v.next().unwrap(),
            indexer: Box::new(DenseIndexer),
            dump: None,
        })
    }
    pub fn position(&self) -> usize {
        self.pos
    }
    /// Full next-token distribution for serving and GPU byte gates.
    pub fn logits(&self, b: &Backend<'a>) -> Result<Vec<f32>> {
        self.head.logits(b)
    }
    /// Allocate every layer's state before admitting any request.
    pub fn prepare(&mut self, b: &Backend<'a>) -> Result<()> {
        for li in 0..LAYERS {
            if li % 4 == 3 {
                self.qsa_state[li] = Some(
                    Qsa::load_next(
                        b,
                        &self.model.weights,
                        li,
                        self.capacity,
                        self.model.qsa_proj[li],
                        self.qsa_state[li].take(),
                    )?
                    .into_state(),
                );
            } else {
                self.gdn_state[li] = Some(
                    Gdn::load_next(
                        b,
                        &self.model.weights,
                        li,
                        self.model.gdn_proj[li],
                        self.gdn_state[li].take(),
                    )?
                    .into_state(),
                );
            }
        }
        Ok(())
    }
    pub fn set_indexer(&mut self, indexer: Box<dyn IndexerHook>) {
        self.indexer = indexer;
    }
    pub fn enable_dump(&mut self) {
        self.dump = Some(vec!["VADEDBG1 flashnext-gpu world=1 batch=1".into()]);
    }
    pub fn take_dump(&mut self) -> Option<String> {
        self.dump.take().map(|lines| lines.join("\n") + "\n")
    }
    pub fn reset(&mut self, b: &Backend<'a>) -> Result<()> {
        if let Some(feed) = &mut self.feed {
            feed.reset(b)?;
        }
        self.pos = 0;
        self.hash.reset();
        self.ple.reset(b)?;
        for state in self.gdn_state.iter_mut().flatten() {
            state.reset(b)?;
        }
        for state in self.qsa_state.iter_mut().flatten() {
            state.rewind(0);
        }
        Ok(())
    }
    fn rec(&mut self, stage: &str, values: &[f32]) {
        if let Some(d) = &mut self.dump {
            d.push(dump::rec_line(stage, values));
        }
    }
    fn ple_embedding(&self, rows: &[u64]) -> Result<Vec<f32>> {
        let mut e = vec![0.0; H];
        for (h, &row) in rows.iter().enumerate() {
            let pr = self
                .model
                .plan
                .ple_row((row / PLE_PART_ROWS) as u32, row % PLE_PART_ROWS)
                .map_err(err)?;
            let mut packed = vec![0u8; pr.packed_bytes as usize];
            self.model
                .checkpoint
                .read_range(
                    &self.model.root,
                    &pr.matrix.weight.name,
                    pr.packed_offset,
                    &mut packed,
                )
                .map_err(err)?;
            let ScaleView::E4m3(t) = &pr.matrix.scales else {
                return Err("PLE rows must use direct E4M3 scales".into());
            };
            let mut scales = vec![0u8; pr.scale_bytes as usize];
            self.model
                .checkpoint
                .read_range(&self.model.root, &t.name, pr.scale_offset, &mut scales)
                .map_err(err)?;
            refcpu::ple_row_nvfp4(
                &packed,
                &scales,
                self.model.ple_global,
                &mut e[h * 160..(h + 1) * 160],
            )
            .map_err(err)?;
        }
        Ok(e)
    }
    fn axpy(&self, b: &Backend<'_>, scale: f32) -> Result<()> {
        let (mut y, mut x, mut s, mut n) = (
            self.block.device_address(),
            self.tmp.device_address(),
            scale,
            H as u32,
        );
        unsafe {
            b.launch_flash(
                "flash_next_axpy",
                (H / 128) as u32,
                128,
                &mut [
                    (&mut y as *mut u64).cast(),
                    (&mut x as *mut u64).cast(),
                    (&mut s as *mut f32).cast(),
                    (&mut n as *mut u32).cast(),
                ],
            )
        }
    }
    fn moe(&mut self, b: &Backend<'a>, li: usize) -> Result<()> {
        use gdn::Arg::{Ptr, U32};
        let (model, u) = (self.model, self.hc_scratch.mixed().device_address());
        let addr = &model.moe[li];
        blocks::bf16_rows(
            b,
            addr.router,
            u,
            self.logits.device_address(),
            EXPERTS,
            H,
            1,
            EXPERTS,
        )?;
        let logits = if li == 0 {
            if let Some(f) = &mut self.feed {
                f.activation(b, &self.logits)?
            } else {
                b.read_vector(&self.logits)?
            }
        } else {
            b.read_vector(&self.logits)?
        };
        let (ids, weights) = refcpu::route(&logits, 10);
        let mut pairs: Vec<(usize, f32)> = ids.into_iter().zip(weights).collect();
        pairs.sort_by_key(|&(i, _)| i);
        if let Some(d) = &mut self.dump {
            d.push(dump::moe_line(li, &pairs));
        }
        let host = if li == 0 {
            if let Some(f) = &mut self.feed {
                f.activation(b, self.hc_scratch.mixed())?
            } else {
                b.read_vector(self.hc_scratch.mixed())?
            }
        } else {
            b.read_vector(self.hc_scratch.mixed())?
        };
        let packed: Vec<f32> = host
            .chunks_exact(2)
            .map(|p| {
                f32::from_bits(
                    nvfp4::f32_to_bf16(p[0]) as u32 | ((nvfp4::f32_to_bf16(p[1]) as u32) << 16),
                )
            })
            .collect();
        b.write_vector(&mut self.packed_in, &packed)?;
        b.zero_vector(&mut self.block)?;
        for &(id, wt) in &pairs {
            let base = format!("model.language_model.layers.{li}.mlp.experts.{id}");
            for pi in 0..3 {
                if pi == 2 {
                    unsafe {
                        b.fp4_exact_swiglu(&self.gate, &self.up, 320, &mut self.mid)?;
                    }
                }
                let (input, rows, cols): (&DeviceVector, usize, usize) = if pi == 2 {
                    (&self.mid, H, 640)
                } else {
                    (&self.packed_in, 640, H)
                };
                let out = if pi == 0 {
                    &mut self.gate
                } else if pi == 1 {
                    &mut self.up
                } else {
                    &mut self.tmp
                };
                let m = format!("{base}.{}", ["gate_proj", "up_proj", "down_proj"][pi]);
                unsafe {
                    b.fp4_exact_project_addr(
                        input,
                        model.weights.device_address(&format!("{m}.weight"))?,
                        rows * cols / 2,
                        model.scales[li].device_address() + ((id * 3 + pi) * SCALE_BYTES) as u64,
                        SCALE_BYTES,
                        model.globals[(li * EXPERTS + id) * 3 + pi],
                        1,
                        rows,
                        cols,
                        out,
                    )?;
                }
            }
            self.axpy(b, wt)?;
        }
        blocks::bf16_rows(
            b,
            addr.shared_gate,
            u,
            self.shared_gate.device_address(),
            1,
            H,
            1,
            1,
        )?;
        if model.fast.proj {
            native::project(
                b,
                &[
                    (addr.native[0], u, self.gate.device_address()),
                    (addr.native[1], u, self.up.device_address()),
                ],
                1,
                false,
                false,
            )?;
        } else {
            blocks::bf16_rows(
                b,
                addr.shared[0],
                u,
                self.gate.device_address(),
                640,
                H,
                1,
                640,
            )?;
            blocks::bf16_rows(
                b,
                addr.shared[1],
                u,
                self.up.device_address(),
                640,
                H,
                1,
                640,
            )?;
        }
        gdn::launch(
            b,
            "flash_moe_silu",
            640,
            &mut [
                Ptr(self.gate.device_address()),
                Ptr(self.up.device_address()),
                U32(640),
            ],
        )?;
        if model.fast.proj {
            native::project(
                b,
                &[(
                    addr.native[2],
                    self.gate.device_address(),
                    self.tmp.device_address(),
                )],
                1,
                false,
                false,
            )?;
        } else {
            blocks::bf16_rows(
                b,
                addr.shared[2],
                self.gate.device_address(),
                self.tmp.device_address(),
                H,
                640,
                1,
                H,
            )?;
        }
        let gate = if li == 0 {
            if let Some(f) = &mut self.feed {
                f.activation(b, &self.shared_gate)?
            } else {
                b.read_vector(&self.shared_gate)?
            }
        } else {
            b.read_vector(&self.shared_gate)?
        };
        self.axpy(b, refcpu::sigmoid(gate[0]))
    }
    /// Process one token at the current position; returns the greedy next id.
    pub fn step(&mut self, b: &Backend<'a>, token: u32) -> Result<u32> {
        if self.feed.is_some() {
            self.forward(b, Some(token), false, NextOutput::Greedy)?;
            return self.feed.as_mut().unwrap().read(b, self.model);
        }
        self.forward(b, Some(token), false, NextOutput::Greedy)?
            .ok_or("missing next token".into())
    }
    pub fn step_logits(&mut self, b: &Backend<'a>, token: u32) -> Result<Vec<f32>> {
        self.forward(b, Some(token), false, NextOutput::Logits)?;
        self.head.logits(b)
    }
    pub fn step_top_k(&mut self, b: &Backend<'a>, token: u32, k: usize) -> Result<Vec<(u32, f32)>> {
        if !(1..=64).contains(&k) {
            return Err("flashnext top_k must be 1..=64".into());
        }
        self.forward(b, Some(token), false, NextOutput::Logits)?;
        self.head.top_k(b, k)
    }
    /// Commit the same recurrent/KV/PLE state while skipping vocabulary work.
    pub fn step_prompt(&mut self, b: &Backend<'a>, token: u32) -> Result<()> {
        self.forward(b, Some(token), false, NextOutput::Prompt)
            .map(|_| ())
    }
    fn forward(
        &mut self,
        b: &Backend<'a>,
        token: Option<u32>,
        late: bool,
        output: NextOutput,
    ) -> Result<Option<u32>> {
        if self.pos >= self.capacity {
            return Err("flashnext context capacity reached; reset the session".into());
        }
        if token.is_some_and(|t| t as u64 >= self.model.checkpoint.config.vocab) {
            return Err("token outside vocabulary".into());
        }
        let mut previous = None;
        let rows = if let Some(feed) = &mut self.feed {
            feed.begin(b, self.model, &self.residual, token)?;
            if late {
                previous = Some(feed.read(b, self.model)?);
            }
            feed.gather(b, self.model, &self.lexical)?;
            Vec::new()
        } else {
            let token = token.ok_or("host frontend requires a token")?;
            let rows = self.hash.advance(token);
            let embedding = gdn::bf16(
                &self.model.weights,
                EMBED,
                &[self.model.checkpoint.config.vocab, H as u64],
                self.model.checkpoint.config.vocab as usize * H,
            )?;
            gdn::launch(
                b,
                "flash_embedding_expand",
                W,
                &mut [
                    gdn::Arg::Ptr(embedding + token as u64 * (H * 2) as u64),
                    gdn::Arg::Ptr(self.residual.device_address()),
                    gdn::Arg::U32(H as u32),
                ],
            )?;
            rows
        };
        if let Some(d) = &mut self.dump {
            d.push(format!(
                "token {} {}",
                self.pos,
                token.or(previous).ok_or("missing input token")?
            ));
        }
        if self.dump.is_some() {
            let x = b.read_vector(&self.residual)?;
            self.rec("embed", &x);
        }
        for li in 0..LAYERS {
            if li == 1 {
                if let Some(feed) = &self.feed {
                    feed.join(b)?;
                    if self.dump.is_some() {
                        let e = b.read_vector(&self.lexical)?;
                        self.rec("layer1.ple_e", &e);
                    }
                } else {
                    let e = self.ple_embedding(&rows)?;
                    self.rec("layer1.ple_e", &e);
                    b.write_vector(&mut self.lexical, &e)?;
                }
                self.ple.project(b, &mut self.residual, &self.lexical)?;
                if self.dump.is_some() {
                    let v = b.read_vector(&self.residual)?;
                    self.rec("layer1.ple", &v);
                }
            }
            self.attn_hc[li].mix(b, &mut self.hc_scratch, &self.residual)?;
            if li % 4 == 3 {
                self.qsa_scratch
                    .use_norm(self.qsa_norms[li].as_ref().unwrap())?;
                let mut qsa = Qsa::load_next(
                    b,
                    &self.model.weights,
                    li,
                    self.capacity,
                    self.model.qsa_proj[li],
                    self.qsa_state[li].take(),
                )?;
                if self.model.fast.proj {
                    qsa.use_native(self.model.qsa_native[li])?;
                }
                qsa.project(b, &mut self.qsa_scratch, self.hc_scratch.mixed(), 1)?;
                (if self.indexer.select(li, self.pos, self.pos + 1) == IndexSelection::Dense {
                    qsa.attend_dense(b, &mut self.qsa_scratch)
                } else {
                    qsa.attend(b, &mut self.qsa_scratch)
                })?;
                b.copy_range(self.qsa_scratch.output(), &mut self.block, 0, 0, H)?;
                self.qsa_state[li] = Some(qsa.into_state());
            } else {
                self.gdn_scratch
                    .use_norm(self.gdn_norms[li].as_ref().unwrap())?;
                let mut gdn = Gdn::load_next(
                    b,
                    &self.model.weights,
                    li,
                    self.model.gdn_proj[li],
                    self.gdn_state[li].take(),
                )?;
                if self.model.fast.proj {
                    gdn.run_next_native(
                        b,
                        &mut self.gdn_scratch,
                        self.hc_scratch.mixed(),
                        self.model.gdn_native[li],
                    )?;
                } else {
                    gdn.run(b, &mut self.gdn_scratch, self.hc_scratch.mixed(), 1)?;
                }
                b.copy_range(self.gdn_scratch.output(), &mut self.block, 0, 0, H)?;
                self.gdn_state[li] = Some(gdn.into_state());
            }
            if self.dump.is_some() {
                let v = b.read_vector(&self.block)?;
                self.rec(&format!("layer{li}.attn"), &v);
            }
            self.attn_hc[li].combine(b, &mut self.hc_scratch, &mut self.residual, &self.block)?;
            self.mlp_hc[li].mix(b, &mut self.hc_scratch, &self.residual)?;
            self.moe(b, li)?;
            self.mlp_hc[li].combine(b, &mut self.hc_scratch, &mut self.residual, &self.block)?;
            if self.dump.is_some() {
                let v = b.read_vector(&self.residual)?;
                self.rec(&format!("layer{li}.x"), &v);
            }
        }
        if output != NextOutput::Prompt {
            self.mixer.mix(b, &mut self.hc_scratch, &self.residual)?;
            if self.dump.is_some() {
                let v = b.read_vector(self.hc_scratch.mixed())?;
                self.rec("final", &v);
            }
            self.head.project(b, self.hc_scratch.mixed())?;
            if self.dump.is_some() {
                let line = dump::top_line(&self.head.logits(b)?);
                self.dump.as_mut().unwrap().push(line);
            }
        }
        let next = if output != NextOutput::Greedy {
            None
        } else if let Some(feed) = &mut self.feed {
            self.head.enqueue_max(b, &mut feed.ids, 0)?;
            feed.publish(b)?;
            previous
        } else {
            Some(self.head.local_max(b)?.ok_or("empty vocabulary")?.token)
        };
        self.pos += 1;
        Ok(next)
    }
    /// Feed the prompt token by token, then greedily generate `new_tokens` ids.
    pub fn generate(
        &mut self,
        b: &Backend<'a>,
        prompt: &[u32],
        new_tokens: usize,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err("flashnext prompt must not be empty".into());
        }
        if new_tokens == 0 {
            return Ok(Vec::new());
        }
        if self.feed.is_some() {
            for &t in &prompt[..prompt.len() - 1] {
                self.step(b, t)?;
            }
            self.forward(b, prompt.last().copied(), false, NextOutput::Greedy)?;
            let mut out = Vec::with_capacity(new_tokens);
            while out.len() + 1 < new_tokens {
                out.push(
                    self.forward(b, None, true, NextOutput::Greedy)?
                        .ok_or("missing delayed token")?,
                );
            }
            out.push(self.feed.as_mut().unwrap().read(b, self.model)?);
            return Ok(out);
        }
        let mut next = 0;
        for &t in prompt {
            next = self.step(b, t)?;
        }
        let mut out = vec![next];
        while out.len() < new_tokens {
            next = self.step(b, next)?;
            out.push(next);
        }
        Ok(out)
    }
}
