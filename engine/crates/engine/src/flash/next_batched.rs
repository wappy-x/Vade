//! First GPU decode path for the FP4-Local mixed checkpoint (qwen4_exp_text),
//! World 1 prompt tiles plus world 2/4 native rank-row adapters:
//! embeddings x4 streams, PLE at layer 1, HC mix -> attention -> combine,
//! HC mix -> MoE -> combine, final mixer + lm_head. MXFP8 weights expand
//! exactly to BF16 at load; routed experts use device routing and grouped
//! K16 BF16 operands against CSF-expanded scales. Sparse QSA is the default.
//! All row counts share one arithmetic schedule; decode replays slot graphs.
use super::{
    Weights,
    blocks::{self, H, Hc, HcScratch, Ple, W},
    gdn::{self, Gdn, GdnScratch, GdnState},
    head::Head,
    qsa::{Qsa, QsaScratch, QsaState},
};
use crate::flash_next_projection::text_mxfp8;
use engine_model::{
    flash::{
        Checkpoint, PLE_PREFIX, PLE_ROWS, dump,
        mixed::{Plan, Storage},
        mxfp8, nvfp4,
        plan::RankPlan,
        refcpu::PleHash,
    },
    plan::Format,
    safetensors::Dtype,
};
use spark_gpu_probe::{
    driver::{Backend, DeviceVector},
    kernel_params::{FlashBf16Pack, HybridCopy},
};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, String>;
pub use super::next::{DenseIndexer, IndexSelection, IndexerHook, SparseIndexer};
use super::next::{EMBED, EXPERTS, LAYERS, SCALE_BYTES, err, module_of, pack_bf16, read_all};
struct MoeAddr<'a> {
    router: u64,
    shared_gate: u64,
    shared: [u64; 3],
    experts: DeviceVector<'a>,
}
pub struct MemoryPlan {
    pub resident_weights: u64,
    pub expanded_bf16: u64,
    pub expanded_scales: u64,
    pub kv: u64,
    pub recurrent: u64,
    pub total: u64,
}
/// Native slot count shared with the HTTP admission cap.
pub const POOL_SIZE: usize = 16;

/// Resident world-1 weights: checkpoint arena (BF16/plain tensors and packed
/// expert nibbles; PLE shards and CSF containers stay on NVMe), MXFP8
/// matrices expanded to packed BF16, and CSF scales expanded to E4M3 bytes.
pub struct NextModel<'a> {
    mtp_depth: usize,
    draft_moe: Option<MoeAddr<'a>>,
    draft_head_rows: usize,
    draft_head_mx8: Option<DeviceVector<'a>>,
    /// Drafter-only NVFP4 head image (Head::load_fp4_draft), VADE_FLASHNEXT_DRAFT_FP4=1|2 (unset = 2).
    draft_head_fp4: Option<DeviceVector<'a>>,
    /// VADE_FLASHNEXT_DRAFT_MX8_LAYER (unset = on): MXFP8 twins of the MTP layer's BF16 fc and
    /// attention matrices (blocks::load_mx8_twin), read by one-row draft launches.
    pub(super) draft_layer_mx8: Vec<blocks::Mx8Twin<'a>>,
    /// LBF12 tile flags + row base exponents of the in-place head (Head::encode_lbf12).
    head_lbf12: Option<DeviceVector<'a>>,
    head_bounds: Option<super::head::Bounds<'a>>,
    /// LBF12 twins of the HC up weights (blocks::lbf12_twin).
    lbf12_twins: Vec<DeviceVector<'a>>,
    frontend_host: std::cell::Cell<(u64, std::time::Duration)>,
    prefill_reader: crate::flash_next_prefill::Reader,
    /// Set false by a pipeline back before admission or workspace allocation.
    prompt_frontend: bool,

    checkpoint: Checkpoint,
    plan: Plan,
    weights: Weights<'a>,
    expanded: BTreeMap<String, DeviceVector<'a>>,
    /// MXFP8 twins (as stored) of expanded matrices, registered by BF16 address for
    /// the M <= 16 latency launches; VADE_FLASHNEXT_MX8=0 keeps decode on BF16.
    /// Each twin owns its registry entry (dropped with the model or a failed load).
    mx8: Vec<blocks::Mx8Twin<'a>>,
    scales: Vec<DeviceVector<'a>>,
    gdn_proj: Vec<[u64; 5]>,
    qsa_proj: Vec<[u64; 5]>,
    moe: Vec<MoeAddr<'a>>,
    pool: RefCell<execution::Pool<'a>>,
    /// The one prompt / verify / replay workspace (prompt-tile rows), funded once in
    /// shared_reservation_bytes; cohorts run one at a time, so it is never shared live.
    work: RefCell<Option<Workspace<'a>>>,
    gdn_norms: Vec<Option<DeviceVector<'a>>>,
    qsa_norms: Vec<Option<DeviceVector<'a>>>,
    ple_global: f32,
    hash: (Vec<i64>, Vec<i64>, Vec<i64>, i64),
    /// Pipeline front link (rank 0); None everywhere else (world 1 is unchanged).
    pp: RefCell<Option<pp::Link>>, tp: RefCell<Option<tp::Swap<'a>>>, tp_box: std::cell::OnceCell<spark_gpu_probe::driver::TpMailbox<'a>>,
    /// Pipeline first back layer on both ranks; 0 = no pipeline.
    pp_split: std::cell::Cell<usize>,
    /// Exclusive last layer owned by a pipeline worker.
    pp_end: std::cell::Cell<usize>,
    /// Layers whose session GDN/QSA state this rank allocates: [0, 48), or a pipeline
    /// stage's own interval after own_stage_state; `state_charge`: the intervals funded.
    state_layers: std::cell::Cell<(usize, usize)>,
    state_charge: RefCell<Vec<std::ops::Range<usize>>>,
}
impl<'a> NextModel<'a> {
    /// A pipeline rank allocates
    /// session GDN/QSA state for its own layers [first, end) only; the other stages' layers
    /// never run on it. Each chat is funded at the costliest of `stages` (every stage's
    /// interval), so every rank admits the same count from the same budget, as with whole
    /// state. Call before admission, while no session exists.
    pub fn own_stage_state(&self, first: usize, end: usize, stages: &[std::ops::Range<usize>]) -> Result<()> {
        crate::flash_next_rows::check_stage_charge(first..end, stages)?;
        if self.pool.borrow().used[1..].contains(&true) {
            return Err("pipeline stage state must be set before any session exists".into());
        }
        self.state_layers.set((first, end));
        *self.state_charge.borrow_mut() = stages.to_vec();
        Ok(())
    }
    /// Layers whose session state this rank owns (all 48 unless own_stage_state ran).
    pub(super) fn state_range(&self) -> std::ops::Range<usize> {
        let (first, end) = self.state_layers.get();
        first..end
    }
    pub(super) fn owns_state(&self, li: usize) -> bool {
        self.state_range().contains(&li)
    }
    /// Pipeline backs receive the residual and never expand token inputs. Call before
    /// admission and workspace preparation; changing a live model is refused.
    pub fn disable_prompt_frontend(&mut self) -> Result<()> {
        if self.work.borrow().is_some() || !self.pool.borrow().decode.is_empty() {
            return Err("pipeline frontend role must be set before workspace allocation".into());
        }
        self.prompt_frontend = false;
        Ok(())
    }
    pub fn load(b: &Backend<'a>, root: &Path) -> Result<Self> {
        Self::load_with_options(b, root, crate::flash_next_options::Options::load()?)
    }
    pub fn load_with_options(
        b: &Backend<'a>,
        root: &Path,
        options: crate::flash_next_options::Options,
    ) -> Result<Self> {
        Self::load_with_mtp(b, root, options, 2)
    }
    pub fn load_with_mtp(
        b: &Backend<'a>,
        root: &Path,
        options: crate::flash_next_options::Options,
        mtp_depth: usize,
    ) -> Result<Self> {
        Self::load_with_checkpoint(b, root, None, options, mtp_depth)
    }
    /// `load_with_mtp` reusing the caller's inspection of `root` (one header parse per start).
    pub fn load_with_checkpoint(
        b: &Backend<'a>,
        root: &Path,
        checkpoint: Option<Checkpoint>,
        options: crate::flash_next_options::Options,
        mtp_depth: usize,
    ) -> Result<Self> {
        Self::load_with_checkpoint_role(b, root, checkpoint, options, mtp_depth, false)
    }
    /// `headless`: a pipeline rank without the head (back, middle ranks) does not execute the
    /// local bounded head and must not fund its twin; the front funds it as one Spark does.
    pub fn load_with_checkpoint_role(
        b: &Backend<'a>, root: &Path, checkpoint: Option<Checkpoint>,
        options: crate::flash_next_options::Options, mtp_depth: usize, headless: bool,
    ) -> Result<Self> {
        if mtp_depth > 15 {
            return Err("MTP depth must be 0..=15".into());
        }
        options.validate_rows()?;
        crate::flash_prompt_options::load()?;
        crate::flash_head_options::load()?;
        super::qsa::qsa_tc_switch()?;
        // The drafting stack's switches (crate::flash_draft_stack): refuse a malformed value
        // before allocation and state the effective settings in the startup log.
        eprintln!("flashnext {} mtp_depth={mtp_depth}", crate::flash_draft_stack::describe()?);
        // Load phase seconds for the startup log; timing only.
        let started = std::time::Instant::now();
        let mut phase = [std::time::Duration::ZERO; 12];
        let mut mark = std::time::Instant::now();
        let mut lap = |slot: usize| { let now = std::time::Instant::now(); phase[slot] += now - mark; mark = now; };

        let checkpoint = match checkpoint {
            Some(checkpoint) => checkpoint,
            None => Checkpoint::inspect(root).map_err(err)?,
        };
        if checkpoint.config.mixed.is_none() {
            return Err("flashnext GPU path requires the mixed FP4-Local checkpoint".into());
        }
        let plan = checkpoint.mixed_plan(1).map_err(err)?;
        let Format::Mixed(q) = &plan.model.format else {
            return Err("expected mixed storage plan".into());
        };
        let mxfp8 = |name: &str| module_of(name).is_some_and(|m| q.storage(m) == Storage::Mxfp8);
        let mut rank = RankPlan::new(0, 1).map_err(err)?;
        rank.mtp = mtp_depth != 0;
        let mx8_on = std::env::var("VADE_FLASHNEXT_MX8").map_or(true, |v| v != "0");
        let mut arena = b.stats();
        let before = arena;
        // MXFP8 and CSF reads + host expansion run on load workers while this thread
        // streams the weight arena; results arrive in their serial order through bounded
        // queues, and every allocation, upload and launch keeps its serial order. Same bytes.
        let shards = super::next_load::Shards::open(&checkpoint, root)?;
        let mut targets = Vec::new();
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
            targets.push((name, module, t.shape[0] as usize, t.shape[1] as usize));
        }
        const PROJ: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
        let shape = |pi: usize| if pi == 2 { (2560, 640) } else { (640, 2560) };
        let window = 2 * super::next_load::threads();
        let (weights, draft_moe, staging, expanded, mx8, scales) = std::thread::scope(|scope| -> Result<_> {
            // Up to ~6 GB of host results wait here while the arena streams (~6 s at 12 GB/s).
            let (mx8_send, mx8_ready) = std::sync::mpsc::sync_channel(4 * window);
            let (csf_send, csf_ready) = std::sync::mpsc::sync_channel(16);
            let (shards, targets) = (&shards, &targets);
            scope.spawn(move || {
                for chunk in targets.chunks(window) {
                    let hosts = super::next_load::parallel_map(chunk, |&(name, module, rows, cols)| -> Result<_> {
                        let (w, s) = (shards.read(name)?, shards.read(&format!("{module}.weight_scale"))?);
                        let mut bf = vec![0u16; rows * cols];
                        mxfp8::expand_to_bf16(&w, &s, &mut bf).map_err(err)?;
                        Ok((bf, mx8_on.then(|| blocks::mx8_image(&w, &s, rows, cols)).flatten()))
                    });
                    for host in hosts {
                        if mx8_send.send(host).is_err() { return; }
                    }
                }
            });
            scope.spawn(move || {
                for li in 0..LAYERS {
                    let mut arena = vec![0u8; EXPERTS * 3 * SCALE_BYTES];
                    let done = super::next_load::parallel_chunks(&mut arena, 3 * SCALE_BYTES, |id, out| -> Result<()> {
                        for (pi, proj) in PROJ.iter().enumerate() {
                            let m = format!("model.language_model.layers.{li}.mlp.experts.{id}.{proj}");
                            let (rows, cols) = shape(pi);
                            let fixed = shards.read(&format!("{m}.weight_scale.nvfp4_csf_fixed"))?;
                            let raw = shards.read(&format!("{m}.weight_scale.nvfp4_csf_exceptions"))?;
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
                            out[pi * SCALE_BYTES..(pi + 1) * SCALE_BYTES].copy_from_slice(&expanded_scales);
                            let g = shards.read(&format!("{m}.weight_scale_2"))?;
                            let global = f32::from_le_bytes(
                                g.try_into().map_err(|_| format!("{m}: bad global scale"))?,
                            );
                            if !global.is_finite() || global <= 0.0 {
                                return Err(format!("{m}: invalid global scale"));
                            }
                        }
                        Ok(())
                    });
                    let layer = done.into_iter().collect::<Result<()>>().map(|()| arena);
                    if csf_send.send(layer).is_err() { return; }
                }
            });
            let weights = Weights::load_selected(b, root, &checkpoint, rank,
                |name| {
                    !name.contains(".ngram_embedding.shard_")
                        && !name.contains(".nvfp4_csf_")
                        && !mxfp8(name)
                },
            )?;
            lap(1);
            arena = b.stats();
            let draft_moe = if mtp_depth == 0 {
                None
            } else {
                Some(mtp::load_moe(b, &weights, &plan)?)
            };
            lap(0);
            let staging = b.vector(super::next::PACK_TILES * 64 * 6144 / 2)?;
            let mut expanded = BTreeMap::new();
            let mut mx8 = Vec::new();
            for &(_, module, rows, cols) in targets {
                let (bf, image) = mx8_ready.recv().map_err(|_| "MXFP8 load workers stopped")??;
                lap(2);
                let mut v = b.vector(rows * cols / 2)?;
                b.write_vector_bytes(&mut v, unsafe {
                    std::slice::from_raw_parts(bf.as_ptr().cast::<u8>(), bf.len() * 2)
                })?;
                pack_bf16(b, v.device_address(), rows, cols, &staging)?;
                if let Some(image) = image {
                    let mut twin = b.vector(image.len().div_ceil(4))?;
                    b.write_vector_bytes(&mut twin, &image)?;
                    mx8.push(blocks::Mx8Twin::register(v.device_address(), twin, rows, cols));
                }
                expanded.insert(module.to_owned(), v);
                lap(5);
            }
            if expanded.len() != 384 {
                return Err(format!(
                    "expected 384 MXFP8 projections, expanded {}",
                    expanded.len()
                ));
            }
            let mut scales = Vec::with_capacity(LAYERS);
            for li in 0..LAYERS {
                for id in 0..EXPERTS {
                    for (pi, proj) in PROJ.iter().enumerate() {
                        let m = format!("model.language_model.layers.{li}.mlp.experts.{id}.{proj}");
                        let (rows, cols) = shape(pi);
                        if weights.tensor(&format!("{m}.weight"))?.bytes != rows * cols / 2 {
                            return Err(format!("{m}: packed expert length disagree"));
                        }
                    }
                }
                lap(0);
                let arena = csf_ready.recv().map_err(|_| "CSF load workers stopped")??;
                lap(6);
                let mut v = b.vector(arena.len() / 4)?;
                b.write_vector_bytes(&mut v, &arena)?;
                scales.push(v);
                lap(8);
            }
            Ok((weights, draft_moe, staging, expanded, mx8, scales))
        })?;
        drop(shards);
        // Reorder the sole resident BF16 head in bounded 64-row tiles. The
        // aligned K32 layout shares one weight pass between both M8 groups.
        // Consecutive 64-row tiles per step (same bytes: both layouts keep each 64-row block contiguous).
        let head = weights.device_address("lm_head.weight")?;
        let vocab = checkpoint.config.vocab as usize;
        let tiles = if vocab % 64 == 0 { (staging.len() * 4 / (64 * H * 2)).clamp(1, super::next::PACK_TILES) } else { 1 };
        for row in (0..vocab).step_by(64 * tiles) {
            let n = if tiles == 1 { 64 } else { (vocab - row).min(64 * tiles) };
            let destination = head + (row * H * 2) as u64;
            let mut copy = HybridCopy {
                source: destination as _,
                destination: staging.device_address() as _,
                words: (n * H / 2) as u32,
                padding: 0,
                rows: 0,
            };
            let mut pack = FlashBf16Pack {
                source: staging.device_address(),
                weight: destination,
                rows: n as u32,
                cols: H as u32,
            };
            unsafe {
                b.launch_flash_rows(
                    "flash_hybrid_copy",
                    (n * H / 2).div_ceil(128) as u32,
                    1,
                    128,
                    0,
                    &mut [(&mut copy as *mut HybridCopy).cast()],
                )?;
                b.launch_flash(
                    "flash_next_head_layout",
                    (n * H).div_ceil(128) as u32,
                    128,
                    &mut [(&mut pack as *mut FlashBf16Pack).cast()],
                )?;
            }
        }
        b.synchronize()?;
        lap(9);
        let scratch_savings = crate::flash_next_workspace::saved_bytes(H, 48, 16, crate::flash_next_rows::prompt_tile());
        let head_bounds = crate::flash_head_options::load_local(
            crate::flash_head_options::load_funded(scratch_savings, Head::mx8_draft_bytes(vocab), vocab,
                POOL_SIZE, mtp_depth != 0)?, crate::flash_head_options::bounds_rows(vocab, headless).is_none(), || { let (start, rows) = crate::flash_head_options::bounds_rows(vocab, headless).ok_or("no head bounds rows")?; Head::load_bounds(b, head, start, rows) })?;
        // Capture draft configuration once; all sessions borrow this immutable twin.
        let draft_head_rows = Head::draft_rows(checkpoint.config.vocab as usize,
            crate::flash_draft_stack::draft_vocab()?);
        let draft_head_mx8 = if mtp_depth != 0
            && !std::env::var("VADE_FLASHNEXT_DRAFT_MX8").is_ok_and(|v| v == "0") {
            Head::load_mx8_draft(b, head, draft_head_rows)?
        } else { None };
        // Drafter head (VADE_FLASHNEXT_DRAFT_FP4=1|2, unset = 2, =0 off): NVFP4 image of the
        // scored rows for one-chat proposals; verification keeps the full head.
        let draft_head_fp4 = if mtp_depth != 0 && crate::flash_draft_stack::draft_fp4()? != 0 {
            let twin = Head::load_fp4_draft(b, head, draft_head_rows)?;
            eprintln!("flashnext draft_head_fp4 rows={} bytes={}", draft_head_rows, Head::fp4_draft_bytes(draft_head_rows));
            twin
        } else { None };
        // VADE_FLASHNEXT_DRAFT_MX8_LAYER (unset = on, =0 off): MXFP8 twins of the MTP layer's
        // BF16 fc / q,k,v,index / o matrices; M <= 4 draft launches read them (half the bytes).
        // Only proposals read the MTP layer, so output bytes never depend on it.
        let draft_layer_mx8 = if mtp_depth != 0 && blocks::draft_mx8_layer() {
            let p = "mtp.layers.0.self_attn.";
            let heads = (weights.plan.query_heads.end - weights.plan.query_heads.start) as usize;
            let keys = (weights.plan.kv_heads.end - weights.plan.kv_heads.start) as usize;
            let mut twins = Vec::new();
            for (name, rows, cols) in [("mtp.fc_embedding.weight".to_string(), H, H), ("mtp.fc_hidden.weight".to_string(), H, H),
                (format!("{p}q_proj.weight"), heads * 512, H), (format!("{p}k_proj.weight"), keys * 256, H),
                (format!("{p}v_proj.weight"), keys * 256, H), (format!("{p}indexer.index_qk_proj.weight"), 640, H),
                (format!("{p}o_proj.weight"), H, heads * 256)] {
                let address = super::gdn::bf16(&weights, &name, &[rows as u64, cols as u64], rows * cols)?;
                twins.push(blocks::load_mx8_twin(b, address, rows, cols)?);
            }
            b.synchronize()?;
            eprintln!("flashnext draft_layer_mx8 matrices={} bytes={}", twins.len(), twins.iter().map(|t| t.twin.len() * 4).sum::<usize>());
            twins
        } else { Vec::new() };
        // After the drafter twin (the last reader of the BF16 head): LBF12 in place.
        let head_lbf12 = Head::encode_lbf12(b, head, checkpoint.config.vocab as usize)?;
        // LBF12 twins of the HC up weights (decode M1; VADE_HEAD_LBF12=0 keeps every head and
        // HC weight read in BF16).
        let mut lbf12_twins = Vec::new();
        if head_lbf12.is_some() {
            for (name, t) in checkpoint.tensors() {
                if name.ends_with(".input_mix_weight_up.weight") && t.dtype == Dtype::Bf16 && t.shape.len() == 2 {
                    let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
                    if let Some(twin) = blocks::lbf12_twin(b, weights.device_address(name)?, rows, cols)? {
                        lbf12_twins.push(twin);
                    }
                }
            }
            b.synchronize()?;
        }
        let addr = |module: &str| {
            expanded
                .get(module)
                .map(|v| v.device_address())
                .ok_or_else(|| format!("missing expanded projection {module}"))
        };
        let project = |base: String, names: [&str; 5]| -> Result<[u64; 5]> {
            Ok(names
                .map(|n| addr(&format!("{base}.{n}")))
                .into_iter()
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .unwrap())
        };
        let (mut gdn_proj, mut qsa_proj, mut moe) = (Vec::new(), Vec::new(), Vec::new());
        for li in 0..LAYERS {
            let p = format!("model.language_model.layers.{li}");
            // Immutable descriptors retain stored nibbles, expanded E4M3 scales
            // and the checkpoint's global scalar. No per-step host route work.
            use spark_gpu_probe::driver::fp4_abi::{Expert, Matrix};
            let mut descriptors = Vec::with_capacity(EXPERTS);
            for id in 0..EXPERTS {
                let mut matrices = [Matrix::default(); 3];
                for (pi, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                    let name = format!("{p}.mlp.experts.{id}.{proj}");
                    matrices[pi] = Matrix {
                        weights: weights.device_address(&format!("{name}.weight"))?,
                        scales: scales[li].device_address() + ((id * 3 + pi) * SCALE_BYTES) as u64,
                        global_scale: weights.device_address(&format!("{name}.weight_scale_2"))?,
                    };
                }
                descriptors.push(Expert {
                    gate: matrices[0],
                    up: matrices[1],
                    down: matrices[2],
                    width: 640,
                    reserved: 0,
                });
            } if crate::flash_tp_switch::on() { tp::shifted(&mut descriptors, &tp::layout(&checkpoint.config)?); } // the tensor split: rank 1's halves follow
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    descriptors.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(descriptors.as_slice()),
                )
            };
            let mut experts = b.vector(bytes.len() / 4)?;
            b.write_vector_bytes(&mut experts, bytes)?;
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
            moe.push(MoeAddr {
                experts,
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
        let [gdn_norms, qsa_norms] =
            crate::flash_next_storage::session_norms(b, |name| weights.values_f32(name))?;
        let pool = RefCell::new(execution::Pool::new(b)?);
        let prefill_reader = crate::flash_next_prefill::Reader::load(&checkpoint, &plan, root)?;
        b.synchronize()?;
        lap(10);
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        if std::env::var_os("VADE_LOAD_DIGEST").is_some() {
            // Receipt only: every resident weight byte, read back after the load (address-bearing
            // expert descriptors excluded). Equal digests = equal weights across builds.
            let at = std::time::Instant::now();
            let mut arena = super::next_load::Digest::new();
            let mut chunk = vec![0u8; 256 << 20];
            let total = weights.resident_bytes();
            for offset in (0..total).step_by(chunk.len()) {
                let n = chunk.len().min(total - offset);
                weights.read_arena(offset, &mut chunk[..n])?;
                arena.bytes(&chunk[..n]);
            }
            let digest = |vectors: &mut dyn Iterator<Item = &DeviceVector<'a>>| -> Result<super::next_load::Digest> {
                let mut d = super::next_load::Digest::new();
                for v in vectors { d.words(&b.read_vector_values(v)?); }
                Ok(d)
            };
            let (x, m, sc) = (digest(&mut expanded.values())?, digest(&mut mx8.iter().map(|t| &t.twin))?, digest(&mut scales.iter())?);
            let (dh, nr) = (digest(&mut draft_head_mx8.iter())?, digest(&mut gdn_norms.iter().chain(&qsa_norms).flatten())?);
            eprintln!("flashnext load_digest arena={:016x}/{} expanded={:016x}/{} mx8={:016x}/{} scales={:016x}/{} draft_head={:016x}/{} norms={:016x}/{} ms={:.0}",
                arena.0, arena.1, x.0, x.1, m.0, m.1, sc.0, sc.1, dh.0, dh.1, nr.0, nr.1, ms(at.elapsed()));
        }
        eprintln!("flashnext load_phases total_ms={:.0} other_ms={:.0} arena_ms={:.0} arena_read_upload_ms={:.0} alloc_ms={:.0} arena_file_bytes={} arena_buffered_bytes={} \
            mxfp8_read_ms={:.0} mxfp8_expand_ms={:.0} mxfp8_image_ms={:.0} mxfp8_upload_ms={:.0} \
            csf_read_ms={:.0} csf_expand_ms={:.0} csf_upload_ms={:.0} head_ms={:.0} tail_ms={:.0}",
            ms(started.elapsed()), ms(phase[0]), ms(phase[1]), arena.weight_read_upload_ns as f64 / 1e6,
            (arena.allocation_ns - before.allocation_ns) as f64 / 1e6, arena.weight_file_bytes, arena.weight_buffered_bytes,
            ms(phase[2]), ms(phase[3]), ms(phase[4]), ms(phase[5]), ms(phase[6]), ms(phase[7]), ms(phase[8]),
            ms(phase[9]), ms(phase[10]));
        Ok(Self {
            mtp_depth,
            draft_moe,
            draft_head_rows,
            draft_head_mx8,
            draft_head_fp4,
            draft_layer_mx8,
            head_lbf12,
            head_bounds,
            lbf12_twins,
            prefill_reader,
            prompt_frontend: true,
            frontend_host: std::cell::Cell::new((0, std::time::Duration::ZERO)),
            pool,
            work: RefCell::new(None),
            gdn_norms,
            qsa_norms,
            checkpoint,
            plan,
            weights,
            expanded,
            mx8,
            scales,

            gdn_proj,
            qsa_proj,
            moe,
            ple_global,
            hash: (mult, sizes, offsets, 248044),
            pp: RefCell::new(None), tp: RefCell::new(None), tp_box: std::cell::OnceCell::new(),
            pp_split: std::cell::Cell::new(0),
            pp_end: std::cell::Cell::new(LAYERS),
            state_layers: std::cell::Cell::new((0, LAYERS)),
            state_charge: RefCell::new(vec![0..LAYERS]),
        })
    }
    /// Device-byte estimate at `capacity` (loaded weights; FP32 KV + index
    /// vectors and GDN state from the loader plan; session scratch excluded).
    pub fn memory_plan(&self, capacity: usize) -> Result<MemoryPlan> {
        let resident_weights = self.weights.resident_bytes() as u64
            + self.draft_head_mx8.as_ref().map_or(0, |_| Head::mx8_draft_bytes(self.draft_head_rows) as u64)
            + self.draft_head_fp4.as_ref().map_or(0, |_| Head::fp4_draft_bytes(self.draft_head_rows) as u64)
            + self.draft_layer_mx8.iter().map(|t| t.twin.len() as u64 * 4).sum::<u64>()
            + self.head_lbf12.as_ref().map_or(0, |v| v.len() as u64 * 4)
            + self.head_bounds.as_ref().map_or(0, |v| v.bytes())
            + self.lbf12_twins.iter().map(|v| v.len() as u64 * 4).sum::<u64>()
            + self
                .draft_moe
                .as_ref()
                .map_or(0, |m| m.experts.len() as u64 * 4)
            + self.pool.borrow().table.len() as u64 * 4
            + self
                .gdn_norms
                .iter()
                .chain(&self.qsa_norms)
                .flatten()
                .map(|v| v.len() as u64 * 4)
                .sum::<u64>();
        // MXFP8 twins count with the expansion they shadow.
        let expanded_bf16 = self.expanded.values().chain(self.mx8.iter().map(|m| &m.twin))
            .map(|v| v.len() as u64 * 4).sum();
        let expanded_scales = self.scales.iter().map(|v| v.len() as u64 * 4).sum::<u64>()
            + self
                .moe
                .iter()
                .map(|m| m.experts.len() as u64 * 4)
                .sum::<u64>();
        let (kv, recurrent) = self.plan.cache_bytes(capacity as u64, false).map_err(err)?;
        Ok(MemoryPlan {
            resident_weights,
            expanded_bf16,
            expanded_scales,
            kv,
            recurrent,
            total: resident_weights + expanded_bf16 + expanded_scales + kv + recurrent,
        })
    }
    pub fn mtp_depth(&self) -> usize {
        self.mtp_depth
    }
    pub fn vocabulary(&self) -> usize {
        self.checkpoint.config.vocab as usize
    }
    /// Number of leading target IDs the drafter head can propose.
    pub fn draft_head_vocabulary(&self) -> usize { self.draft_head_rows }
    /// Pooled frontend submissions: cohort steps and host read/decode/upload time.
    /// Excludes graph capture/replay and GPU completion; cumulative since load.
    pub fn frontend_host_stats(&self) -> (u64, std::time::Duration) {
        self.frontend_host.get()
    }
    /// Conservative admission for session-owned KV, recurrent state and draft state.
    pub fn session_reservation_bytes(&self, capacity: usize) -> Result<usize> {
        self.session_reservation_bytes_for_chunk(capacity, crate::flash_next_rows::prompt_tile())
    }
    /// Admission for a session capped at this prompt chunk; use session_with_chunk.
    pub fn session_reservation_bytes_for_chunk(
        &self,
        capacity: usize,
        chunk: usize,
    ) -> Result<usize> {
        // Row scratch lives in the model's one workspace (shared_reservation_bytes).
        crate::flash_next_rows::reservation_rows(capacity, chunk)?;
        if !(1..=262144).contains(&capacity) {
            return Err("Flash-Next context must be 1..=262144".into());
        }
        let plan = self.memory_plan(capacity)?;
        // The costliest funded stage (all 48 layers unless own_stage_state ran).
        Ok(crate::flash_next_rows::stage_charge_bytes(plan.kv as usize, &self.state_charge.borrow())
            + (16 << 20)
            + mtp::Draft::reservation_bytes(self, capacity)?)
    }
    /// Shared decode scratch and graph metadata, funded once per loaded model.
    pub fn shared_reservation_bytes(&self, capacity: usize) -> Result<usize> {
        self.shared_reservation_bytes_for_arms(capacity, 1)
    }
    /// A/B keeps two graph banks, while their scratch and state remain shared.
    pub fn shared_reservation_bytes_for_arms(&self, capacity: usize, arms: usize) -> Result<usize> {
        self.shared_reservation_bytes_for_sessions(capacity, arms, self.pool.borrow().session_limit)
    }
    fn shared_reservation_bytes_for_sessions(&self, capacity: usize, arms: usize, sessions: usize) -> Result<usize> {
        if !(1..=2).contains(&arms) { return Err("Flash-Next requires one or two kernel arms".into()); }
        if !(1..=262144).contains(&capacity) {
            return Err("Flash-Next context must be 1..=262144".into());
        }
        let tile = crate::flash_next_rows::prompt_tile();
        Ok(execution::Pool::reservation_bytes(capacity, arms, self.mtp_depth, sessions, self.prompt_frontend)?
            + Workspace::reservation_bytes(capacity.max(tile), tile, self.mtp_depth, self.prompt_frontend)
            + spark_gpu_probe::flash_dense_reserve_bytes()
            + spark_gpu_probe::flash_mixed_reserve_bytes())
    }
    /// Decode-graph storage the shared reservation funds for the current slot limit (one
    /// kernel arm; a second arm's allowance only adds unreserved headroom).
    pub fn decode_graph_allowance_bytes(&self) -> Result<usize> {
        execution::Pool::graph_allowance_bytes(1, self.pool.borrow().session_limit)
    }
    /// A speculative cohort does not use the pooled one-row execution cache.
    /// Finish its readers before releasing graphs and their scratch. The slot
    /// table, admitted sessions and their state remain live and unchanged.
    pub fn release_pooled_decode(&self, b: &Backend<'a>) -> Result<()> {
        if !self.pool.borrow().decode.is_empty() {
            b.synchronize()?;
            eprintln!("flash_next cohort_release_pooled widths={}", self.pool.borrow().decode.len());
            self.pool.borrow_mut().decode.clear();
        }
        self.pp_release_pooled_decode();
        Ok(())
    }
    /// Largest affordable slot limit, session pool bytes and still-unallocated shared bytes.
    /// `resident_tiles` counts backend dense/mixed tiles already deducted from `available`.
    pub fn session_pool_plan(&self, capacity: usize, arms: usize, available: usize, resident_tiles: usize)
        -> Result<(usize, usize, usize)>
    {
        let needed = self.session_reservation_bytes(capacity)?;
        crate::flash_next_rows::fit_session_pool(available, needed, |sessions| {
            Ok(self.shared_reservation_bytes_for_sessions(capacity, arms, sessions)?
                .saturating_sub(resident_tiles))
        })
    }
    /// Set before admitting chats: no slot above this limit can create a wider graph bank.
    pub fn set_session_limit(&self, sessions: usize) -> Result<()> {
        let mut pool = self.pool.borrow_mut();
        if !(1..=POOL_SIZE).contains(&sessions) || pool.used[1..].contains(&true) || !pool.decode.is_empty() {
            return Err("Flash-Next session limit requires 1..=16 slots and an unused pool".into());
        }
        pool.session_limit = sessions;
        Ok(())
    }
    pub fn session<'w>(&'w self, b: &Backend<'a>, capacity: usize) -> Result<NextSession<'a, 'w>> {
        self.session_with_chunk(b, capacity, crate::flash_next_rows::prompt_tile())
    }
    /// Cap prompt scratch to the width used for memory admission.
    pub fn session_with_chunk<'w>(
        &'w self,
        b: &Backend<'a>,
        capacity: usize,
        chunk: usize,
    ) -> Result<NextSession<'a, 'w>> {
        crate::flash_next_rows::reservation_rows(capacity, chunk)?;
        self.pp_cache_releases()?; let mut session = NextSession::new(b, self, capacity)?;
        session.chunk_limit = chunk;
        session.chunk = chunk;
        Ok(session)
    }
}

#[path = "next_execution.rs"]
mod execution;
#[path = "next_mtp.rs"]
mod mtp;
#[path = "next_pp.rs"]
mod pp; #[path = "next_tp.rs"] mod tp;
pub use pp::{wire_capacity as pp_wire_capacity, negotiate_capacity as pp_negotiate_capacity, negotiate_wire as pp_negotiate_wire, LINK_TIMEOUT as PP_LINK_TIMEOUT, Stage as PpStage, PpWire};
#[path = "next_prefix.rs"]
mod prefix;
pub use prefix::{NextPrefix, NextStorage};
use execution::Workspace;
/// Sequence-owned GDN/KV/PLE histories. Scratch can serve either consecutive
/// prompt tokens or independent decode rows; failure requires reset or replacement.
pub struct NextSession<'a, 'w> {
    model: &'w NextModel<'a>,
    draft: Option<mtp::Draft<'a, 'w>>,
    saved: Vec<mtp::Prefix<'a>>,
    rollback: Option<mtp::Rollback<'a>>,
    retained: usize,
    verify_rows: usize,
    capacity: usize,
    pos: usize,
    hash: PleHash,
    attn_hc: Vec<Hc<'w>>,
    mlp_hc: Vec<Hc<'w>>,
    mixer: Hc<'w>,
    ple: Ple<'a, 'w>,
    head: Head<'a, 'w>,
    gdn_state: Vec<Option<GdnState<'a>>>,
    qsa_state: Vec<Option<QsaState<'a>>>,
    slot: u32,
    chunk: usize,
    chunk_limit: usize,
    layer_bytes: Option<PathBuf>,
    valid: bool,
    logits_ready: bool,
    indexer: Box<dyn IndexerHook>,
    dump: Option<Vec<String>>,
    /// VADE_MTP_BATCH: draft history over a chat's rows in tiles (Draft::observe_rows).
    batch_drafts: bool,
    /// VADE_MTP_DEFER_FEED: pooled steps defer this chat's drafter rows (its cohort cannot draft).
    defer_feed: bool,
    allow_pooled_decode: bool,
    pool_graph_fallback: bool,
    /// Prefix cache (next_prefix.rs): capture the prompt checkpoint on output prefills.
    /// Interior: publishing moves it out of a shared session (Session::fork takes &self).
    checkpointing: bool,
    checkpoint: RefCell<Option<prefix::Checkpoint<'a>>>,
}
impl<'a, 'w> NextSession<'a, 'w> {
    /// VADE_MTP_BATCH (serve sets it per chat; default off = one draft row at a time).
    pub fn batch_drafts(&mut self, on: bool) {
        self.batch_drafts = on;
    }
    /// Fixed cohort-wide policy for this tick; install the same values on every
    /// scheduled session. No environment reads in execution. Defaults need no writes.
    pub fn configure_pooled_decode(&mut self, allowed: bool, fallback: bool) {
        self.allow_pooled_decode=allowed; self.pool_graph_fallback=fallback;
    }
    /// VADE_MTP_DEFER_FEED (serve sets it per tick: the chat's cohort is above its draft limit).
    pub fn defer_draft_feed(&mut self, on: bool) {
        self.defer_feed = on && mtp::defer_feed();
    }
    fn new(b: &Backend<'a>, model: &'w NextModel<'a>, capacity: usize) -> Result<Self> {
        Self::new_from(b, model, capacity, None, None)
    }
    /// `qsa` / `draft`: a finished session's KV arenas to reuse instead of allocating
    /// (next_prefix.rs adoption); everything else starts fresh.
    fn new_from(b: &Backend<'a>, model: &'w NextModel<'a>, capacity: usize,
        mut qsa: Option<Vec<Option<QsaState<'a>>>>, draft: Option<QsaState<'a>>) -> Result<Self> {
        if capacity == 0 || capacity > 262144 {
            return Err("flashnext capacity must be 1..=262144".into());
        }
        let w = &model.weights;
        let hc = |role: &str| {
            (0..LAYERS)
                .map(|li| {
                    Hc::load_rows(
                        w,
                        &format!("model.language_model.layers.{li}.{role}_hyper_connection"),
                    )
                })
                .collect::<Result<Vec<_>>>()
        };
        let (attn_hc, mlp_hc) = (hc("attn")?, hc("mlp")?);
        let (mult, sizes, offsets, eos) = &model.hash;
        // gdn::session_arena: every fresh state from one allocation (load_next then only checks it).
        let mut fresh = if gdn::session_arena() {
            GdnState::fresh_arena(b, w, &model.state_range().filter(|li| li % 4 != 3).collect::<Vec<_>>())?
        } else {
            Vec::new()
        }
        .into_iter();
        let gdn_state = (0..LAYERS)
            .map(|li| {
                if li % 4 == 3 || !model.owns_state(li) {
                    Ok(None)
                } else {
                    Gdn::load_next(b, w, li, model.gdn_proj[li], fresh.next()).map(|g| Some(g.into_state()))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let qsa_state = (0..LAYERS)
            .map(|li| {
                if li % 4 != 3 || !model.owns_state(li) {
                    Ok(None)
                } else {
                    Qsa::load_next_rows(b, w, li, capacity, model.qsa_proj[li], qsa.as_mut().and_then(|q| q[li].take()))
                        .map(|q| Some(q.into_state()))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let mut result = Self {
            draft: if model.mtp_depth == 0 {
                None
            } else {
                Some(mtp::Draft::new(b, model, capacity, draft)?)
            },
            saved: Vec::new(),
            rollback: None,
            retained: 0,
            verify_rows: model.mtp_depth + 1,
            slot: 0,
            model,
            capacity,
            pos: 0,
            hash: PleHash::new(mult.clone(), sizes.clone(), offsets.clone(), *eos),
            attn_hc,
            mlp_hc,
            mixer: Hc::mixer_rows(w, "model.language_model.hyper_connection_mixer")?,
            ple: Ple::load(b, w)?,
            head: Head::load(b, w)?,
            gdn_state,
            qsa_state,
            chunk: 512,
            chunk_limit: crate::flash_next_rows::prompt_tile(),
            layer_bytes: None,
            valid: true,
            logits_ready: false,
            indexer: Box::new(SparseIndexer),
            dump: None,
            batch_drafts: false,
            defer_feed: false,
            allow_pooled_decode: true,
            pool_graph_fallback: false,
            checkpointing: false,
            checkpoint: RefCell::new(None),
        };
        let mut pool = model.pool.borrow_mut();
        let slot = (1..=pool.session_limit)
            .find(|&i| !pool.used[i])
            .ok_or("Flash-Next state pool has reached its admitted session limit")?;
        if capacity > pool.capacity {
            pool.decode.clear();
            pool.capacity = capacity;
        }
        let record = spark_gpu_probe::next_math::StateSlot {
            // Layers another stage owns never run here: their slot words stay zero.
            layers: std::array::from_fn(|li| {
                if !model.owns_state(li) {
                    [0; 3]
                } else if li % 4 == 3 {
                    result.qsa_state[li].as_ref().unwrap().pool_addresses()
                } else {
                    result.gdn_state[li].as_ref().unwrap().pool_addresses()
                }
            }),
            ple: result.ple.history_address(),
            position: 0,
            capacity: capacity as u32,
        };
        let words = std::mem::size_of_val(&record) / 4;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&record as *const spark_gpu_probe::next_math::StateSlot).cast(),
                words * 4,
            )
        };
        b.write_vector_bytes(
            &mut pool.table.slice(slot * words..(slot + 1) * words)?,
            bytes,
        )?;
        pool.used[slot] = true;
        result.slot = slot as u32;
        Ok(result)
    }
    /// Diagnostic complete layer outputs; disabled during timing and graph replay.
    pub fn check_layers(&mut self, path: PathBuf) -> Result<()> {
        std::fs::create_dir(&path).map_err(err)?;
        self.layer_bytes = Some(path);
        Ok(())
    }
    pub fn stop_layer_check(&mut self) {
        self.layer_bytes = None;
    }
    pub fn position(&self) -> usize {
        self.pos
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
        self.valid = false;
        self.logits_ready = false;
        if let Some(d) = self.draft.as_ref() { d.bound_at.set(None); }
        crate::flash_next_rows::check_owners(&self.gdn_state, &self.qsa_state, self.model.state_range())?;
        self.retained = 0;
        *self.checkpoint.get_mut() = None;
        if let Some(d) = &mut self.draft {
            d.reset();
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
        self.sync_pool_position(b)?;
        self.valid = true;
        Ok(())
    }
    fn sync_pool_state(&self, b: &Backend<'_>) -> Result<()> {
        crate::flash_next_rows::check_owners(&self.gdn_state, &self.qsa_state, self.model.state_range())?;
        let mut layers = [[0u64; 3]; LAYERS];
        for (li, layer) in layers.iter_mut().enumerate().filter(|(li, _)| self.model.owns_state(*li)) {
            *layer = if li % 4 == 3 {
                self.qsa_state[li]
                    .as_ref()
                    .ok_or("missing QSA owner")?
                    .pool_addresses()
            } else {
                self.gdn_state[li]
                    .as_ref()
                    .ok_or("missing GDN owner")?
                    .pool_addresses()
            };
        }
        let record = spark_gpu_probe::next_math::StateSlot {
            layers,
            ple: self.ple.history_address(),
            position: self.pos as u32,
            capacity: self.capacity as u32,
        };
        let words = std::mem::size_of_val(&record) / 4;
        b.write_vector_bytes(
            &mut self
                .model
                .pool
                .borrow()
                .table
                .slice(self.slot as usize * words..(self.slot as usize + 1) * words)?,
            unsafe {
                std::slice::from_raw_parts(
                    (&record as *const spark_gpu_probe::next_math::StateSlot).cast(),
                    words * 4,
                )
            },
        )
    }
    fn sync_pool_position(&self, b: &Backend<'_>) -> Result<()> {
        let words = std::mem::size_of::<spark_gpu_probe::next_math::StateSlot>() / 4;
        let at = self.slot as usize * words + words - 2;
        b.write_vector(
            &mut self.model.pool.borrow().table.slice(at..at + 1)?,
            &[f32::from_bits(self.pos as u32)],
        )
    }
    fn rec(&mut self, stage: &str, values: &[f32]) {
        if let Some(d) = &mut self.dump {
            d.push(dump::rec_line(stage, values));
        }
    }
    /// Bound prompt work between serving scheduler turns; arithmetic is invariant.
    pub fn set_chunk_size(&mut self, chunk: usize) -> Result<()> {
        if !(1..=self.chunk_limit).contains(&chunk) {
            return Err(format!(
                "Flash-Next chunk must be 1..={} (admitted limit)",
                self.chunk_limit
            ));
        }
        self.chunk = chunk;
        Ok(())
    }
    pub fn chunk_size(&self) -> usize {
        self.chunk
    }
    pub fn logits(&self, b: &Backend<'a>) -> Result<Vec<f32>> {
        if !self.valid {
            return Err("Flash-Next session needs reset after failure".into());
        }
        if !self.logits_ready {
            return Err("Flash-Next has no logits for the current position".into());
        }
        if let Some(rows) = self.canonical_bound_logits(b)? {
            return rows.into_iter().last().ok_or_else(|| "missing canonical head row".into());
        }
        self.canonical_split_logits(b)?.map_or_else(|| self.head.logits(b), Ok)
    }
    /// Process a prompt through bounded tiles. Only its last row projects lm_head.
    pub fn prefill(&mut self, b: &Backend<'a>, tokens: &[u32]) -> Result<Vec<f32>> {
        self.prefill_inner(b, tokens, true)?;
        self.logits(b)
    }
    /// Read-ahead for the prompt continuing at this position (`tokens` starts
    /// with the slice about to be prefilled, `chunk` tokens at most): a
    /// background thread prestages the frontend of the next slice's first tile
    /// (`prefill_inner`'s tiling) and hints the rows after it. No state or
    /// output changes; the caller never waits.
    pub fn read_ahead(&self, tokens: &[u32]) {
        let chunk = crate::flash_next_rows::prompt_tile_for(self.pos + tokens.len(), self.chunk);
        let slice = tokens.len().min(chunk);
        let rest = (tokens.len() - slice).min(chunk);
        let tile = crate::flash_next_rows::prompt_rows(self.pos + slice, rest, chunk);
        self.model.prefill_reader.read_ahead(
            self.hash.clone(),
            tokens.to_vec(),
            slice..slice + tile,
            self.model.ple_global,
        );
    }
    pub fn prefill_advance(&mut self, b: &Backend<'a>, tokens: &[u32]) -> Result<()> {
        self.prefill_inner(b, tokens, false)
    }
    fn prefill_inner(&mut self, b: &Backend<'a>, tokens: &[u32], output: bool) -> Result<()> {
        crate::flash_next_rows::validate(
            &[self.pos],
            &[tokens],
            &[self.capacity],
            self.model.checkpoint.config.vocab,
        )?;
        // VADE_PROMPT_TILE=auto: the tile follows where this prompt work ends.
        let chunk = crate::flash_next_rows::prompt_tile_for(self.pos + tokens.len(), self.chunk);
        let mut start = 0;
        while start < tokens.len() {
            // Prefix cache: the last 64-aligned row before the final partial tile.
            if output && tokens.len() - start < 64 {
                self.capture_checkpoint(b)?;
            }
            let mut n = self.model.tp_seq_cut(self, output, tokens.len() - start, chunk)
                .unwrap_or_else(|| crate::flash_next_rows::prompt_rows(self.pos, tokens.len() - start, chunk));
            // Debug dumps retain all per-token head/stage records. Custom hooks
            // that change selection inside a tile also keep their token contract.
            if self.dump.is_some()
                || (3..LAYERS).step_by(4).any(|li| {
                    (1..n).any(|i| {
                        self.indexer.select(li, self.pos, self.pos + 1)
                            != self.indexer.select(li, self.pos + i, self.pos + i + 1)
                    })
                })
            {
                n = 1;
            }
            let project = output && (start + n == tokens.len() || self.dump.is_some());
            Self::run_rows(b, &mut [self], &[&tokens[start..start + n]], project)?;
            start += n;
        }
        if output {
            self.capture_checkpoint(b)?;
        }
        Ok(())
    }
    /// Share bounded prompt tiles across independent causal histories.
    pub fn prefill_batch(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        tokens: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>> {
        crate::flash_next_rows::validate(
            &sessions.iter().map(|s| s.pos).collect::<Vec<_>>(),
            tokens,
            &sessions.iter().map(|s| s.capacity).collect::<Vec<_>>(),
            sessions
                .first()
                .map_or(0, |s| s.model.checkpoint.config.vocab),
        )?;
        let model = sessions[0].model;
        if sessions
            .iter()
            .any(|s| !s.valid || !std::ptr::eq(s.model, model))
        {
            return Err("Flash-Next cohort needs valid sessions of the same model".into());
        }
        let mut offsets = vec![0; tokens.len()];
        while offsets.iter().zip(tokens).any(|(&at, t)| at < t.len()) {
            let live: Vec<_> = offsets
                .iter()
                .zip(tokens)
                .enumerate()
                .filter_map(|(i, (&at, t))| (at < t.len()).then_some(i))
                .take(8)
                .collect();
            // Match single-chat prefill: retain the last complete 64-row prefix
            // before a final partial tile consumes its recurrent state.
            for &i in &live {
                if tokens[i].len() - offsets[i] < 64 {
                    sessions[i].capture_checkpoint(b)?;
                }
            }
            // VADE_PROMPT_TILE=auto: each chat's tile follows where its prompt work ends.
            let chunks: Vec<_> = sessions
                .iter()
                .zip(tokens.iter().zip(&offsets))
                .map(|(s, (t, &at))| crate::flash_next_rows::prompt_tile_for(s.pos + t.len() - at, s.chunk))
                .collect();
            let budget = crate::flash_next_rows::prefill_budget(
                sessions.iter().zip(&chunks).map(|(s, &chunk)| (chunk, s.chunk_limit)), live.len(),
            )?;
            let lengths: Vec<_> = live
                .iter()
                .map(|&i| {
                    let s = &sessions[i];
                    let n = crate::flash_next_rows::prompt_rows(
                        s.pos,
                        tokens[i].len() - offsets[i],
                        chunks[i].min(budget),
                    );
                    if s.dump.is_some()
                        || (3..LAYERS).step_by(4).any(|li| {
                            (1..n).any(|j| {
                                s.indexer.select(li, s.pos, s.pos + 1)
                                    != s.indexer.select(li, s.pos + j, s.pos + j + 1)
                            })
                        })
                    {
                        1
                    } else {
                        n
                    }
                })
                .collect();
            let input: Vec<_> = live
                .iter()
                .zip(&lengths)
                .map(|(&i, &n)| &tokens[i][offsets[i]..offsets[i] + n])
                .collect();
            let mut rows: Vec<_> = sessions
                .iter_mut()
                .enumerate()
                .filter_map(|(i, s)| live.contains(&i).then_some(&mut **s))
                .collect();
            let output = live
                .iter()
                .zip(&lengths)
                .any(|(&i, &n)| offsets[i] + n == tokens[i].len());
            Self::run_rows(b, &mut rows, &input, output)?;
            for (&i, &n) in live.iter().zip(&lengths) {
                offsets[i] += n;
            }
        }
        for session in sessions.iter_mut() {
            session.capture_checkpoint(b)?;
        }
        sessions.iter().map(|s| s.logits(b)).collect()
    }
    /// Process one token; this is the identical M-row path at M=1, including cold state.
    pub fn step(&mut self, b: &Backend<'a>, token: u32) -> Result<u32> {
        Self::run_rows(b, &mut [self], &[&[token]], true)?;
        let result = self
            .head
            .local_max(b)
            .and_then(|v| v.map(|v| v.token).ok_or_else(|| "empty vocabulary".into()));
        if result.is_err() {
            self.valid = false;
        }
        result
    }
    /// One token from each of 1..=16 independent chats. Row order is caller order.
    /// All participants must share this model; on failure every participant needs reset.
    pub fn step_batch(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let inputs = tokens.iter().map(std::slice::from_ref).collect::<Vec<_>>();
        Self::run_rows(b, sessions, &inputs, true)?;
        sessions.iter().map(|s| s.logits(b)).collect()
    }
    /// Feed bounded prompt chunks, then greedily generate ids.
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
        self.prefill_inner(b, prompt, true)?;
        let mut next = self.head.local_max(b)?.ok_or("empty vocabulary")?.token;
        let mut out = vec![next];
        while out.len() < new_tokens {
            next = self.step(b, next)?;
            out.push(next);
        }
        Ok(out)
    }
}

impl Drop for NextSession<'_, '_> {
    fn drop(&mut self) {
        if self.slot != 0 {
            self.model.pool.borrow_mut().used[self.slot as usize] = false;
            // Pipeline front: kept rows never outlive their chat.
            self.model.pp_drop_kept(self.slot);
        }
    }
}

impl Drop for NextModel<'_> {
    fn drop(&mut self) {
        self.pool.get_mut().decode.clear();
        // self.mx8 / self.draft_layer_mx8: each Mx8Twin unregisters itself when dropped.
    }
}

/// Rank-owned experts use the world runner's BF16 boundaries and separate
/// multiply/add fold. One M16 tile shares each owned expert across chat rows.
/// Scratch is retained by the controller session between layer boundaries.
pub struct RankRows<'a> {
    buffers: Vec<DeviceVector<'a>>,
}
impl<'a> RankRows<'a> {
    pub fn new(b: &Backend<'a>) -> Result<Self> {
        Ok(Self {
            buffers: b.vector_arena(&[
                8 * H,
                80,
                80 * 3,
                1,
                80 * H / 2,
                80 * 1280 / 2,
                80 * 640 / 2,
                80 * H / 2,
            ])?,
        })
    }
    pub fn experts(
        &mut self,
        b: &Backend<'a>,
        table: u64,
        owned: std::ops::Range<usize>,
        inputs: &[&DeviceVector<'a>],
        routes: &[Vec<(u32, f32)>],
    ) -> Result<Vec<Vec<f32>>> {
        use spark_gpu_probe::kernel_params::{NextActivation, NextMoe, NextRouted};
        if inputs.is_empty() || inputs.len() > 8 || inputs.len() != routes.len() {
            return Err("invalid rank expert rows".into());
        }
        let mut groups = BTreeMap::<u32, Vec<(usize, f32)>>::new();
        for (row, route) in routes.iter().enumerate() {
            for &(id, w) in route {
                if owned.contains(&(id as usize)) {
                    groups.entry(id).or_default().push((row, w));
                }
            }
        }
        let mut rows = vec![0u32; 80];
        let mut tiles = Vec::new();
        let mut pairs = Vec::new();
        for (&id, group) in &groups {
            tiles.extend([id, pairs.len() as u32, group.len() as u32]);
            for &(row, w) in group {
                rows[pairs.len()] = row as u32;
                pairs.push((row, w));
            }
        }
        for (row, input) in inputs.iter().enumerate() {
            b.copy_range(input, &mut self.buffers[0], 0, row * H, H)?;
        }
        b.write_vector(
            &mut self.buffers[1],
            &rows.into_iter().map(f32::from_bits).collect::<Vec<_>>(),
        )?;
        if !tiles.is_empty() {
            b.write_vector(
                &mut self.buffers[2].slice(0..tiles.len())?,
                &tiles
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>(),
            )?;
        }
        b.write_vector(&mut self.buffers[3], &[f32::from_bits(groups.len() as u32)])?;
        for i in [5, 6, 7] {
            b.zero_vector(&mut self.buffers[i])?;
        }
        let a = self
            .buffers
            .iter()
            .map(|v| v.device_address())
            .collect::<Vec<_>>();
        let mut gather = NextMoe {
            input: a[0],
            rows: a[1],
            gathered: a[4],
            tokens: inputs.len() as u32,
            ..Default::default()
        };
        let mut project = NextRouted {
            experts: table,
            input: a[4],
            output: a[5],
            tiles: a[2],
            tile_count: a[3],
            rows: 1280,
            cols: H as u32,
            ..Default::default()
        };
        let mut activation = NextActivation {
            input: a[5],
            output: a[6],
            rows: 80,
            reserved: 0,
        };
        unsafe {
            b.launch_flash_shared(
                "flash_next_gather",
                (inputs.len() * 10 * H).div_ceil(256) as u32,
                256,
                0,
                &mut [(&mut gather as *mut NextMoe).cast()],
            )?;
            b.launch_flash_shared(
                "flash_next_routed",
                192,
                256,
                0,
                &mut [(&mut project as *mut NextRouted).cast()],
            )?;
            b.launch_flash_shared(
                "flash_next_routed_activate",
                (80usize * 640).div_ceil(256) as u32,
                256,
                0,
                &mut [(&mut activation as *mut NextActivation).cast()],
            )?;
            project.input = a[6];
            project.output = a[7];
            project.rows = H as u32;
            project.cols = 640;
            project.down = 1;
            b.launch_flash_shared(
                "flash_next_routed",
                384,
                256,
                0,
                &mut [(&mut project as *mut NextRouted).cast()],
            )?;
        }
        let packed = b.read_vector(&self.buffers[7])?;
        let mut out = vec![vec![0.; H]; inputs.len()];
        // groups and pairs are in ascending expert ID, exactly as per-request.
        for (pair, (row, w)) in pairs.into_iter().enumerate() {
            for i in 0..H {
                out[row][i] += nvfp4::bf16_to_f32(
                    (packed[(pair * H + i) / 2].to_bits() >> ((i % 2) * 16)) as u16,
                ) * w;
            }
        }
        Ok(out)
    }
}

/// Independent native projection descriptors keep the M1 arithmetic on each
/// row while sharing launches. Causal GDN state remains in each session.
pub fn native_gdn_rows<'a>(
    b: &Backend<'a>,
    jobs: &mut [(
        &mut Gdn<'a, '_>,
        &mut GdnScratch<'a>,
        &DeviceVector<'a>,
        [super::next_native::Matrix; 5],
    )],
) -> Result<()> {
    let mut projections = Vec::new();
    for (_, s, x, m) in jobs.iter() {
        let ys = [
            s.qkv.device_address(),
            s.z.device_address(),
            s.a.device_address(),
            s.b.device_address(),
        ];
        for i in 0..4 {
            projections.push((m[i], x.device_address(), ys[i]));
        }
    }
    for group in projections.chunks(9) {
        super::next_native::project(b, group, 1, false, false)?;
    }
    for (g, s, x, _) in jobs.iter_mut() {
        g.run_projected(b, s, x, 1, false, false, 0)?;
    }
    projections.clear();
    for (_, s, _, m) in jobs.iter() {
        projections.push((m[4], s.gated.device_address(), s.output().device_address()));
    }
    for group in projections.chunks(9) {
        super::next_native::project(b, group, 1, false, false)?;
    }
    Ok(())
}
