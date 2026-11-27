//! Official four-stream MTP. Proposals share target embeddings and vocabulary
//! weights; only target predictions may commit a token or recurrent prefix.
use super::*;
use super::super::blocks::{bf16_rows, add_rows};

const PREFIX: &str = "mtp.layers.0.self_attn.";
const TILE: usize = 64;
pub(super) const MAX_MTP_BATCH: usize = 8;

pub(super) struct MtpBatchWork<'a, 'w> {
    previous: DeviceVector<'a>,
    embedding: DeviceVector<'a>,
    normalized: DeviceVector<'a>,
    projected: DeviceVector<'a>,
    hidden: DeviceVector<'a>,
    input: DeviceVector<'a>,
    hc: HcScratch<'a>,
    mlp: Moe<'w>,
    moe: MoeScratch<'a>,
    descriptors: DeviceVector<'a>,
    qsa: QsaBatch<'a>,
    qsa_norm: Option<DeviceVector<'a>>,
    argmax_partials: DeviceVector<'a>,
    argmax_result: DeviceVector<'a>,
}
impl<'a, 'w> MtpBatchWork<'a, 'w> {
    pub(super) fn reservation_bytes(model: &Model<'_>) -> usize {
        if model.mtp_depth == 0 { return 0; }
        let vocab_shard = (model.persistent[0].plan.vocabulary.end - model.persistent[0].plan.vocabulary.start) as usize;
        MAX_MTP_BATCH * (3 * W + 3 * H) * 4
            + MAX_MTP_BATCH * HcScratch::DEVICE_BYTES
            + MAX_MTP_BATCH * GROUP_TOKEN_BYTES + EXPERT_PADDING_BYTES
            + 512 * 7 * 8
            + MAX_MTP_BATCH * (12288 + 512 + 512 + 640 + 6144) * 4 + 640
            + MAX_MTP_BATCH * std::mem::size_of::<spark_gpu_probe::kernel_params::QsaBatchRow>()
            + (768 + vocab_shard.div_ceil(1024) * 3 + MAX_MTP_BATCH * 3) * 4
            + 65536
    }
    pub(super) fn new(b: &Backend<'a>, model: &'w Model<'a>) -> Result<Self> {
        let w = &model.persistent[0];
        let vocab_shard = (w.plan.vocabulary.end - w.plan.vocabulary.start) as usize;
        let mut v = b.vector_arena(&[
            MAX_MTP_BATCH * W, MAX_MTP_BATCH * H, MAX_MTP_BATCH * W,
            MAX_MTP_BATCH * H, MAX_MTP_BATCH * W, MAX_MTP_BATCH * H,
            (vocab_shard.div_ceil(1024) * 3).max(1), MAX_MTP_BATCH * 3,
        ])?.into_iter();
        let mlp = Moe::load_prefix(w, "mtp.layers.0.mlp", model.activation_quantization)?;
        let mut moe = MoeScratch::with_capacity(b, MAX_MTP_BATCH)?;
        moe.reserve_grouped(b, MAX_MTP_BATCH, model.activation_quantization, None)?;
        let descriptors = mlp.device_descriptors(b)?;
        Ok(Self {
            previous: v.next().unwrap(), embedding: v.next().unwrap(), normalized: v.next().unwrap(),
            projected: v.next().unwrap(), hidden: v.next().unwrap(), input: v.next().unwrap(),
            argmax_partials: v.next().unwrap(), argmax_result: v.next().unwrap(),
            hc: HcScratch::with_tokens(b, MAX_MTP_BATCH)?, mlp, moe, descriptors,
            qsa: QsaBatch::new(b, &w.plan, MAX_MTP_BATCH)?, qsa_norm: None,
        })
    }
    pub(super) fn device_bytes(&self) -> usize {
        (self.previous.len() + self.embedding.len() + self.normalized.len()
            + self.projected.len() + self.hidden.len() + self.input.len()
            + self.descriptors.len() + self.argmax_partials.len() + self.argmax_result.len()
            + self.qsa_norm.as_ref().map_or(0, DeviceVector::len)) * 4
            + self.hc.device_bytes() + self.moe.device_bytes() + self.qsa.device_bytes()
    }
}

fn copy_words(b: &Backend<'_>, source: u64, destination: u64, bytes: usize) -> Result<()> {
    if bytes == 0 { return Ok(()); }
    let mut p = HybridCopy { source: source as _, destination: destination as _, words: (bytes / 4) as u32, padding: 0, rows: 0 };
    unsafe { b.launch_flash("flash_hybrid_copy", p.words.div_ceil(128), 128,
        &mut [(&mut p as *mut HybridCopy).cast()]) }
}

pub(in crate::flash) fn reservation_bytes(plan: &RankPlan, capacity: usize, depth: usize) -> usize {
    if depth == 0 { return 0; }
    let heads = (plan.value_heads.end - plan.value_heads.start) as usize;
    let keys = (plan.key_heads.end - plan.key_heads.start) as usize;
    // One owner retains `depth` recurrent prefixes. Attention needs one query
    // row; its old/new buffers can coexist during capacity growth. The GDN
    // decode bound covers the smaller projection-only verifier workspace.
    // A one-token MoE has at most 160 padded W8A8 rows (2,563,200 bytes),
    // plus <1 MiB of HC, routing, projections, descriptors and alignment.
    depth * (36 * (heads * 128 * 128 + (2 * keys + heads) * 128 * 4) * 4 + Ple::HISTORY_BYTES)
        + 2 * QsaScratch::reservation_bytes(plan, 1, capacity, false)
        + GdnScratch::decode_reservation_bytes(H, heads, keys, depth + 1)
        + (depth + 1) * (plan.vocabulary.end - plan.vocabulary.start) as usize * 4
        + 4 * 1024 * 1024 + 65536
}

pub(super) fn snapshot_bytes(plan: &RankPlan, capacity: usize) -> usize {
    W * 4 + capacity * (2 * (plan.kv_heads.end - plan.kv_heads.start) as usize * 256 + 128) * 4 + 256
}
pub(super) fn history_bytes(plan: &RankPlan, capacity: usize, depth: usize) -> usize {
    if depth == 0 { return 0; }
    (W + TILE * (3 * W + 3 * H)) * 4 + TILE * HcScratch::DEVICE_BYTES
        + QsaScratch::reservation_bytes(plan, TILE, capacity, true) + 2 * snapshot_bytes(plan, capacity)
        // Head shortlists survive release of the shared drafter workspace.
        // Fund each session, including its largest supported top-k (64).
        + Head::top_k_reservation_bytes((plan.vocabulary.end - plan.vocabulary.start) as usize)
}

#[repr(C)]
struct Norm { x: u64, weight: u64, output: u64, width: u32, groups: u32,
    zero_centered: u32, epsilon: f32, repeat: u32, precision: u32 }
fn normalize(b: &Backend<'_>, x: u64, weight: u64, output: u64, width: usize, rows: usize) -> Result<()> {
    let mut p = Norm { x, weight, output, width: width as u32, groups: 1,
        zero_centered: 1, epsilon: 1e-6, repeat: 0, precision: 0 };
    unsafe { b.launch_flash_rows("flash_norm", 8, rows as u32, 128, 0, &mut [
        (&mut p.x as *mut u64).cast(), (&mut p.weight as *mut u64).cast(),
        (&mut p.output as *mut u64).cast(), (&mut p.width as *mut u32).cast(),
        (&mut p.groups as *mut u32).cast(), (&mut p.zero_centered as *mut u32).cast(),
        (&mut p.epsilon as *mut f32).cast(), (&mut p.repeat as *mut u32).cast(),
        (&mut p.precision as *mut u32).cast()]) }
}
/// Official fusion shared by FP8 and NVFP4 drafters.
pub(in crate::flash) fn fuse_hidden(b: &Backend<'_>,w: &Weights<'_>,buffers: [u64; 5],count: usize) -> Result<()> {
    let [previous, normalized, hidden, embedding, projected] = buffers;
        // VADE_FLASHNEXT_DRAFT_MX8_LAYER=1: proposal widths 1..=4 (M 1..4 and 4*M <= 16 HC streams)
        // read the fc twins through the latency entries; wider widths keep the official BF16.
        let (fc_embedding, fc_hidden) = (w.device_address("mtp.fc_embedding.weight")?, w.device_address("mtp.fc_hidden.weight")?);
        let mx8 = (1..=4).contains(&count) && super::super::blocks::draft_mx8_layer()
            && super::super::blocks::mx8_has(fc_embedding, H, H) && super::super::blocks::mx8_has(fc_hidden, H, H);
        normalize(b, embedding, w.device_address("mtp.pre_fc_norm_embedding.weight")?,
            projected, H, count)?;
        if mx8 {
            super::super::blocks::next_rows_strided(b, fc_embedding, projected, embedding, H, H, count, H)?;
        } else {
            bf16_rows(b, fc_embedding, projected, embedding, H, H, count, H)?;
        }
        normalize(b, previous, w.device_address("mtp.pre_fc_norm_hidden.weight")?,
            normalized, W, count)?;
        if mx8 {
            super::super::blocks::next_rows_strided(b, fc_hidden, normalized, hidden, H, H, count * 4, H)?;
        } else {
            bf16_rows(b, fc_hidden, normalized, hidden, H, H, count * 4, H)?;
        }
        add_rows(b, hidden, embedding, W, count, 4)?;
    Ok(())
}
pub(super) struct Draft<'a, 'w> {
    // Drop captured executables before any of their activation/cache owners.
    work: Option<DraftWork<'a>>,
    last: DeviceVector<'a>,
    previous: DeviceVector<'a>,
    embedding: DeviceVector<'a>,
    normalized: DeviceVector<'a>,
    projected: DeviceVector<'a>,
    hidden: DeviceVector<'a>,
    input: DeviceVector<'a>,
    hc: HcScratch<'a>,
    mlp: Moe<'w>,
    attention: Qsa<'a, 'w>,
    scratch: QsaScratch<'a>,
    norms: Option<Vec<f32>>,
    ready: bool,
}
struct DraftWork<'a> {
    graphs: [Option<DecodeGraph<'a>>; 3],
    moe: MoeScratch<'a>,
    descriptors: DeviceVector<'a>,
    attention: QsaScratch<'a>,
}
impl<'a, 'w> Draft<'a, 'w> {
    pub(super) fn new(b: &Backend<'a>, w: &'w Weights<'a>, capacity: usize, activation: bool) -> Result<Self> {
        let mut v = b.vector_arena(&[W, TILE * W, TILE * H, TILE * W, TILE * H, TILE * W, TILE * H])?.into_iter();
        let mlp = Moe::load_prefix(w, "mtp.layers.0.mlp", activation)?;
        Ok(Self { last: v.next().unwrap(), previous: v.next().unwrap(), embedding: v.next().unwrap(),
            normalized: v.next().unwrap(), projected: v.next().unwrap(), hidden: v.next().unwrap(), input: v.next().unwrap(),
            hc: HcScratch::with_tokens(b, TILE)?, mlp,
            attention: Qsa::load_prefix(b, w, PREFIX, 48, capacity, None)?,
            scratch: QsaScratch::cache_only(b, &w.plan, TILE, capacity)?, norms: None, work: None, ready: false })
    }
    pub(super) fn reserve(&mut self, b: &Backend<'a>, plan: &RankPlan, capacity: usize) -> Result<()> {
        self.attention.reserve(b, capacity)?;
        self.scratch = QsaScratch::cache_only(b, plan, TILE, capacity)?;
        if let Some(norms) = &self.norms { self.scratch.set_norms(b, norms)?; }
        Ok(())
    }
    /// Norm spans all four streams; the shared hidden projection is applied
    /// separately to each stream, then the projected embedding is broadcast.
    fn fuse(&mut self, b: &Backend<'_>, w: &Weights<'_>, embedding: &[u8], count: usize) -> Result<()> {
        let values = embedding.as_chunks::<2>().0.iter().map(|v|
            f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16)).collect::<Vec<_>>();
        if count == 0 || count > TILE || values.len() != count * H { return Err("invalid MTP embedding rows".into()); }
        self.embedding.resize_within_capacity(count * H)?;
        self.input.resize_within_capacity(count * H)?;
        b.write_vector(&mut self.embedding, &values)?;
        self.fuse_device(b, w, count)
    }
    fn fuse_device(&mut self, b: &Backend<'_>, w: &Weights<'_>, count: usize) -> Result<()> {
        fuse_hidden(b,w,[self.previous.device_address(),self.normalized.device_address(),self.hidden.device_address(),
            self.embedding.device_address(),self.projected.device_address()],count)?;
        Hc::load(w, "mtp.layers.0.attn_hyper_connection")?.mix_tiled(b, &mut self.hc, &self.hidden, &self.input, count)
    }
    pub(super) fn drop_graphs(&mut self) {
        if let Some(work) = &mut self.work { work.graphs = std::array::from_fn(|_| None); }
    }
    fn segment(&mut self, b: &Backend<'a>, key: usize, graphs: bool,
        run: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        if !graphs || b.is_profiling() { return run(self); }
        if self.work.as_ref().unwrap().graphs[key].is_none() {
            // The workspace owns every captured address until release/drop_graphs.
            let graph = unsafe { b.capture_batch_bounded(DecodeGraph::node_limit(1), || run(self))? };
            self.work.as_mut().unwrap().graphs[key] = Some(graph);
        }
        unsafe { b.replay_decode(self.work.as_ref().unwrap().graphs[key].as_ref().unwrap()) }
    }
}

pub(super) struct SavedDraft<'a> {
    last: DeviceVector<'a>,
    attention: QsaState<'a>,
    ready: bool,
}
impl SavedDraft<'_> {
    pub(super) fn bytes(&self) -> usize { self.last.len() * 4 + self.attention.device_bytes() }
}
impl<'a, 'w> Draft<'a, 'w> {
    pub(super) fn snapshot(&self, b: &Backend<'a>, position: usize) -> Result<SavedDraft<'a>> {
        let mut last = b.vector(W)?;
        b.copy_range(&self.last, &mut last, 0, 0, W)?;
        Ok(SavedDraft { last, attention: self.attention.snapshot_prefix(b, position.saturating_sub(1))?, ready: self.ready })
    }
    pub(super) fn restore(&mut self, b: &Backend<'a>, saved: &SavedDraft<'a>) -> Result<()> {
        b.copy_range(&saved.last, &mut self.last, 0, 0, W)?;
        self.attention.restore(b, &saved.attention)?;
        self.ready = saved.ready;
        Ok(())
    }
    pub(super) fn cache_bytes(&self) -> usize { self.attention.kv_bytes() + self.attention.index_bytes() }
    pub(super) fn scratch_bytes(&self) -> usize {
        (W + TILE * (3 * W + 3 * H)) * 4 + self.hc.device_bytes() + self.scratch.device_bytes()
            + self.work.as_ref().map_or(0, |w| w.moe.device_bytes() + w.descriptors.len() * 4 + w.attention.device_bytes())
    }
    pub(super) fn snapshot_bytes(&self) -> usize { W * 4 + self.cache_bytes() }
}

pub(super) struct Verification<'a> {
    pub entry: usize,
    pub tokens: Vec<u32>,
    pub saved: Vec<Recurrent<'a>>,
    pub hidden: Vec<DeviceVector<'a>>,
    pub embeddings: Vec<Vec<u8>>,
}
impl<'a, 'w, E: Exchange> Session<'a, 'w, E> {
    /// Serving creates plain sessions and funds one history owner. Enabling is
    /// safe only before the first target row: omitted residuals cannot be
    /// reconstructed from target KV after a contended turn has advanced.
    pub fn enable_mtp_history(&mut self, b: &Backend<'a>) -> Result<bool> {
        if self.mtp_enabled { return Ok(!self.mtp_failed); }
        if self.model.mtp_depth == 0 || self.mtp_failed || self.position != 0 || !self.valid {
            return Ok(false);
        }
        let allocated = (|| -> Result<()> {
            self.resize_decode_group(b, self.model.mtp_depth + 1)?;
            self.mtp = self.model.persistent.iter().map(|w|
                Draft::new(b, w, self.capacity, self.model.activation_quantization)).collect::<Result<_>>()?;
            Ok(())
        })();
        b.synchronize()?;
        let world = self.model.persistent[0].plan.ranks as usize;
        let failed = parallel(&mut self.ranks, vec![vec![if allocated.is_err() { 1.0 } else { 0.0 }]; self.model.persistent.len()],
            |rank, mut flag| { rank.reduce(&mut flag, &mut vec![0.0; world])?; Ok(flag) })?
            .iter().any(|flag| flag[0] != 0.0);
        if failed {
            self.disable_mtp_history(b)?;
            self.mtp_failed = true;
            engine_model::events::note_line("mtp_fallback_plain", &format!("flash_mtp fallback=plain rank={} reason={}", self.model.persistent[0].plan.rank,
                allocated.err().unwrap_or_else(|| "peer history allocation failed".into())));
            return Ok(false);
        }
        self.mtp_enabled = true;
        Ok(true)
    }
    /// Lightweight MTP history for multi-chat cohorts: keeps `group = 1`
    /// while recording target hidden states and MTP attention KV.
    pub fn enable_mtp_history_batch(&mut self, b: &Backend<'a>) -> Result<bool> {
        if self.mtp_enabled { return Ok(!self.mtp_failed); }
        if self.model.mtp_depth == 0 || self.mtp_failed || self.position != 0 || !self.valid {
            return Ok(false);
        }
        let allocated = (|| -> Result<()> {
            self.mtp = self.model.persistent.iter().map(|w|
                Draft::new(b, w, self.capacity, self.model.activation_quantization)).collect::<Result<_>>()?;
            Ok(())
        })();
        b.synchronize()?;
        let world = self.model.persistent[0].plan.ranks as usize;
        let failed = parallel(&mut self.ranks, vec![vec![if allocated.is_err() { 1.0 } else { 0.0 }]; self.model.persistent.len()],
            |rank, mut flag| { rank.reduce(&mut flag, &mut vec![0.0; world])?; Ok(flag) })?
            .iter().any(|flag| flag[0] != 0.0);
        if failed {
            self.disable_mtp_history(b)?;
            self.mtp_failed = true;
            engine_model::events::note_line("mtp_fallback_plain", &format!("flash_mtp fallback=plain rank={} reason={}", self.model.persistent[0].plan.rank,
                allocated.err().unwrap_or_else(|| "peer batch history allocation failed".into())));
            return Ok(false);
        }
        self.mtp_enabled = true;
        Ok(true)
    }
    /// Whether solo verification logits are retained on this session.
    pub fn has_verify_logits(&self) -> bool { !self.verify_logits.is_empty() }
    /// Physical ownership, including a failed proposal allocation awaiting
    /// cleanup; failure must not hide buffers from the service's release path.
    pub fn has_mtp_history(&self) -> bool { self.mtp_enabled }
    /// A contended request stays plain through retirement. Its main recurrence,
    /// pending target choice and cached target prefix are unchanged. A later
    /// cold single-chat request may acquire the shared drafter anew.
    pub fn disable_mtp_history(&mut self, b: &Backend<'a>) -> Result<bool> {
        if self.verification.is_some() { return Ok(false); }
        if !self.mtp_enabled && self.mtp.is_empty() && self.group == 1 { return Ok(true); }
        self.release_mtp_workspace(b)?;
        b.synchronize()?;
        self.mtp_enabled = false;
        self.mtp.clear();
        // Unshared checkpoints belong solely to this request. Shared cache
        // checkpoints remain immutable and carry their existing cache charge.
        if let Some(saved) = self.canonical.as_mut().and_then(Rc::get_mut) { saved.mtp.clear(); }
        self.resize_decode_group(b, 1)?;
        Ok(true)
    }
    /// Replace only decode execution buffers. Target KV, recurrence, output
    /// logits and prompt scratch are independent of verification row capacity.
    fn resize_decode_group(&mut self, b: &Backend<'a>, group: usize) -> Result<()> {
        if self.group == group { return Ok(()); }
        if !self.gdn_group.is_empty() { return Err("Flash decode resize during shared prefill".into()); }
        b.synchronize()?;
        self.drop_graphs();
        let local = self.ranks.len();
        let front = (0..local).map(|_| FrontScratch::with_tokens(b, group)).collect::<Result<Vec<_>>>()?;
        let routing = (0..local).map(|_| Fp8RoutingFlag::new(b, group)).collect::<Result<Vec<_>>>()?;
        let mut hc = Vec::new();
        let mut moe = Vec::new();
        let mut input = Vec::new();
        let mut hidden = Vec::new();
        let mut exchange = Vec::new();
        for _ in 0..self.slots * local {
            hc.push(HcScratch::with_tokens(b, group)?);
            let mut scratch = MoeScratch::with_capacity(b, group)?;
            scratch.reserve_grouped(b, group, self.model.main_activation_quantization(),
                self.model.persistent[0].plan.expert_output_rows.as_ref().map(|_| self.model.persistent[0].plan.ranks as usize))?;
            moe.push(scratch);
            let (input_rows, hidden_rows) = execution_rows(b, group)?;
            input.push(input_rows);
            hidden.push(hidden_rows);
            exchange.push(b.exchange_slot(group * PARTIALS)?);
        }
        self.front = front;
        self.routing = routing;
        self.hc = hc;
        self.moe = moe;
        self.input = input;
        self.hidden = hidden;
        self.exchange = exchange;
        self.group = group;
        self.graphs = (0..super::forward::decode_graph_slots(self.model, group)).map(|_| None).collect();
        self.qsa_shapes = vec![[0; 3]; group];
        Ok(())
    }
    pub fn mtp_depth(&self) -> usize {
        if !self.mtp_failed && self.mtp.len() == self.ranks.len() && self.mtp.iter().all(|d| d.ready) { self.model.mtp_depth } else { 0 }
    }
    /// Only one serving session owns proposal/verification workspace. Committed
    /// history survives workspace growth; contention discards it separately.
    /// Pending verification pins the owner until its own accepted-prefix commit.
    pub fn release_mtp_workspace(&mut self, b: &Backend<'a>) -> Result<bool> {
        if self.verification.is_some() { return Ok(false); }
        if self.model.mtp_depth == 0 { return Ok(true); }
        if !self.mtp_workspace && self.mtp.iter().all(|draft| draft.work.is_none())
            && self.rollback.is_empty() && self.verify_logits.is_empty()
            && self.gdn_verify.iter().all(Option::is_none) { return Ok(true); }
        b.synchronize()?;
        self.drop_graphs();
        self.mtp_workspace = false;
        for draft in &mut self.mtp { draft.work = None; }
        self.rollback.clear();
        self.verify_logits.clear();
        self.verify_staging.clear();
        for scratch in &mut self.gdn_verify { *scratch = None; }
        Ok(true)
    }
    /// Try every optional device allocation before entering draft/verify kernels.
    /// A failed rank still votes; all ranks discard the workspace and stay plain.
    #[allow(clippy::possible_missing_comma)] // one element written over two lines.
    pub fn prepare_mtp_workspace(&mut self, b: &Backend<'a>) -> Result<bool> {
        if self.mtp_workspace { return Ok(true); }
        if self.mtp_depth() == 0 { return Ok(false); }
        self.mtp_norms(b)?;
        self.drop_graphs();
        let allocated = (|| -> Result<()> {
            if self.group < self.model.mtp_depth + 1 {
                self.resize_decode_group(b, self.model.mtp_depth + 1)?;
            }
            for (r, d) in self.mtp.iter_mut().enumerate() {
                let plan = &self.model.persistent[r].plan;
                let mut attention = QsaScratch::new(b, plan, 1, self.capacity)?;
                attention.set_norms(b, d.norms.as_ref().ok_or("missing MTP norms")?)?;
                let mut moe = MoeScratch::new(b)?;
                moe.reserve_grouped(b, 1, self.model.activation_quantization, None)?;
                d.work = Some(DraftWork { graphs: std::array::from_fn(|_| None), moe, descriptors: d.mlp.device_descriptors(b)?, attention });
                self.gdn_verify[r] = Some(GdnScratch::shared_projections(b, H,
                    (plan.value_heads.end - plan.value_heads.start) as usize,
                    (plan.key_heads.end - plan.key_heads.start) as usize,
                    self.model.mtp_depth + 1,
                )?);
                self.verify_logits.push(
                    b.vector_arena(&[(self.model.mtp_depth + 1)
                        * (plan.vocabulary.end - plan.vocabulary.start) as usize])?
                        .remove(0),
                );
                let rows = self.model.mtp_depth + 1;
                self.verify_staging.push(b.vector_arena(&[rows * (EMBEDDING_BYTES + LEXICAL_BYTES) / 4,
                    rows * 17 * std::mem::size_of::<spark_gpu_probe::kernel_params::HybridCopyRow>() / 4,
                    rows * (plan.vocabulary.end - plan.vocabulary.start).div_ceil(1024) as usize * 3, rows * 3])?);
                // A one-token prompt may not have grown target expert scratch.
                self.moe[r].reserve_grouped(b, self.model.mtp_depth + 1, self.model.main_activation_quantization(),
                    plan.expert_output_rows.as_ref().map(|_| plan.ranks as usize))?;
            }
            for _ in 0..self.model.mtp_depth {
                self.rollback.push(Recurrent { position: 0, rows: Vec::new(), mtp: Vec::new(),
                    ple: self.ple.iter().map(|p| p.save_history(b)).collect::<Result<_>>()?,
                    gdn: self.gdn_cache.iter().map(|layers| layers.iter().map(|v|
                        v.as_ref().map(|v| v.snapshot(b)).transpose()).collect()).collect::<Result<_>>()? });
            }
            Ok(())
        })();
        b.synchronize()?;
        let world = self.model.persistent[0].plan.ranks as usize;
        let failed = parallel(&mut self.ranks, vec![vec![if allocated.is_err() { 1.0 } else { 0.0 }]; self.mtp.len()],
            |rank, mut flag| { rank.reduce(&mut flag, &mut vec![0.0; world])?; Ok(flag) })?
            .iter().any(|flag| flag[0] != 0.0);
        if failed {
            self.disable_mtp_history(b)?;
            self.mtp_failed = true;
            engine_model::events::note_line("mtp_fallback_plain", &format!("flash_mtp fallback=plain rank={} device_bytes={} reason={}",
                self.model.persistent[0].plan.rank, b.allocated_bytes(),
                allocated.err().unwrap_or_else(|| "peer workspace allocation failed".into())));
            return Ok(false);
        }
        self.mtp_workspace = true;
        Ok(true)
    }
    fn mtp_norms(&mut self, b: &Backend<'a>) -> Result<()> {
        if self.mtp.is_empty() || self.mtp.iter().all(|d| d.norms.is_some()) { return Ok(()); }
        let norms = self.gather_qsa(self.model.persistent.iter()
            .map(|w| QsaGather::norms_prefix(w, PREFIX)).collect::<Result<_>>()?)?;
        for (draft, norms) in self.mtp.iter_mut().zip(norms) {
            draft.scratch.set_norms(b, &norms)?;
            draft.norms = Some(norms);
        }
        Ok(())
    }
    /// Shift the true target residual by one token; speculative residuals never
    /// enter committed draft history. Only attention KV is needed for catchup.
    pub(super) fn observe_mtp(&mut self, b: &Backend<'a>, entry: usize, count: usize,
        hidden: &[DeviceVector<'a>], embedding: &[Vec<u8>], offset: usize) -> Result<()> {
        if self.mtp.is_empty() || count == 0 { return Ok(()); }
        self.mtp_norms(b)?;
        for start in (0..count).step_by(TILE) {
            let n = TILE.min(count - start);
            let skip = usize::from(entry + start == 0);
            let rows = n - skip;
            for (r, d) in self.mtp.iter_mut().enumerate() {
                if rows != 0 {
                    if skip == 0 { b.copy_range(&d.last, &mut d.previous, 0, 0, W)?; }
                    if n > 1 { b.copy_range(&hidden[r], &mut d.previous, (offset + start) * W, (1 - skip) * W, (n - 1) * W)?; }
                    d.attention.rewind((entry + start).saturating_sub(1));
                    d.fuse(b, &self.model.persistent[r], &embedding[r][(offset + start + skip) * EMBEDDING_BYTES..(offset + start + n) * EMBEDDING_BYTES], rows)?;
                    d.attention.project_cache(b, &mut d.scratch, &d.input, rows)?;
                    d.attention.observe(b, &mut d.scratch)?;
                }
                b.copy_range(&hidden[r], &mut d.last, (offset + start + n - 1) * W, 0, W)?;
                d.ready = true;
            }
        }
        Ok(())
    }
    pub fn propose_mtp(&mut self, b: &Backend<'a>, token: u32, depth: usize) -> Result<Vec<u32>> {
        self.propose_mtp_with(b, token, depth, true, |s, b, _| Ok(s.collect_output(b, false)?.0))
    }
    pub fn propose_mtp_sampled(&mut self, b: &Backend<'a>, token: u32, depth: usize, k: usize,
        mut choose: impl FnMut(SampleRow, usize) -> Result<Option<u32>>) -> Result<Vec<u32>> {
        self.propose_mtp_with(b, token, depth, false, |s, b, i| {
            let row = s.collect_sample_row(b, k)?;
            let local = if s.model.persistent.iter().any(|w| w.plan.rank == 0) { choose(row, i)? } else { None };
            if let Some(token) = s.draft_choice(local)? { return Ok(token); }
            let row = s.collect_sample_row(b, 0)?;
            let local = if s.model.persistent.iter().any(|w| w.plan.rank == 0) { choose(row, i)? } else { None };
            s.draft_choice(local)?.ok_or_else(|| "missing sampled MTP choice".into())
        })
    }
    fn propose_mtp_with(&mut self, b: &Backend<'a>, mut token: u32, depth: usize, argmax_only: bool,
        mut choose: impl FnMut(&mut Self, &Backend<'a>, usize) -> Result<u32>) -> Result<Vec<u32>> {
        self.commit_verification(b, self.position)?;
        if self.mtp_failed { return Ok(Vec::new()); }
        if token as usize >= VOCABULARY || !self.valid || depth == 0 || depth > self.mtp_depth() || self.position + depth + 1 > self.capacity {
            return Err("Flash MTP proposal exceeds ready history, depth or context".into());
        }
        if !self.prepare_mtp_workspace(b)? { return Ok(Vec::new()); }
        for (r, d) in self.mtp.iter_mut().enumerate() {
            self.routing[r].clear(b, 1, self.model.persistent[r].plan.rank as usize)?;
            d.work.as_mut().unwrap().moe.set_routing_flag(&self.routing[r])?;
        }
        let mut output = Vec::with_capacity(depth);
        let graphs = self.ranks.len() == 1 && self.model.resident;
        for i in 0..depth {
            self.draft_embedding(b, token)?;
            for (r, d) in self.mtp.iter_mut().enumerate() {
                let w = &self.model.persistent[r];
                if i == 0 {
                    d.attention.rewind(self.position - 1);
                    b.copy_range(&d.last, &mut d.previous, 0, 0, W)?;
                } else { b.copy_range(&d.hidden, &mut d.previous, 0, 0, W)?; }
                d.embedding.resize_within_capacity(H)?;
                d.input.resize_within_capacity(H)?;
                d.attention.prepare_project(&mut d.work.as_mut().unwrap().attention, &d.input, 1)?;
                d.segment(b, 0, graphs, |d| {
                    let (mut raw, mut out, mut width) = (d.projected.device_address(), d.embedding.device_address(), H as u32);
                    // Only the first H threads run: the existing bit-expansion kernel
                    // converts one BF16 row, without its three residual replicas.
                    unsafe { b.launch_flash("flash_embedding_expand", H.div_ceil(128) as u32, 128,
                        &mut [(&mut raw as *mut u64).cast(), (&mut out as *mut u64).cast(), (&mut width as *mut u32).cast()])?; }
                    d.fuse_device(b, w, 1)?;
                    d.attention.project(b, &mut d.work.as_mut().unwrap().attention, &d.input, 1)
                })?;
                d.attention.replay_projected(1);
                d.attention.attend(b, &mut d.work.as_mut().unwrap().attention)?;
            }
            reduce_device(b, &mut self.ranks, &mut self.communication, None,
                &mut self.mtp.iter_mut().map(|d| d.work.as_mut().unwrap().attention.output_mut()).collect::<Vec<_>>())?;
            for (r, d) in self.mtp.iter_mut().enumerate() {
                d.segment(b, 1, graphs, |d| {
                    Hc::load(&self.model.persistent[r], "mtp.layers.0.attn_hyper_connection")?
                        .combine_tiled(b, &mut d.hc, &mut d.hidden, d.work.as_ref().unwrap().attention.output(), 1)?;
                    let work = d.work.as_mut().unwrap();
                    d.mlp.project_batch(b, &mut work.moe, &d.hidden, 1, Some(&work.descriptors))
                })?;
            }
            reduce_device(b, &mut self.ranks, &mut self.communication, None,
                &mut self.mtp.iter_mut().map(|d| d.work.as_mut().unwrap().moe.partials_mut()).collect::<Vec<_>>())?;
            for (r, d) in self.mtp.iter_mut().enumerate() {
                d.segment(b, 2, graphs, |d| {
                    d.mlp.finish_batch(b, &mut d.work.as_mut().unwrap().moe, &mut d.hidden, 1)?;
                    Hc::mixer(&self.model.persistent[r], "mtp.hyper_connection_mixer")?
                        .mix_tiled(b, &mut d.hc, &d.hidden, &d.input, 1)
                })?;
                if argmax_only {
                    self.head[r].project_draft_argmax(b, &d.input)?;
                } else {
                    self.head[r].project_draft(b, &d.input)?;
                }
            }
            token = choose(self, b, i)?;
            output.push(token);
        }
        Ok(output)
    }
    /// Copy opaque embedding words through mapped RDMA slots; no host conversion
    /// or stream upload enters the distributed draft segments.
    fn draft_embedding(&mut self, b: &Backend<'a>, token: u32) -> Result<()> {
        let mut embedding = Vec::new();
        for (r, d) in self.mtp.iter_mut().enumerate() {
            let w = &self.model.persistent[r];
            let owns = w.plan.vocabulary.contains(&(token as u64));
            let offset = (token as u64).saturating_sub(w.plan.vocabulary.start) as usize * EMBEDDING_BYTES;
            let mut offsets = vec![0];
            for rank in 0..w.plan.ranks {
                let owner = partition(VOCABULARY as u64, rank, w.plan.ranks).map_err(|e| e.to_string())?.contains(&(token as u64));
                offsets.push(offsets.last().unwrap() + if owner { EMBEDDING_BYTES } else { 0 });
            }
            if let (1, Some(mailbox)) = (self.ranks.len(), self.mailbox.as_deref()) {
                let raw = d.projected.device_address();
                if owns { copy_words(b, w.row_address("model.language_model.embed_tokens.weight", offset, EMBEDDING_BYTES)?, raw, EMBEDDING_BYTES)?; }
                let fill = |at: usize, slot: &mut [u8]| queue_consumer(b, ||
                    copy_words(b, raw + at as u64, mailbox.device_address(slot.as_ptr())?, slot.len()));
                let receive = |at: usize, peer: &[u8]| queue_consumer(b, ||
                    copy_words(b, mailbox.device_address(peer.as_ptr())?, raw + at as u64, peer.len()));
                if w.plan.is_pair() {
                    self.ranks[r].exchange_slots_pipeline(&[(if owns { EMBEDDING_BYTES } else { 0 }, if owns { 0 } else { EMBEDDING_BYTES })],
                        |_, slot| fill(0, slot), |_, peer| receive(0, peer), |event| b.wait_fence(event))?;
                } else {
                    self.ranks[r].gather_slots_pipeline(&[offsets], |_, at, slot| fill(at, slot),
                        |_, _, at, peer| receive(at, peer), |event| b.wait_fence(event))?;
                }
            } else {
                let mut bytes = vec![0u8; EMBEDDING_BYTES];
                if owns { w.read("model.language_model.embed_tokens.weight", offset, &mut bytes)?; }
                embedding.push((bytes, offsets));
            }
        }
        if !embedding.is_empty() {
            let embedding = parallel(&mut self.ranks, embedding, |rank, (mut bytes, offsets)| {
                rank.gather_bytes(&mut bytes, &offsets)?; Ok((bytes, offsets))
            })?;
            for (d, (bytes, _)) in self.mtp.iter_mut().zip(embedding) { b.write_vector_bytes(&mut d.projected, &bytes)?; }
        }
        Ok(())
    }
    pub fn position(&self) -> usize { self.position }
    pub fn verify_mtp(&mut self, b: &Backend<'a>, tokens: &[u32], logits: bool) -> Result<Vec<Output>> {
        if tokens.len() == 1 { return self.forward(b, tokens, logits).map(|v| vec![v]); }
        let started = Instant::now();
        let mut output = self.verify_mtp_rows(b, tokens, |s, b, count| {
            if !logits { return s.collect_verify_max(b, count); }
            s.collect_verify_rows(b, count, |s, b| {
                let (token, logits) = s.collect_output(b, true)?;
                Ok(Output { token, logits, elapsed_ms: 0.0, load_ms: 0.0 })
            })
        })?;
        let ms = started.elapsed().as_secs_f64() * 1000.0 / tokens.len() as f64;
        for row in &mut output { row.elapsed_ms = ms; }
        Ok(output)
    }
    pub fn verify_mtp_sampled(&mut self, b: &Backend<'a>, tokens: &[u32], k: usize) -> Result<Vec<SampleRow>> {
        if tokens.len() == 1 {
            self.forward(b, tokens, false)?;
            return self.collect_sample_row(b, k).map(|v| vec![v]);
        }
        self.verify_mtp_rows(b, tokens, |s, b, count| s.collect_verify_rows(b, count, |s, b| s.collect_sample_row(b, k)))
    }
    pub fn decode_logits(&mut self, b: &Backend<'a>, rows: usize) -> Result<Vec<Vec<f32>>> {
        if rows == 1 && self.verification.is_none() {
            return Ok(vec![self.collect_output(b, true)?.1.ok_or("missing Flash decode logits")?]);
        }
        if !self.valid || self.verification.as_ref().map(|v| v.tokens.len()) != Some(rows) {
            return Err("missing Flash retained verification rows".into());
        }
        let mut output = Vec::with_capacity(rows);
        for row in 0..rows {
            for r in 0..self.ranks.len() { self.head[r].select_row(b, &self.verify_logits[r], row)?; }
            output.push(self.collect_output(b, true)?.1.ok_or("missing Flash verification logits")?);
        }
        Ok(output)
    }
    fn collect_verify_rows<T>(&mut self, b: &Backend<'a>, count: usize,
        mut collect: impl FnMut(&mut Self, &Backend<'a>) -> Result<T>) -> Result<Vec<T>> {
        (0..count).map(|row| {
            for r in 0..self.ranks.len() { self.head[r].select_row(b, &self.verify_logits[r], row)?; }
            collect(self, b)
        }).collect()
    }
    fn collect_verify_max(&mut self, b: &Backend<'a>, count: usize) -> Result<Vec<Output>> {
        let offsets = (0..=self.model.persistent[0].plan.ranks as usize).map(|r| r * count * 2).collect::<Vec<_>>();
        let mut wires = Vec::new();
        for r in 0..self.ranks.len() {
            let range = self.head[r].vocabulary();
            let (_, scratch) = self.verify_staging[r].split_at_mut(2);
            let (partials, result) = scratch.split_at_mut(1);
            b.enqueue_argmax_batch_device(&self.verify_logits[r].slice(0..count * range.len())?,
                &mut partials[0].slice(0..count * range.len().div_ceil(1024) * 3)?, &mut result[0].slice(0..count * 3)?, count)?;
            let values = b.read_argmax_raw(&result[0], count, |v| Ok(v.to_vec()))?;
            let failed = self.routing[r].failed_after_readback()?;
            let rank = self.model.persistent[r].plan.rank as usize;
            let mut wire = vec![0.0; *offsets.last().unwrap()];
            for (row, v) in values.as_chunks::<3>().0.iter().enumerate() {
                let token = if failed { u32::MAX } else if v[2] != 1.0 || v.iter().any(|x| !x.is_finite()) { NONFINITE_LOGIT }
                    else { range.start + v[1] as u32 };
                wire[offsets[rank] + row * 2] = f32::from_bits(token);
                wire[offsets[rank] + row * 2 + 1] = if failed || token == NONFINITE_LOGIT { 0.0 } else { v[0] };
            }
            wires.push(wire);
        }
        let wires = parallel(&mut self.ranks, wires, |rank, mut wire| { rank.gather_f32(&mut wire, &offsets)?; Ok(wire) })?;
        let winners = wires.iter().map(|wire| (0..count).map(|row| Candidate::reduce(offsets[..offsets.len()-1].iter()
            .map(|&at| Candidate { token: wire[at + row * 2].to_bits(), logit: wire[at + row * 2 + 1] })).map(|c| c.token))
            .collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
        if winners.windows(2).any(|v| v[0] != v[1]) { return Err("local ranks reduced different Flash winners".into()); }
        for r in 0..self.ranks.len() { self.head[r].select_row(b, &self.verify_logits[r], count - 1)?; }
        b.synchronize()?;
        Ok(winners[0].iter().map(|&token| Output { token, logits: None, elapsed_ms: 0.0, load_ms: 0.0 }).collect())
    }
    fn verify_mtp_rows<T>(&mut self, b: &Backend<'a>, tokens: &[u32],
        mut collect: impl FnMut(&mut Self, &Backend<'a>, usize) -> Result<Vec<T>>) -> Result<Vec<T>> {
        self.commit_verification(b, self.position)?;
        if tokens.is_empty() || tokens.len() > self.model.mtp_depth + 1 || self.model.mtp_depth == 0
            || self.position == 0 || self.position + tokens.len() > self.capacity { return Err("invalid Flash verification rows".into()); }
        if !self.mtp_workspace { return Err("Flash verification needs an agreed draft workspace".into()); }
        let entry = self.position;
        let local = self.ranks.len();
        let mut saved = std::mem::take(&mut self.rollback);
        for (i, prefix) in saved.iter_mut().enumerate() { prefix.position = self.position + i + 1; prefix.rows.clear(); }
        self.verification = Some(Verification { entry: self.position, tokens: tokens.to_vec(), saved,
            hidden: Vec::new(), embeddings: vec![Vec::new(); local] });
        self.verifying = true;
        let profile = self.model.mtp_depth == 2 && tokens.len() == 3;
        if profile { b.decode_profile_begin(1, self.model.persistent[0].plan.rank as usize)?; }
        b.set_spin_waits(true);
        let result = (|| {
            self.forward_tile(b, &[tokens], false, true, None, None, None)?;
            for r in 0..local {
                Hc::mixer(&self.model.persistent[r], MIXER)?.mix_tiled(b, &mut self.hc[r], &self.hidden[r], &self.input[r], tokens.len())?;
                self.head[r].project_rows(b, &self.input[r], &mut self.verify_logits[r], tokens.len())?;
            }
            collect(self, b, tokens.len())
        })();
        self.verifying = false;
        if profile { b.decode_profile_end(result.is_ok())?; }
        b.set_spin_waits(false);
        if result.is_err() {
            crate::flash_verification::reject(&mut self.verification, &mut self.position, &mut self.valid, entry);
        }
        result
    }
    pub fn commit_verification(&mut self, b: &Backend<'a>, position: usize) -> Result<()> {
        let verification = self.verification.take();
        let kept = match crate::flash_verification::committed_rows(
            verification.as_ref().map(|v| (v.entry, v.tokens.len())), self.position, self.valid, position) {
            Ok(kept) => kept,
            Err(error) => { if verification.is_some() { self.valid = false; } return Err(error); }
        };
        let Some(v) = verification else { return Ok(()); };
        if kept < v.tokens.len() { self.restore_recurrent(b, &v.saved[kept - 1])?; }
        self.observe_mtp(b, v.entry, kept, &v.hidden, &v.embeddings, 0)?;
        self.position = position;
        if self.caching { self.prefill_tail.extend_from_slice(&v.tokens[..kept]); }
        self.rollback = v.saved;
        Ok(())
    }

    /// Batched MTP proposals for 1..=8 concurrent sessions sharing one MTP layer
    /// and one MXFP8 draft-head pass per depth step.
    pub fn propose_mtp_batch(b: &Backend<'a>, batch: &mut DecodeBatch<'a, 'w>, sessions: &mut [&mut Self],
        next_tokens: &[u32], limits: &[usize]) -> Result<Vec<Vec<u32>>> {
        let n = sessions.len();
        if n == 0 || n > MAX_MTP_BATCH || next_tokens.len() != n || limits.len() != n {
            return Err("invalid Flash MTP batch shape".into());
        }
        let model = batch.model;
        if sessions.iter().any(|s| !std::ptr::eq(s.model, model) || s.head.len() != 1) {
            return Err("Flash MTP batch needs independent sessions of one distributed model".into());
        }
        for s in sessions.iter_mut() { s.commit_verification(b, s.position)?; }
        let mut eff_limits = vec![0usize; n];
        for i in 0..n {
            if sessions[i].mtp_failed || sessions[i].mtp_depth() == 0 || limits[i] == 0 { continue; }
            if next_tokens[i] as usize >= VOCABULARY || !sessions[i].valid
                || limits[i] > sessions[i].mtp_depth()
                || sessions[i].position == 0
                || sessions[i].position + limits[i] + 1 > sessions[i].capacity {
                return Err("Flash MTP batch proposal exceeds ready history, depth or context".into());
            }
            sessions[i].mtp_norms(b)?;
            eff_limits[i] = limits[i];
        }
        let max_depth = eff_limits.iter().copied().max().unwrap_or(0);
        if max_depth == 0 { return Ok(vec![Vec::new(); n]); }
        if batch.mtp.is_none() {
            batch.mtp = Some(MtpBatchWork::new(b, model)?);
        }
        let work = batch.mtp.as_mut().unwrap();
        if work.qsa_norm.is_none() {
            let norms = sessions.iter().find_map(|s| s.mtp.first().and_then(|d| d.norms.as_deref()))
                .ok_or("missing MTP norms")?;
            let mut dev = b.vector_arena(&[norms.len()])?.pop().unwrap();
            b.write_vector(&mut dev, norms)?;
            work.qsa_norm = Some(dev);
        }
        for (i, s) in sessions.iter_mut().enumerate() {
            if eff_limits[i] > 0 { s.mtp[0].attention.rewind(s.position - 1); }
        }
        let w = &model.persistent[0];
        let shard = w.plan.rank as usize;
        let shard_count = w.plan.ranks as usize;
        let range = sessions[0].head[0].vocabulary();
        let mut current = next_tokens.to_vec();
        let mut outputs: Vec<Vec<u32>> = (0..n).map(|i| Vec::with_capacity(eff_limits[i])).collect();
        b.set_spin_waits(true);
        let result: Result<()> = (|| {
            for step in 0..max_depth {
                let active: Vec<usize> = (0..n).filter(|&i| step < eff_limits[i]).collect();
                let count = active.len();
                // Gather raw BF16 embeddings for all active chats in one collective.
                let mut offsets = vec![0usize];
                let mut shard_ranges = Vec::with_capacity(shard_count);
                for r in 0..w.plan.ranks {
                    let sr = partition(VOCABULARY as u64, r, w.plan.ranks).map_err(|e| e.to_string())?;
                    let owned = active.iter().filter(|&&i| sr.contains(&(current[i] as u64))).count();
                    offsets.push(offsets.last().unwrap() + owned * EMBEDDING_BYTES);
                    shard_ranges.push(sr);
                }
                let mut local_count = 0usize;
                for &i in &active {
                    let tok = current[i] as u64;
                    if w.plan.vocabulary.contains(&tok) {
                        let off = (tok - w.plan.vocabulary.start) as usize * EMBEDDING_BYTES;
                        copy_words(b, w.row_address("model.language_model.embed_tokens.weight", off, EMBEDDING_BYTES)?,
                            work.projected.device_address() + (local_count * EMBEDDING_BYTES) as u64, EMBEDDING_BYTES)?;
                        local_count += 1;
                    }
                }
                let mut wire = vec![0u8; *offsets.last().unwrap()];
                if local_count != 0 {
                    let words = b.read_vector_range(&work.projected, 0, local_count * (EMBEDDING_BYTES / 4))?;
                    let dst = &mut wire[offsets[shard]..offsets[shard + 1]];
                    for (chunk, &val) in dst.as_chunks_mut::<4>().0.iter_mut().zip(&words) {
                        *chunk = val.to_bits().to_le_bytes();
                    }
                }
                sessions[0].ranks[0].gather_bytes(&mut wire, &offsets)?;
                let mut cursors = offsets[..shard_count].to_vec();
                let mut ordered = vec![0u8; count * EMBEDDING_BYTES];
                for (slot, &i) in active.iter().enumerate() {
                    let tok = current[i] as u64;
                    let owner = shard_ranges.iter().position(|sr| sr.contains(&tok))
                        .ok_or("MTP token outside vocabulary shards")?;
                    let src = cursors[owner];
                    ordered[slot * EMBEDDING_BYTES..(slot + 1) * EMBEDDING_BYTES]
                        .copy_from_slice(&wire[src..src + EMBEDDING_BYTES]);
                    cursors[owner] += EMBEDDING_BYTES;
                }
                b.write_vector_bytes(&mut work.projected, &ordered)?;
                for slot in 0..count {
                    b.flash_embedding_expand_row(&work.projected, slot * EMBEDDING_BYTES, &mut work.embedding, slot * H, H)?;
                }
                for (slot, &i) in active.iter().enumerate() {
                    if step == 0 {
                        b.copy_range(&sessions[i].mtp[0].last, &mut work.previous, 0, slot * W, W)?;
                    } else {
                        b.copy_range(&sessions[i].mtp[0].hidden, &mut work.previous, 0, slot * W, W)?;
                    }
                }
                work.input.resize_within_capacity(count * H)?;
                fuse_hidden(b, w, [
                    work.previous.device_address(), work.normalized.device_address(),
                    work.hidden.device_address(), work.embedding.device_address(),
                    work.projected.device_address(),
                ], count)?;
                Hc::load(w, "mtp.layers.0.attn_hyper_connection")?.mix_tiled(b, &mut work.hc, &work.hidden, &work.input, count)?;
                let norm_dev = work.qsa_norm.as_ref().unwrap();
                let qsa_rows = active.iter().map(|&i| {
                    sessions[i].qsa[0].use_norm(norm_dev)?;
                    sessions[i].mtp[0].attention.session_row(&sessions[i].qsa[0])
                }).collect::<Result<Vec<_>>>()?;
                work.qsa.upload_sessions(b, &qsa_rows)?;
                sessions[active[0]].mtp[0].attention.project_batch(b, &mut work.qsa, &work.input, count)?;
                sessions[active[0]].mtp[0].attention.attend_batch(b, &work.qsa, count)?;
                sessions[active[0]].mtp[0].attention.finish_batch(b, &work.qsa, &mut work.input, count)?;
                for &i in &active { sessions[i].mtp[0].attention.commit_batch(); }
                reduce_device(b, &mut sessions[0].ranks, &mut sessions[0].communication, None, &mut [&mut work.input])?;
                Hc::load(w, "mtp.layers.0.attn_hyper_connection")?.combine_tiled(b, &mut work.hc, &mut work.hidden, &work.input, count)?;
                batch.routing.clear(b, count, shard)?;
                work.moe.set_routing_flag(&batch.routing)?;
                work.moe.resize(count)?;
                work.mlp.project_batch(b, &mut work.moe, &work.hidden, count, Some(&work.descriptors))?;
                reduce_device(b, &mut sessions[0].ranks, &mut sessions[0].communication, None, &mut [work.moe.partials_mut()])?;
                work.mlp.finish_batch(b, &mut work.moe, &mut work.hidden, count)?;
                for (slot, &i) in active.iter().enumerate() {
                    if step + 1 < eff_limits[i] {
                        b.copy_range(&work.hidden, &mut sessions[i].mtp[0].hidden, slot * W, 0, W)?;
                    }
                }
                Hc::mixer(w, "mtp.hyper_connection_mixer")?.mix_tiled(b, &mut work.hc, &work.hidden, &work.input, count)?;
                sessions[0].head[0].project_draft_rows(b, &work.input, count, &mut batch.logits,
                    &mut work.argmax_partials, &mut work.argmax_result)?;
                let values = b.read_argmax_raw(&work.argmax_result, count, |v| Ok(v.to_vec()))?;
                let cand_offsets = (0..=shard_count).map(|r| r * count * 2).collect::<Vec<_>>();
                let mut cand_wire = vec![0.0f32; *cand_offsets.last().unwrap()];
                for (slot, v) in values.as_chunks::<3>().0.iter().enumerate() {
                    let failed = batch.routing.row_failed_after_readback(slot)?;
                    let tok = if failed { u32::MAX } else if v[2] != 1.0 || v.iter().any(|x| !x.is_finite()) { NONFINITE_LOGIT }
                        else { range.start + v[1] as u32 };
                    cand_wire[cand_offsets[shard] + slot * 2] = f32::from_bits(tok);
                    cand_wire[cand_offsets[shard] + slot * 2 + 1] = if failed || tok == NONFINITE_LOGIT { 0.0 } else { v[0] };
                }
                sessions[0].ranks[0].gather_f32(&mut cand_wire, &cand_offsets)?;
                for (slot, &i) in active.iter().enumerate() {
                    let winner = Candidate::reduce(cand_offsets[..shard_count].iter().map(|&at|
                        Candidate { token: cand_wire[at + slot * 2].to_bits(), logit: cand_wire[at + slot * 2 + 1] }))?.token;
                    current[i] = winner;
                    outputs[i].push(winner);
                }
            }
            Ok(())
        })();
        b.set_spin_waits(false);
        result?;
        Ok(outputs)
    }

    /// Multi-chat greedy verification over 1..=8 concurrent sessions (total rows <= 16).
    /// Shares one dense/MoE/head pass across all verification rows while preserving each
    /// session's exact per-token PLE/GDN/QSA recurrence order and rollback checkpoints.
    #[allow(clippy::needless_range_loop)]
    pub fn verify_batch_greedy(b: &Backend<'a>, batch: &mut DecodeBatch<'a, 'w>, sessions: &mut [&mut Self],
        token_slices: &[&[u32]]) -> Result<Vec<Vec<Output>>> {
        let chats = sessions.len();
        if chats == 0 || chats > MAX_MTP_BATCH || token_slices.len() != chats {
            return Err("invalid Flash verify batch shape".into());
        }
        let model = batch.model;
        let mut row_starts = Vec::with_capacity(chats);
        let mut total_rows = 0usize;
        for (c, s) in sessions.iter().enumerate() {
            let len = token_slices[c].len();
            if !std::ptr::eq(s.model, model) || s.head.len() != 1 || !s.valid
                || len == 0 || len > model.mtp_depth + 1
                || s.position == 0 || s.position + len > s.capacity
                || token_slices[c].iter().any(|&t| t as usize >= VOCABULARY) {
                return Err("Flash verify batch needs valid cached sessions within depth and capacity".into());
            }
            row_starts.push(total_rows);
            total_rows += len;
        }
        if total_rows == 0 || total_rows > batch.capacity || total_rows > sessions[0].tile {
            return Err("Flash verify batch rows exceed decode batch capacity".into());
        }
        for s in sessions.iter_mut() { s.commit_verification(b, s.position)?; }
        let entries: Vec<usize> = sessions.iter().map(|s| s.position).collect();
        for s in sessions.iter_mut() { s.valid = false; }

        let mut all_saved: Vec<Vec<Recurrent<'a>>> = Vec::with_capacity(chats);
        for c in 0..chats {
            let len = token_slices[c].len();
            let mut saved = std::mem::take(&mut sessions[c].rollback);
            let need = len.saturating_sub(1);
            while saved.len() < need {
                saved.push(Recurrent {
                    position: 0, rows: Vec::new(), mtp: Vec::new(),
                    ple: vec![sessions[c].ple[0].save_history(b)?],
                    gdn: vec![sessions[c].gdn_cache[0].iter().map(|v|
                        v.as_ref().map(|v| v.snapshot(b)).transpose()).collect::<Result<_>>()?],
                });
            }
            let before = sessions[c].rows[0].history();
            for (k, &tok) in token_slices[c].iter().enumerate() {
                sessions[c].rows[0].advance(tok)?;
                if k < need {
                    saved[k].position = entries[c] + k + 1;
                    saved[k].rows.clear();
                    saved[k].rows.push(sessions[c].rows[0].history());
                }
            }
            sessions[c].rows[0].restore_history(before);
            all_saved.push(saved);
        }

        let w = &model.persistent[0];
        let shard = w.plan.rank as usize;
        let shard_count = w.plan.ranks as usize;
        let compact = w.plan.expert_output_rows.is_some();
        let layers = model.checkpoint.config.layers.len();
        let segments = if compact { 4 } else { 3 };
        let graphs = model.resident && batch.descriptors.len() == layers;
        let max_len = token_slices.iter().map(|t| t.len()).max().unwrap_or(0);

        b.set_spin_waits(true);
        let result: Result<Vec<Vec<Output>>> = (|| {
            let started = Instant::now();
            if total_rows > batch.prepared_rows {
                b.synchronize()?;
                for graph in &mut batch.graphs { *graph = None; }
                batch.graph_widths = [0; 2];
                batch.prepared_rows = total_rows;
            }
            if graphs { batch.retain_graph_width(b, total_rows)?; }
            batch.routing.clear(b, total_rows, shard)?;
            batch.moe.reserve_grouped(b, total_rows, model.main_activation_quantization(), compact.then_some(shard_count))?;
            batch.moe.set_routing_flag(&batch.routing)?;
            batch.input.resize_within_capacity(total_rows * H)?;
            batch.moe.resize(total_rows)?;

            let mut captured = Vec::with_capacity(total_rows);
            for c in 0..chats {
                for (k, &tok) in token_slices[c].iter().enumerate() {
                    let slot = row_starts[c] + k;
                    captured.push(sessions[c].rows[0].capture_device(w, tok,
                        batch.row_wire.device_address() + (slot * (EMBEDDING_BYTES + LEXICAL_BYTES)) as u64)?);
                }
            }
            RowBatch::gather_device(b, &mut captured, shard, &mut batch.row_wire, &mut batch.row_copies,
                |v, o| sessions[0].ranks[0].gather_bytes(v, o))?;

            let mut chat_embeddings: Vec<Vec<u8>> = (0..chats).map(|c| Vec::with_capacity(token_slices[c].len() * EMBEDDING_BYTES)).collect();
            for c in 0..chats {
                for k in 0..token_slices[c].len() {
                    let slot = row_starts[c] + k;
                    let token = captured[slot].token(0)?;
                    sessions[c].front[0].expand(b, &token)?;
                    b.copy_range(&sessions[c].front[0].residual, &mut batch.hidden, 0, slot * W, W)?;
                    let (lead, rest) = sessions.split_first_mut().unwrap();
                    if c == 0 {
                        lead.ple[0].set_lexical(b, &lead.front[0].lexical, slot)?;
                    } else {
                        lead.ple[0].set_lexical(b, &rest[c - 1].front[0].lexical, slot)?;
                    }
                    if !sessions[c].mtp.is_empty() {
                        chat_embeddings[c].extend_from_slice(&token.embedding);
                    }
                }
            }

            for layer in 0..layers {
                let prefix = format!("model.language_model.layers.{layer}.");
                let streamed = if model.resident { None } else {
                    Some(Weights::load_selected(b, &model.root, &model.checkpoint, w.plan.clone(),
                        |name| name.starts_with(&prefix) && !name.starts_with(PLE))?)
                };
                let weights = streamed.as_ref().unwrap_or(w);
                for s in sessions.iter_mut() { s.prepare_attention(b, std::slice::from_ref(weights), layer)?; }
                let connection = Hc::load(weights, &format!("{prefix}attn_hyper_connection"))?;

                if model.checkpoint.config.ple_layers.contains(&(layer as u64)) {
                    let (keys, values) = sessions[0].ple[0].project_lexical(b, total_rows)?;
                    sessions[0].ple[0].gate_rows(b, batch.hidden.device_address(), keys, values, total_rows)?;
                    let q_work = sessions[0].ple[0].query_work_address();
                    for c in 0..chats {
                        let len = token_slices[c].len();
                        for k in 0..len {
                            let slot = row_starts[c] + k;
                            sessions[c].ple[0].convolve_row(b,
                                q_work + (slot * W * 4) as u64,
                                batch.hidden.device_address() + (slot * W * 4) as u64,
                                keys + (slot * W * 4) as u64)?;
                            if k + 1 < len {
                                sessions[c].ple[0].copy_history(b, &mut all_saved[c][k].ple[0])?;
                            }
                        }
                    }
                }

                let prev_deferred = graphs && compact && layer > 0
                    && !model.checkpoint.config.ple_layers.contains(&(layer as u64));
                let prev_mlp = prev_deferred.then(|| &sessions[0].resident_moe[0][layer - 1]);
                batch.segment(b, graphs.then_some(((total_rows - 1) * layers + layer) * segments), |batch| {
                    if let Some(prev) = prev_mlp {
                        prev.finish_batch(b, &mut batch.moe, &mut batch.hidden, total_rows)?;
                    }
                    connection.mix_tiled(b, &mut batch.hc, &batch.hidden, &batch.input, total_rows)
                })?;

                if matches!(model.checkpoint.config.layers[layer], LayerKind::DeltaNet) {
                    let attention = sessions.iter_mut().map(|s| {
                        let state = s.gdn_cache[0][layer].take().ok_or("missing cached Flash GDN state")?;
                        Gdn::load_cached(b, weights, layer, Some(state))
                    }).collect::<Result<Vec<_>>>()?;
                    attention[0].project_batch(b, &batch.gdn, &batch.input, total_rows, false)?;
                    for k in 0..max_len {
                        let mut rows = Vec::with_capacity(chats);
                        for c in 0..chats {
                            if k < token_slices[c].len() {
                                rows.push(attention[c].session_row(
                                    sessions[c].gdn[0].as_ref().ok_or("missing cached Flash GDN scratch")?,
                                    sessions[c].gdn_norms[0][layer].as_ref().ok_or("missing cached Flash GDN norm")?,
                                    row_starts[c] + k)?);
                            }
                        }
                        if !rows.is_empty() {
                            batch.gdn.upload_sessions(b, &rows)?;
                            attention[0].step_batch(b, &batch.gdn, rows.len())?;
                            for c in 0..chats {
                                if k + 1 < token_slices[c].len() {
                                    attention[c].save_into(b, &mut all_saved[c][k].gdn[0][layer])?;
                                }
                            }
                        }
                    }
                    attention[0].finish_batch(b, &batch.gdn, &mut batch.input, total_rows)?;
                    for (s, q) in sessions.iter_mut().zip(attention) { s.gdn_cache[0][layer] = Some(q.into_state()); }
                } else {
                    let mut attention = sessions.iter_mut().map(|s| {
                        let state = s.qsa_cache[0][layer].take().ok_or("missing cached Flash QSA state")?;
                        Qsa::load_cached(b, weights, layer, s.capacity, Some(state))
                    }).collect::<Result<Vec<_>>>()?;
                    attention[0].project_batch(b, &mut batch.qsa, &batch.input, total_rows)?;
                    for c in 0..chats {
                        sessions[c].qsa[0].use_norm(sessions[c].qsa_norms[0][layer].as_ref().ok_or("missing cached Flash QSA norm")?)?;
                        attention[c].attend_batch_row(b, &batch.qsa, row_starts[c], &mut sessions[c].qsa[0], token_slices[c].len())?;
                    }
                    attention[0].finish_batch(b, &batch.qsa, &mut batch.input, total_rows)?;
                    for (s, q) in sessions.iter_mut().zip(attention) { s.qsa_cache[0][layer] = Some(q.into_state()); }
                }

                let lead = &mut *sessions[0];
                reduce_device(b, &mut lead.ranks, &mut lead.communication, None, &mut [&mut batch.input])?;
                let streamed_moe = if model.resident { None } else { Some(Moe::load(weights, layer, model.activation_quantization)?) };
                let mlp = if let Some(mlp) = &streamed_moe { mlp } else { &lead.resident_moe[0][layer] };
                batch.segment(b, graphs.then_some(((total_rows - 1) * layers + layer) * segments + 2), |batch| {
                    connection.combine_tiled(b, &mut batch.hc, &mut batch.hidden, &batch.input, total_rows)?;
                    mlp.project_batch(b, &mut batch.moe, &batch.hidden, total_rows, batch.descriptors.get(layer))
                })?;
                if compact {
                    mlp.prepare_exchange(&mut batch.moe, total_rows, batch.descriptors.get(layer))?;
                    exchange_middle(b, &mut lead.ranks[0], lead.mailbox.as_deref(), &batch.moe,
                        shard, total_rows, false, &mut lead.communication[0])?;
                    batch.segment(b, graphs.then_some(((total_rows - 1) * layers + layer) * segments + 3), |batch|
                        mlp.complete_middle(b, &batch.moe, total_rows))?;
                    exchange_middle(b, &mut lead.ranks[0], lead.mailbox.as_deref(), &batch.moe,
                        shard, total_rows, true, &mut lead.communication[0])?;
                } else {
                    reduce_device(b, &mut lead.ranks, &mut lead.communication, None, &mut [batch.moe.partials_mut()])?;
                }
                let defer_finish = graphs && compact && layer + 1 < layers
                    && !model.checkpoint.config.ple_layers.contains(&((layer + 1) as u64));
                if !defer_finish {
                    mlp.finish_batch(b, &mut batch.moe, &mut batch.hidden, total_rows)?;
                }
                let active_mem = [streamed.as_ref().map_or(0, Weights::resident_bytes), 0];
                for s in sessions.iter_mut() { s.account(b, active_mem, batch.device_bytes()); }
            }

            Hc::mixer(w, MIXER)?.mix_tiled(b, &mut batch.hc, &batch.hidden, &batch.input, total_rows)?;
            let range = sessions[0].head[0].vocabulary();
            let vlen = range.len();
            sessions[0].head[0].project_rows(b, &batch.input, &mut batch.logits, total_rows)?;
            b.enqueue_argmax_batch_device(
                &batch.logits.slice(0..total_rows * vlen)?,
                &mut batch.argmax_partials.slice(0..total_rows * vlen.div_ceil(1024) * 3)?,
                &mut batch.candidates.slice(0..total_rows * 3)?,
                total_rows,
            )?;
            let values = b.read_argmax_raw(&batch.candidates, total_rows, |v| Ok(v.to_vec()))?;
            let offsets = (0..=shard_count).map(|r| r * total_rows * 2).collect::<Vec<_>>();
            let mut wire = vec![0.0f32; *offsets.last().unwrap()];
            for (slot, v) in values.as_chunks::<3>().0.iter().enumerate() {
                let failed = batch.routing.row_failed_after_readback(slot)?;
                let tok = if failed { u32::MAX } else if v[2] != 1.0 || v.iter().any(|x| !x.is_finite()) { NONFINITE_LOGIT }
                    else { range.start + v[1] as u32 };
                wire[offsets[shard] + slot * 2] = f32::from_bits(tok);
                wire[offsets[shard] + slot * 2 + 1] = if failed || tok == NONFINITE_LOGIT { 0.0 } else { v[0] };
            }
            sessions[0].ranks[0].gather_f32(&mut wire, &offsets)?;
            let winners: Vec<u32> = (0..total_rows).map(|slot| Candidate::reduce(
                offsets[..shard_count].iter().map(|&at| Candidate {
                    token: wire[at + slot * 2].to_bits(), logit: wire[at + slot * 2 + 1],
                })).map(|c| c.token)).collect::<Result<Vec<_>>>()?;

            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut out = Vec::with_capacity(chats);
            for (c, (saved_c, emb_c)) in all_saved.into_iter().zip(chat_embeddings).enumerate() {
                let len = token_slices[c].len();
                let start = row_starts[c];
                let mut hidden = b.vector_arena(&[len * W])?.pop().unwrap();
                b.copy_range(&batch.hidden, &mut hidden, start * W, 0, len * W)?;
                sessions[c].verification = Some(Verification {
                    entry: entries[c],
                    tokens: token_slices[c].to_vec(),
                    saved: saved_c,
                    hidden: vec![hidden],
                    embeddings: vec![emb_c],
                });
                sessions[c].position = entries[c] + len;
                sessions[c].valid = true;
                let ms = elapsed_ms / len as f64;
                out.push(winners[start..start + len].iter().map(|&token| Output {
                    token, logits: None, elapsed_ms: ms, load_ms: 0.0,
                }).collect());
            }
            Ok(out)
        })();
        b.set_spin_waits(false);
        result
    }
}
impl Model<'_> {
    /// Decode bytes one rank streams for a verified row, from its own tensor ledger: the
    /// fixed part (the head and every target layer except routed experts and lexical
    /// tables) and the routed-expert bytes one more row adds (top_k of each layer's
    /// experts). Draft policies scale this shape by their measured target times.
    pub fn verify_byte_ledger(&self) -> (usize, usize) {
        let config = &self.checkpoint.config;
        self.persistent.iter().map(|w| {
            let (mut fixed, mut routed) = (w.prefix_bytes("lm_head."), 0);
            for layer in 0..config.layers.len() {
                let prefix = format!("model.language_model.layers.{layer}.");
                let experts = w.prefix_bytes(&format!("{prefix}mlp.experts."));
                fixed += w.prefix_bytes(&prefix) - experts;
                routed += experts;
            }
            (fixed.saturating_sub(w.prefix_bytes(PLE)), routed * config.top_k as usize / config.experts.max(1) as usize)
        }).max().unwrap_or((0, 0))
    }
}
