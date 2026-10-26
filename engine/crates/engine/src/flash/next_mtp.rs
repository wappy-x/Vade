//! Mixed NVFP4/BF16 proposals and causal rows-batched verification.
use super::super::{qsa::QsaGather, session::mtp::fuse_hidden};
use super::*;
const P: &str = "mtp.layers.0.self_attn.";
pub(super) fn load_moe<'a>(b: &Backend<'a>, w: &Weights<'a>, plan: &Plan) -> Result<MoeAddr<'a>> {
    let descriptors = draft_experts(plan, |t| w.device_address(&t.name))?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            descriptors.as_ptr().cast::<u8>(),
            std::mem::size_of_val(descriptors.as_slice()),
        )
    };
    let mut experts = b.vector(bytes.len() / 4)?;
    b.write_vector_bytes(&mut experts, bytes)?;
    let p = "mtp.layers.0.mlp";
    Ok(MoeAddr {
        experts,
        router: gdn::bf16(w, &format!("{p}.gate.weight"), &[512, 2560], EXPERTS * H)?,
        shared_gate: gdn::bf16(w, &format!("{p}.shared_expert_gate.weight"), &[1, 2560], H)?,
        shared: ["gate_proj", "up_proj", "down_proj"]
            .map(|n| w.device_address(&format!("{p}.shared_expert.{n}.weight")))
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| "invalid shared draft projection count")?,
    })
}
pub(super) struct Draft<'a, 'w> {
    pub(super) last: DeviceVector<'a>,
    normalized: DeviceVector<'a>,
    hidden: DeviceVector<'a>,
    embedding: DeviceVector<'a>,
    projected: DeviceVector<'a>,
    input: DeviceVector<'a>,
    saved: DeviceVector<'a>,
    block: DeviceVector<'a>,
    moe: Workspace<'a>,
    hc: HcScratch<'a>,
    attn_hc: Hc<'w>,
    mlp_hc: Hc<'w>,
    mixer: Hc<'w>,
    attention: Qsa<'a, 'w>,
    scratch: QsaScratch<'a>,
    head: Head<'a, 'w>,
    pub(super) verification: DeviceVector<'a>,
    /// Canonical normalized inputs retained only for optional sparse-head fallback.
    pub(super) bound_input: Option<DeviceVector<'a>>,
    pub(super) bound_at: std::cell::Cell<Option<(usize, usize)>>,
    logits: DeviceVector<'a>,
    partials: DeviceVector<'a>,
    result: DeviceVector<'a>,
    ready: bool,
    /// VADE_MTP_DEFER_FEED: target hidden rows (and their input tokens) of positions
    /// `backlog_entry..` not yet in the draft history; allocated on the first deferral.
    backlog: Option<DeviceVector<'a>>,
    backlog_tokens: Vec<u32>,
    backlog_entry: usize,
    /// observe_rows tiles: one flash_hybrid_copy row descriptor per token (OBSERVE_MAX rows).
    copy_rows: Option<DeviceVector<'a>>,
    /// VADE_MTP_CONF: the (value, id, valid, probability) readback row; allocated on first use.
    conf_out: Option<DeviceVector<'a>>,
}
const OBSERVE_MAX: usize = 128;
const COPY_ROW_WORDS: usize = std::mem::size_of::<spark_gpu_probe::kernel_params::HybridCopyRow>() / 4;
/// VADE_MTP_DEFER_FEED (default on; =0 off): while a chat's cohort cannot
/// draft, its pooled decode steps keep their hidden rows instead of feeding the drafter
/// one row per step; the rows enter the draft history in observe_rows tiles, in position
/// order, before the next proposal, verification or direct observe (or when DEFER_ROWS
/// are held). The history equals the per-step feed's; verification picks every token.
pub fn defer_feed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_MTP_DEFER_FEED").as_deref() != Ok("0"))
}
pub(super) const DEFER_ROWS: usize = 64;
/// VADE_MTP_CONF=<p>, 0 <= p < 1 (unset = 0.4, =off no threshold; drafter policy only,
/// crate::flash_draft_stack): a one-chat proposal stops at the first drafted token whose
/// drafter softmax probability is below p, and that token is not proposed, so verification
/// spends rows (each new row adds its own routed experts to the verify pass) only on likely
/// accepts. Verification still picks every token, so output bytes equal the undrafted run's;
/// only the rows per cycle change. VADE_MTP_CONF_LOG=1 prints each proposal's probabilities
/// (p=0 logs without stopping, for calibration). NextModel's load refuses a malformed value.
pub(super) fn conf_threshold() -> Option<f32> {
    static T: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        let t = crate::flash_draft_stack::mtp_conf().ok().flatten();
        if let Some(t) = t { eprintln!("flashnext mtp_conf threshold={t}"); }
        t
    })
}
fn conf_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_MTP_CONF_LOG").as_deref() == Ok("1"))
}
/// VADE_MTP_TILE_ALL (default on; =0 off): observe_rows for every multi-row span, not only
/// for VADE_MTP_BATCH chats.
pub(super) fn tile_all() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_MTP_TILE_ALL").as_deref() != Ok("0"))
}
impl<'a, 'w> Draft<'a, 'w> {
    fn arena_lengths(vocabulary: usize, depth: usize) -> [usize; 12] {
        [
            W,
            W,
            W,
            H,
            W,
            H,
            W,
            H,
            vocabulary,
            192 * 3,
            3,
            (depth + 1) * vocabulary,
        ]
    }
    pub(super) fn reservation_bytes(model: &NextModel<'a>, capacity: usize) -> Result<usize> {
        let depth = model.mtp_depth;
        if depth == 0 {
            return Ok(0);
        }
        let (with_draft, _) = model.plan.cache_bytes(capacity as u64, true).map_err(err)?;
        let (target, _) = model
            .plan
            .cache_bytes(capacity as u64, false)
            .map_err(err)?;
        let arena = Self::arena_lengths(model.vocabulary(), depth)
            .into_iter()
            .map(|n| (n * 4).div_ceil(64) * 64)
            .sum::<usize>();
        Ok((with_draft - target) as usize
            + arena
            + crate::flash_head_options::retained_input_bytes(model.retains_head_inputs())
            + Workspace::reservation_bytes(1, 1, depth, model.prompt_frontend)
            + QsaScratch::reservation_bytes(&model.weights.plan, 1, capacity, false)
            + [W, 320, W, 4, H].into_iter()
                .map(|n| (n * 4).div_ceil(64) * 64).sum::<usize>()
            + Head::reservation_bytes(model.vocabulary())
            + crate::flash_next_rows::RECURRENT_BYTES
            + (depth + 1)
                * (crate::flash_next_rows::PREFIX_BYTES - crate::flash_next_rows::RECURRENT_BYTES)
            + 36 * gdn::GdnReplay::reservation_bytes(depth + 1)
            + if defer_feed() { (DEFER_ROWS * W * 4).div_ceil(64) * 64 } else { 0 })
    }
    /// `attention`: a finished session's draft attention arena to reuse (next_prefix.rs adoption).
    pub(super) fn new(b: &Backend<'a>, model: &'w NextModel<'a>, capacity: usize, attention: Option<QsaState<'a>>) -> Result<Self> {
        let w = &model.weights;
        if model.draft_moe.is_none() {
            return Err("FP4 MTP draft expert descriptors missing".into());
        }
        let mut v = b
            .vector_arena(&Self::arena_lengths(model.vocabulary(), model.mtp_depth))?
            .into_iter();
        let mut scratch = QsaScratch::new(b, &w.plan, 1, capacity)?;
        scratch.set_norms(b, &QsaGather::norms_prefix(w, P)?.gather(|_, _| Ok(()))?)?;
        Ok(Self {
            last: v.next().ok_or("missing draft arena vector")?,
            normalized: v.next().ok_or("missing draft arena vector")?,
            hidden: v.next().ok_or("missing draft arena vector")?,
            embedding: v.next().ok_or("missing draft arena vector")?,
            projected: v.next().ok_or("missing draft arena vector")?,
            input: v.next().ok_or("missing draft arena vector")?,
            saved: v.next().ok_or("missing draft arena vector")?,
            block: v.next().ok_or("missing draft arena vector")?,
            moe: Workspace::new(b, model, 1, 1)?,
            logits: v.next().ok_or("missing draft arena vector")?,
            partials: v.next().ok_or("missing draft arena vector")?,
            result: v.next().ok_or("missing draft arena vector")?,
            verification: v.next().ok_or("missing draft arena vector")?,
            bound_input: if model.retains_head_inputs() {
                Some(b.vector(crate::flash_head_options::RETAINED_INPUT_BYTES / 4)?)
            } else { None },
            bound_at: std::cell::Cell::new(None),
            hc: HcScratch::new(b)?,
            attn_hc: Hc::load_rows(w, "mtp.layers.0.attn_hyper_connection")?,
            mlp_hc: Hc::load_rows(w, "mtp.layers.0.mlp_hyper_connection")?,
            mixer: Hc::mixer_rows(w, "mtp.hyper_connection_mixer")?,
            attention: {
                let mut qsa = Qsa::load_prefix(b, w, P, 48, capacity, attention)?;
                if !model.draft_layer_mx8.is_empty() { qsa.use_draft_mx8()?; }
                qsa
            },
            scratch,
            head: {
                // VADE_FLASHNEXT_DRAFT_VOCAB=N makes the draft score ids 0..N only
                // (unset = 65536: C1 DEFAULT English prose 27.32 -> 31.36 tok/s, but
                // under 2 % of Han ids lie below 65536; =0 the full head). The draft feeds
                // only the greedy proposals (Flash-Next has no sampled drafter) and
                // verification keeps the full head, so output bytes never depend on it.
                let mut head = Head::load(b, w)?;
                head.limit_rows(model.draft_head_rows as u32);
                head.use_mx8_draft(model.draft_head_mx8.as_ref());
                head.use_fp4_draft(model.draft_head_fp4.as_ref());
                head
            },
            ready: false,
            backlog: None,
            backlog_tokens: Vec::new(),
            backlog_entry: 0,
            copy_rows: None,
            conf_out: None,
        })
    }
    pub(super) fn backlog_len(&self) -> usize {
        self.backlog_tokens.len()
    }
    /// Keep row `offset` of `hidden` (the target output at position `entry`, input `token`).
    pub(super) fn defer(&mut self, b: &Backend<'a>, entry: usize, token: u32, hidden: &DeviceVector<'a>,
        offset: usize) -> Result<()> {
        let n = self.backlog_tokens.len();
        if n == 0 {
            self.backlog_entry = entry;
        } else if entry != self.backlog_entry + n || n >= DEFER_ROWS {
            return Err("deferred draft rows out of order".into());
        }
        if self.backlog.is_none() {
            self.backlog = Some(b.vector(DEFER_ROWS * W)?);
        }
        b.copy_range(hidden, self.backlog.as_mut().unwrap(), offset * W, n * W, W)?;
        self.backlog_tokens.push(token);
        Ok(())
    }
    /// The held rows as (entry, tokens, hidden); `restore_backlog` returns the buffer.
    pub(super) fn take_backlog(&mut self) -> Option<(usize, Vec<u32>, DeviceVector<'a>)> {
        if self.backlog_tokens.is_empty() {
            return None;
        }
        Some((self.backlog_entry, std::mem::take(&mut self.backlog_tokens), self.backlog.take()?))
    }
    pub(super) fn restore_backlog(&mut self, hidden: DeviceVector<'a>) {
        self.backlog = Some(hidden);
    }
    pub(super) fn reset(&mut self) {
        self.backlog_tokens.clear();
        self.ready = false;
        self.attention.reset();
    }
    /// Prefix cache: draft attention rows 0..rows (proposals never write below the
    /// prompt's last row). Draft history only shapes proposals, never output bytes.
    pub(super) fn history_lengths(&self, rows: usize) -> [usize; 3] {
        self.attention.prefix_lengths(rows)
    }
    pub(super) fn copy_history_into(&self, b: &Backend<'a>, rows: usize, storage: [DeviceVector<'a>; 3]) -> Result<QsaState<'a>> {
        self.attention.snapshot_prefix_into(b, rows, storage)
    }
    pub(super) fn history_bytes(&self, rows: usize) -> usize {
        self.attention.prefix_bytes(rows)
    }
    pub(super) fn restore_history(&mut self, b: &Backend<'a>, rows: &QsaState<'a>, last: &DeviceVector<'a>) -> Result<()> {
        self.attention.restore(b, rows)?;
        b.copy_range(last, &mut self.last, 0, 0, W)?;
        self.ready = true;
        Ok(())
    }
    /// Adoption: the attention arena already holds rows 0..rows (never rewritten below the
    /// prompt's last row); only its length and the saved last row are restored.
    pub(super) fn adopt_history(&mut self, b: &Backend<'a>, rows: usize, last: &DeviceVector<'a>) -> Result<()> {
        self.attention.rewind(rows);
        b.copy_range(last, &mut self.last, 0, 0, W)?;
        self.ready = true;
        Ok(())
    }
    pub(super) fn into_attention(self) -> QsaState<'a> {
        self.attention.into_state()
    }
    fn fuse(&mut self, b: &Backend<'a>, model: &NextModel<'a>, token: u32) -> Result<()> {
        let mut p = HybridCopy {
            source: model
                .weights
                .device_address(EMBED)?
                .wrapping_add((token as usize * H * 2) as u64) as _,
            destination: self.projected.device_address() as _,
            words: (H / 2) as u32,
            padding: 0,
            rows: 0,
        };
        unsafe {
            b.launch_flash(
                "flash_hybrid_copy",
                p.words.div_ceil(128),
                128,
                &mut [(&mut p as *mut HybridCopy).cast()],
            )?;
        }
        let (mut raw, mut out, mut width) = (
            self.projected.device_address(),
            self.embedding.device_address(),
            H as u32,
        );
        unsafe {
            b.launch_flash(
                "flash_embedding_expand",
                H.div_ceil(128) as u32,
                128,
                &mut [
                    (&mut raw as *mut u64).cast(),
                    (&mut out as *mut u64).cast(),
                    (&mut width as *mut u32).cast(),
                ],
            )?;
        }
        fuse_hidden(
            b,
            &model.weights,
            [
                self.last.device_address(),
                self.normalized.device_address(),
                self.hidden.device_address(),
                self.embedding.device_address(),
                self.projected.device_address(),
            ],
            1,
        )?;
        self.attn_hc
            .mix_tiled(b, &mut self.hc, &self.hidden, &self.input, 1)
    }
    pub(super) fn observe(
        &mut self,
        b: &Backend<'a>,
        model: &NextModel<'a>,
        entry: usize,
        tokens: &[u32],
        hidden: &DeviceVector<'a>,
        offset: usize,
    ) -> Result<()> {
        for (i, &token) in tokens.iter().enumerate() {
            if entry + i > 0 {
                self.attention.rewind(entry + i - 1);
                self.fuse(b, model, token)?;
                self.attention
                    .project_cache(b, &mut self.scratch, &self.input, 1)?;
                self.attention.observe(b, &mut self.scratch)?;
            }
            b.copy_range(hidden, &mut self.last, (offset + i) * W, 0, W)?;
            self.ready = true;
        }
        Ok(())
    }
    /// VADE_MTP_BATCH: `observe` over a chat's rows (prompt chunk or verify rows) in tiles of
    /// up to DRAFT_ROWS: per tile one embedding copy, fuse (fc_hidden in pairs of rows, the
    /// 8-token decode group), HC mix, KV/index projection and cache preparation, instead of
    /// one of each per row. Every stage keeps the per-row graph it runs at M=1, so the draft
    /// history is the one `observe` writes. Scratch: the idle prompt-gate slot (`idle`, read
    /// only by prompt tiles before their experts) and the workspace QSA scratch under this
    /// draft's norms (every target QSA layer selects its own norms first).
    pub(super) fn observe_rows(
        &mut self,
        b: &Backend<'a>,
        model: &NextModel<'a>,
        idle: &DeviceVector<'a>,
        hc: &mut HcScratch<'a>,
        qsa: &mut QsaScratch<'a>,
        entry: usize,
        tokens: &[u32],
        hidden: &DeviceVector<'a>,
        offset: usize,
    ) -> Result<()> {
        if entry == 0 && !tokens.is_empty() {
            self.observe(b, model, 0, &tokens[..1], hidden, offset)?;
            return self.observe_rows(b, model, idle, hc, qsa, 1, &tokens[1..], hidden, offset + 1);
        }
        let (tile, group) = observe_tile();
        let r = tile.min(idle.len() / (3 * (W + H))).min(qsa.max_tokens());
        if tokens.len() < 2 || r < 2 {
            return self.observe(b, model, entry, tokens, hidden, offset);
        }
        qsa.use_norm(&self.scratch.own_norms()?)?;
        let mut previous = idle.slice(0..r * W)?;
        let normalized = idle.slice(r * W..2 * r * W)?;
        let fused = idle.slice(2 * r * W..3 * r * W)?;
        let base = 3 * r * W;
        let embedding = idle.slice(base..base + r * H)?;
        let projected = idle.slice(base + r * H..base + 2 * r * H)?;
        for start in (0..tokens.len()).step_by(r) {
            let n = (tokens.len() - start).min(r);
            // Row i fuses the target hidden of row i - 1 (`last` for the tile's first row).
            b.copy_range(&self.last, &mut previous, 0, 0, W)?;
            if n > 1 {
                b.copy_range(hidden, &mut previous, (offset + start) * W, W, (n - 1) * W)?;
            }
            self.embed_tile(b, model, &tokens[start..start + n], &projected, &embedding)?;
            for c in (0..n).step_by(group) {
                let (w, h) = ((c * W * 4) as u64, (c * H * 4) as u64);
                fuse_hidden(b, &model.weights, [previous.device_address() + w, normalized.device_address() + w,
                    fused.device_address() + w, embedding.device_address() + h,
                    projected.device_address() + h], (n - c).min(group))?;
            }
            let input = idle.slice(base + 2 * r * H..base + 2 * r * H + n * H)?;
            self.attn_hc.mix_tiled(b, hc, &fused, &input, n)?;
            self.attention.rewind(entry + start - 1);
            self.attention.project_cache(b, qsa, &input, n)?;
            self.attention.observe(b, qsa)?;
            b.copy_range(hidden, &mut self.last, (offset + start + n - 1) * W, 0, W)?;
            self.ready = true;
        }
        Ok(())
    }
    /// embed_rows with one flash_hybrid_copy launch per tile (row descriptors in the grid's
    /// second dimension, the same word copies) instead of one launch per token.
    fn embed_tile(&mut self, b: &Backend<'a>, model: &NextModel<'a>, tokens: &[u32], projected: &DeviceVector<'a>,
        embedding: &DeviceVector<'a>) -> Result<()> {
        if tokens.len() > OBSERVE_MAX {
            return embed_rows(b, model, tokens, projected, embedding);
        }
        if self.copy_rows.is_none() {
            // Arena storage: each tile uploads through a slice of its own length.
            self.copy_rows = b.vector_arena(&[OBSERVE_MAX * COPY_ROW_WORDS])?.into_iter().next();
        }
        let base = model.weights.device_address(EMBED)?;
        let rows: Vec<_> = tokens.iter().enumerate().map(|(i, &t)| spark_gpu_probe::kernel_params::HybridCopyRow {
            source: base.wrapping_add((t as usize * H * 2) as u64),
            destination: projected.device_address() + (i * H * 2) as u64,
            words: (H / 2) as u32,
            padding: 0,
        }).collect();
        let bytes = unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows.as_slice())) };
        let table = self.copy_rows.as_mut().unwrap();
        b.write_vector_bytes(&mut table.slice(0..tokens.len() * COPY_ROW_WORDS)?, bytes)?;
        let mut p = HybridCopy { source: std::ptr::null(), destination: std::ptr::null_mut(), words: 0, padding: 0,
            rows: table.device_address() };
        unsafe {
            b.launch_flash_rows("flash_hybrid_copy", ((H / 2) as u32).div_ceil(128), tokens.len() as u32, 128, 0,
                &mut [(&mut p as *mut HybridCopy).cast()])?;
        }
        let (mut raw, mut expanded, mut width) =
            (projected.device_address(), embedding.device_address(), (tokens.len() * H) as u32);
        unsafe {
            b.launch_flash("flash_embedding_expand", (tokens.len() * H).div_ceil(128) as u32, 128,
                &mut [(&mut raw as *mut u64).cast(), (&mut expanded as *mut u64).cast(),
                    (&mut width as *mut u32).cast()])?;
        }
        Ok(())
    }
    fn mlp(&mut self, b: &Backend<'a>, model: &NextModel<'a>) -> Result<()> {
        self.mlp_hc
            .mix_tiled(b, &mut self.hc, &self.hidden, &self.input, 1)?;
        self.moe.draft_moe(
            b,
            model
                .draft_moe
                .as_ref()
                .ok_or("FP4 MTP draft experts missing")?,
            &self.input,
            &mut self.block,
        )?;
        self.mlp_hc
            .combine_tiled(b, &mut self.hc, &mut self.hidden, &self.block, 1)
    }
    /// `hash` (VADE_MTP_BATCH chats only) is the chat's PLE hash state: each step's input row
    /// (the verify rows but the last) gets page-cache hints while the step runs. None = main's
    /// draft steps, with no hint work on the host.
    pub(super) fn propose(
        &mut self,
        b: &Backend<'a>,
        model: &NextModel<'a>,
        position: usize,
        mut token: u32,
        depth: usize,
        mut hash: Option<PleHash>,
    ) -> Result<Vec<u32>> {
        if !self.ready || position == 0 {
            return Err("FP4 MTP needs target history".into());
        }
        b.copy_range(&self.last, &mut self.saved, 0, 0, W)?;
        self.attention.rewind(position - 1);
        let conf = conf_threshold();
        let mut probs = Vec::new();
        let result = (|| {
            let mut out = Vec::new();
            for _ in 0..depth {
                self.fuse(b, model, token)?;
                self.attention
                    .project(b, &mut self.scratch, &self.input, 1)?;
                self.attention.attend(b, &mut self.scratch)?;
                self.attn_hc.combine_tiled(
                    b,
                    &mut self.hc,
                    &mut self.hidden,
                    self.scratch.output(),
                    1,
                )?;
                self.mlp(b, model)?;
                b.copy_range(&self.hidden, &mut self.last, 0, 0, W)?;
                self.mixer
                    .mix_tiled(b, &mut self.hc, &self.hidden, &self.input, 1)?;
                Head::project_next_batch(
                    b,
                    &mut [&mut self.head],
                    &self.input,
                    &mut self.logits,
                    &mut self.partials,
                    &mut self.result,
                )?;
                // Hints only (the verify frontend preads the same row): start them while the step runs.
                if let Some(hash) = hash.as_mut() {
                    model.prefill_reader.hint_rows(&[(token, hash.advance(token))]);
                }
                let mut unsure = false;
                token = if let Some(threshold) = conf {
                    if self.conf_out.is_none() { self.conf_out = Some(b.vector(4)?); }
                    let out_row = self.conf_out.as_ref().ok_or("missing draft confidence row")?;
                    let (candidate, p) = self.head.draft_confidence(b, &self.partials, out_row)?;
                    probs.push(p);
                    unsure = p < threshold;
                    candidate.token
                } else {
                    self
                        .head
                        .local_max(b)?
                        .ok_or("empty draft vocabulary")?
                        .token
                };
                // The argmax readback above completed this step's MoE: check its routing.
                if self.moe.draft_routing_failed()? {
                    return Err("MTP draft routing rejected a row".into());
                }
                if unsure { break; }
                out.push(token);
            }
            Ok(out)
        })();
        if conf.is_some() && conf_log() {
            eprintln!("mtpconf pos={position} depth={depth} proposed={} p={probs:?}", result.as_ref().map_or(0, |o| o.len()));
        }
        b.copy_range(&self.saved, &mut self.last, 0, 0, W)?;
        self.attention.rewind(position - 1);
        result
    }
}
/// BF16 embedding rows of `tokens` (raw into `projected`), expanded to FP32 in `embedding`.
fn embed_rows(b: &Backend<'_>, model: &NextModel<'_>, tokens: &[u32], projected: &DeviceVector<'_>,
    embedding: &DeviceVector<'_>) -> Result<()> {
    for (i, &token) in tokens.iter().enumerate() {
        let mut p = HybridCopy {
            source: model.weights.device_address(EMBED)?.wrapping_add((token as usize * H * 2) as u64) as _,
            destination: (projected.device_address() + (i * H * 2) as u64) as _,
            words: (H / 2) as u32,
            padding: 0,
            rows: 0,
        };
        unsafe {
            b.launch_flash("flash_hybrid_copy", p.words.div_ceil(128), 128,
                &mut [(&mut p as *mut HybridCopy).cast()])?;
        }
    }
    let (mut raw, mut expanded, mut width) =
        (projected.device_address(), embedding.device_address(), (tokens.len() * H) as u32);
    unsafe {
        b.launch_flash("flash_embedding_expand", (tokens.len() * H).div_ceil(128) as u32, 128,
            &mut [(&mut raw as *mut u64).cast(), (&mut expanded as *mut u64).cast(),
                (&mut width as *mut u32).cast()])?;
    }
    Ok(())
}
/// Rows per observe_rows tile and per fuse_hidden call. Default
/// 128 (OBSERVE_MAX): one fuse per tile instead of fc_embedding + fc_hidden passes per row
/// pair; on one Spark (26b1692e base) cold 8K drafting-ON TTFT 5.43 s (pairs) -> 4.77 (64) -> 4.74
/// (128) vs OFF 4.67-4.70, every request's draft cycles, accepted and proposed counts
/// unchanged. VADE_MTP_OBSERVE_ROWS=8 keeps the earlier tiles (pairs), =16 quads, =32 / =64
/// whole tiles. Draft history only; verification picks every token.
fn observe_tile() -> (usize, usize) {
    static TILE: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *TILE.get_or_init(|| match std::env::var("VADE_MTP_OBSERVE_ROWS").as_deref() {
        Ok("8") => (DRAFT_ROWS, 2),
        Ok("16") => (16, 4),
        Ok("32") => (32, 32),
        Ok("64") => (64, 64),
        _ => (OBSERVE_MAX, OBSERVE_MAX),
    })
}
/// Cohort draft tile: the BF16 decode row group, so the MTP layer's projections keep
/// their one-row K order (blocks::project_rows) and a chat drafts what it drafts alone.
const DRAFT_ROWS: usize = spark_gpu_probe::kernel_params::BF16_ROWS_TOKENS;
impl<'a, 'w> Draft<'a, 'w> {
    /// VADE_MTP_BATCH: `propose` for up to DRAFT_ROWS chats, each depth step running the
    /// MTP layer, its experts and the drafter head once over the drafting rows. Per-chat
    /// state stays per chat (draft KV and `last`); rows are ordered deepest first, so a
    /// step's drafting rows are a prefix. One argmax readback per step also covers the
    /// routing status, instead of two host syncs per chat and step. `hashes` are the
    /// chats' PLE hash states: each step's input rows (the verify rows but the last)
    /// get page-cache hints while the step runs.
    fn propose_rows(
        b: &Backend<'a>,
        model: &NextModel<'a>,
        work: &mut Workspace<'a>,
        drafts: &mut [&mut Self],
        mut tokens: Vec<u32>,
        depths: &[usize],
        hashes: &mut [PleHash],
    ) -> Result<Vec<Vec<u32>>> {
        let n = drafts.len();
        if n == 0 || n > DRAFT_ROWS || n > work.rows || tokens.len() != n || depths.len() != n
            || hashes.len() != n || depths.windows(2).any(|d| d[0] < d[1]) {
            return Err("invalid MTP draft rows".into());
        }
        let addr = model.draft_moe.as_ref().ok_or("FP4 MTP draft experts missing")?;
        work.residual.resize_within_capacity(n * W)?;
        work.mixed.resize_within_capacity(n * H)?;
        work.block.resize_within_capacity(n * H)?;
        // Cohort last / normalized / embedding / projected rows borrow the idle BF16
        // prompt-gate slot (read only by prompt tiles wider than 16 rows).
        let (r, idle) = (DRAFT_ROWS, &work.mixed16);
        let mut last = idle.slice(0..n * W)?;
        let normalized = idle.slice(r * W..r * W + n * W)?;
        let embedding = idle.slice(2 * r * W..2 * r * W + n * H)?;
        let projected = idle.slice(2 * r * W + r * H..2 * r * W + r * H + n * H)?;
        for (i, d) in drafts.iter().enumerate() {
            b.copy_range(&d.last, &mut last, 0, i * W, W)?;
        }
        let start = drafts[0].head.vocabulary().start;
        let mut out = vec![Vec::new(); n];
        for step in 0..depths[0] {
            let m = depths.iter().take_while(|&&k| k > step).count();
            embed_rows(b, model, &tokens[..m], &projected, &embedding)?;
            // fc_hidden projects four streams per row: pairs keep it on the 8-token decode group.
            for c in (0..m).step_by(2) {
                let (w, h) = ((c * W * 4) as u64, (c * H * 4) as u64);
                fuse_hidden(b, &model.weights, [last.device_address() + w, normalized.device_address() + w,
                    work.residual.device_address() + w, embedding.device_address() + h,
                    projected.device_address() + h], (m - c).min(2))?;
            }
            let first = &mut *drafts[0];
            first.attn_hc.mix_tiled(b, &mut work.hc, &work.residual, &work.mixed, m)?;
            let batch = work.qsa_batch.as_mut().ok_or("missing draft QSA rows")?;
            first.attention.project_batch(b, batch, &work.mixed, m)?;
            for (i, d) in drafts[..m].iter_mut().enumerate() {
                let d = &mut **d;
                d.attention.attend_batch_row(b, batch, i, &mut d.scratch, 1)?;
            }
            let first = &*drafts[0];
            first.attention.finish_batch(b, batch, &mut work.block, m)?;
            first.attn_hc.combine_tiled(b, &mut work.hc, &mut work.residual, &work.block, m)?;
            first.mlp_hc.mix_tiled(b, &mut work.hc, &work.residual, &work.mixed, m)?;
            work.status.clear(b, m, 0)?;
            work.draft_moe_rows(b, addr, m)?;
            let first = &*drafts[0];
            first.mlp_hc.combine_tiled(b, &mut work.hc, &mut work.residual, &work.block, m)?;
            if step + 1 < depths[0] {
                b.copy_range(&work.residual, &mut last, 0, 0, m * W)?;
            }
            first.mixer.mix_tiled(b, &mut work.hc, &work.residual, &work.mixed, m)?;
            first.head.project_draft_rows(b, &work.mixed, m, &mut work.head_logits, &mut work.head_partials,
                &mut work.head_result)?;
            // Hints only (the verify frontend preads the same rows): start them while the step runs.
            let rows: Vec<_> = tokens[..m].iter().zip(hashes.iter_mut()).map(|(&t, h)| (t, h.advance(t))).collect();
            model.prefill_reader.hint_rows(&rows);
            let ids = b.read_argmax_results(&work.head_result, m)?;
            if work.status.failed_after_readback()? {
                return Err("MTP draft routing rejected a row".into());
            }
            for ((token, row), (id, _)) in tokens.iter_mut().zip(out.iter_mut()).zip(ids) {
                *token = start + id;
                row.push(*token);
            }
        }
        Ok(out)
    }
}
pub(super) struct Prefix<'a> {
    pub(super) position: usize,
    pub(super) hash: PleHash,
    pub(super) ple: DeviceVector<'a>,
    pub(super) last: DeviceVector<'a>,
}
/// One entry snapshot and small projection tapes; rejected prefixes replay only GDN.
pub(super) struct Rollback<'a> {
    pub(super) gdn: Vec<Option<GdnState<'a>>>,
    pub(super) inputs: Vec<Option<gdn::GdnReplay<'a>>>,
    /// The last verify window stayed inside one GDN chunk (gdn::shallow_verify):
    /// its entry snapshots hold the conv history only.
    pub(super) shallow: bool,
}
impl<'a, 'w> NextSession<'a, 'w> {
    /// Set the admitted lookup limit; zero restores the resident MTP width.
    /// Retired wider tapes/prefixes must not survive a smaller new reservation.
    pub fn enable_lookup(&mut self, b: &Backend<'a>, max: usize) -> Result<()> {
        if max > 15 || self.draft.is_none() && max > 0 {
            return Err("lookup requires resident MTP and max draft <=15".into());
        }
        let rows = (max + 1).max(self.mtp_depth() + 1);
        if rows != self.verify_rows {
            if let Some(d) = &mut self.draft {
                d.verification = b.vector(rows * self.model.vocabulary())?;
            }
            self.saved.clear();
            self.rollback = None;
            self.retained = 0;
            self.verify_rows = rows;
        }
        Ok(())
    }
    pub fn max_verify_rows(&self) -> usize {
        self.verify_rows
    }
    pub fn mtp_depth(&self) -> usize {
        if self.draft.is_some() {
            self.model.mtp_depth
        } else {
            0
        }
    }
    /// Plain requests skip draft history and verification allocations.
    pub fn disable_mtp(&mut self) {
        self.draft = None;
        self.saved.clear();
        self.rollback = None;
        self.retained = 0;
    }
    pub fn propose_mtp(&mut self, b: &Backend<'a>, token: u32, depth: usize) -> Result<Vec<u32>> {
        self.flush_draft_feed(b)?;
        if depth == 0
            || depth > self.mtp_depth()
            || self.pos + depth + 1 > self.capacity
            || !self.valid
        {
            return Ok(Vec::new());
        }
        if token as usize >= self.model.vocabulary() {
            return Err("MTP token outside vocabulary".into());
        }
        self.retained = 0;
        self.draft
            .as_mut()
            .ok_or("MTP disabled")?
            .propose(b, self.model, self.pos, token, depth, self.batch_drafts.then(|| self.hash.clone()))
    }
    /// VADE_MTP_BATCH: `propose_mtp` for every chat (same tokens), with the drafting
    /// chats' depth steps run together in tiles of DRAFT_ROWS; a lone drafter (or a chat
    /// propose_mtp would skip) takes exactly propose_mtp's path.
    pub fn propose_mtp_batch(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        tokens: &[u32],
        depths: &[usize],
    ) -> Result<Vec<Vec<u32>>> {
        if sessions.len() != tokens.len() || sessions.len() != depths.len() {
            return Err("invalid MTP draft cohort".into());
        }
        for s in sessions.iter_mut() {
            s.flush_draft_feed(b)?;
        }
        let mut out = vec![Vec::new(); sessions.len()];
        let mut rows = Vec::new();
        for (i, s) in sessions.iter().enumerate() {
            let depth = depths[i];
            if depth == 0 || depth > s.mtp_depth() || s.pos + depth + 1 > s.capacity || !s.valid {
                continue;
            }
            if tokens[i] as usize >= s.model.vocabulary() {
                return Err("MTP token outside vocabulary".into());
            }
            if !s.draft.as_ref().is_some_and(|d| d.ready) || s.pos == 0 {
                return Err("FP4 MTP needs target history".into());
            }
            rows.push(i);
        }
        if rows.len() < 2 {
            for &i in &rows {
                out[i] = sessions[i].propose_mtp(b, tokens[i], depths[i])?;
            }
            return Ok(out);
        }
        rows.sort_by_key(|&i| std::cmp::Reverse(depths[i]));
        let mut slots: Vec<_> = sessions.iter_mut().map(|s| Some(&mut **s)).collect();
        let mut chats: Vec<&mut Self> = rows.iter().map(|&i| slots[i].take().unwrap()).collect();
        // The model's one workspace serves the cohort (funded once in shared_reservation_bytes).
        let model = chats[0].model;
        let capacity = chats.iter().map(|s| s.capacity).max().unwrap_or(1);
        let mut work = model.take_work(b, capacity, DRAFT_ROWS)?;
        let mut result = Ok(());
        for (tile, ids) in chats.chunks_mut(DRAFT_ROWS).zip(rows.chunks(DRAFT_ROWS)) {
            let positions: Vec<_> = tile.iter().map(|s| s.pos).collect();
            let mut hashes: Vec<_> = tile.iter().map(|s| s.hash.clone()).collect();
            // As in propose: drafts append from the last committed row, then rewind.
            let mut drafts: Vec<_> = tile.iter_mut().zip(&positions).filter_map(|(s, &pos)| {
                s.retained = 0;
                let d = s.draft.as_mut()?;
                d.attention.rewind(pos - 1);
                Some(d)
            }).collect();
            let first = ids.iter().map(|&i| tokens[i]).collect();
            let depth: Vec<_> = ids.iter().map(|&i| depths[i]).collect();
            let proposed = Draft::propose_rows(b, model, &mut work, &mut drafts, first, &depth, &mut hashes);
            for (d, &pos) in drafts.iter_mut().zip(&positions) {
                d.attention.rewind(pos - 1);
            }
            match proposed {
                Ok(proposed) => {
                    for (&i, p) in ids.iter().zip(proposed) {
                        out[i] = p;
                    }
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        model.return_work(work);
        result.map(|()| out)
    }
    fn prepare_verify(&mut self, b: &Backend<'a>, tokens: &[u32]) -> Result<()> {
        if self.mtp_depth() == 0
            || tokens.is_empty()
            || tokens.len() > self.verify_rows
            || !self.valid
            || self.pos == 0
            || self.pos + tokens.len() > self.capacity
            || tokens
                .iter()
                .any(|&t| t as usize >= self.model.vocabulary())
        {
            return Err("invalid FP4 verification width".into());
        }
        while self.saved.len() < tokens.len() {
            self.saved.push(Prefix {
                position: 0,
                hash: self.hash.clone(),
                ple: self.ple.save_history(b)?,
                last: b.vector(W)?,
            });
        }
        if self.rollback.is_none() {
            // The rollback storage is funded by the session reservation that an unpublished
            // prompt checkpoint borrowed during prefill (next_prefix.rs): release it first.
            *self.checkpoint.get_mut() = None;
        }
        if self.rollback.is_none() && gdn::session_arena() {
            let tapes: Vec<bool> = (0..LAYERS).map(|li| li % 4 != 3 && self.model.owns_state(li)).collect();
            let (gdn, inputs) = gdn::GdnState::rollback_arena(b, &self.gdn_state, &tapes, self.verify_rows)?;
            self.rollback = Some(Rollback { gdn, inputs, shallow: false });
        }
        if self.rollback.is_none() {
            self.rollback = Some(Rollback {
                gdn: self
                    .gdn_state
                    .iter()
                    .map(|g| g.as_ref().map(|g| g.next_snapshot_storage(b)).transpose())
                    .collect::<Result<_>>()?,
                inputs: (0..LAYERS)
                    .map(|li| {
                        if li % 4 == 3 || !self.model.owns_state(li) {
                            Ok(None)
                        } else {
                            gdn::GdnReplay::new(b, self.verify_rows).map(Some)
                        }
                    })
                    .collect::<Result<_>>()?,
            
                shallow: false,
            });
        }
        let mut hash = self.hash.clone();
        for (i, &token) in tokens.iter().enumerate() {
            self.saved[i].position = self.pos + i;
            self.saved[i].hash = hash.clone();
            hash.advance(token);
        }
        b.copy_range(
            &self.draft.as_ref().ok_or("MTP disabled")?.last,
            &mut self.saved[0].last,
            0,
            0,
            W,
        )?;
        self.retained = tokens.len();
        Ok(())
    }
    /// VADE_W2_PP_MTP=1 back: GDN rollback storage for `rows` verify rows at `pos` (the back
    /// keeps no draft, token hash or PLE history; the front owns those).
    pub(super) fn prepare_back_verify(&mut self, b: &Backend<'a>, rows: usize) -> Result<()> {
        // The front widens its verify rows for lookup drafts (enable_lookup: at most 16 rows);
        // the back widens its GDN tape to match.
        if self.mtp_depth() == 0 || rows == 0 || rows > 16 || !self.valid || self.pos == 0
            || self.pos + rows > self.capacity
        {
            return Err(format!("invalid pipeline verification width {rows} at {} of {}", self.pos, self.capacity));
        }
        if rows > self.verify_rows {
            self.rollback = None;
            self.verify_rows = rows;
        }
        if self.rollback.is_none() {
            self.rollback = Some(Rollback {
                gdn: self
                    .gdn_state
                    .iter()
                    .map(|g| g.as_ref().map(|g| g.next_snapshot_storage(b)).transpose())
                    .collect::<Result<_>>()?,
                inputs: (0..LAYERS)
                    .map(|li| if li % 4 == 3 || !self.model.owns_state(li) { Ok(None) } else { gdn::GdnReplay::new(b, self.verify_rows).map(Some) })
                    .collect::<Result<_>>()?,
                shallow: false,
            });
        }
        self.retained = rows;
        Ok(())
    }
    /// VADE_W2_PP_MTP=1 back: keep `position` (the front's commit) of the last verify's rows:
    /// the GDN restore + replay and QSA rewind of commit_mtp, for layers [split, 48).
    pub(super) fn commit_back(&mut self, b: &Backend<'a>, position: usize) -> Result<()> {
        let base = self.pos - self.retained;
        if position == self.pos {
            self.retained = 0;
            return Ok(());
        }
        if position <= base || position > self.pos {
            return Err(format!("pipeline commit {position} outside the verified rows {base}..{}", self.pos));
        }
        let prefix = position - base;
        self.valid = false;
        let rollback = self.rollback.as_mut().ok_or("MTP rollback missing")?;
        if rollback.shallow {
            for (live, entry) in self.gdn_state.iter_mut().zip(rollback.gdn.iter_mut()) {
                if let (Some(live), Some(entry)) = (live.as_mut(), entry.as_mut()) { live.restore_history(entry); }
            }
        } else {
            std::mem::swap(&mut self.gdn_state, &mut rollback.gdn);
        }
        let mut work = self.model.take_work(b, self.capacity, 1)?;
        let layers = self.model.pp_split.get()..self.model.pp_end.get();
        let replayed = work.replay_gdn(b, self.model, &mut self.gdn_state, rollback, prefix, base, layers);
        self.model.return_work(work);
        replayed?;
        for q in self.qsa_state.iter_mut().flatten() {
            q.rewind(position);
        }
        self.pos = position;
        self.sync_pool_state(b)?;
        self.logits_ready = false;
        if let Some(d) = self.draft.as_ref() { d.bound_at.set(None); }
        self.valid = true;
        self.retained = 0;
        Ok(())
    }
    pub fn verify_batch(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        tokens: &[&[u32]],
    ) -> Result<Vec<Vec<u32>>> {
        Self::verify_batch_kind(b, sessions, tokens, false)
    }
    /// Greedy results may be sparse; full-logit readers reconstruct the canonical head.
    pub fn verify_batch_greedy(
        b: &Backend<'a>, sessions: &mut [&mut Self], tokens: &[&[u32]],
    ) -> Result<Vec<Vec<u32>>> {
        Self::verify_batch_kind(b, sessions, tokens, true)
    }
    fn verify_batch_kind(
        b: &Backend<'a>, sessions: &mut [&mut Self], tokens: &[&[u32]], greedy_head: bool,
    ) -> Result<Vec<Vec<u32>>> {
        if sessions.len() != tokens.len() || sessions.is_empty() || sessions.len() > 16 {
            return Err("invalid MTP cohort".into());
        }
        let count = tokens.iter().try_fold(0usize, |n, t| {
            n.checked_add(t.len()).ok_or("MTP row count overflow")
        })?;
        let _ = crate::flash_next_rows::head_tiles(count)?;
        b.next_rows_profile(true, sessions[0].pos, count, || {
            Self::verify_batch_inner(b, sessions, tokens, greedy_head)
        })
    }
    fn verify_batch_inner(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        tokens: &[&[u32]],
        greedy_head: bool,
    ) -> Result<Vec<Vec<u32>>> {
        for (s, t) in sessions.iter_mut().zip(tokens) {
            s.prepare_verify(b, t)?;
        }
        Self::run_rows_inner(b, sessions, tokens, true, true, greedy_head)?;
        let model = sessions[0].model;
        let held = model.work.borrow();
        let work = held.as_ref().ok_or("missing verifier workspace")?;
        let result = Self::split_results(b, sessions, &work.head_result, tokens.iter().map(|t| t.len()).sum());
        drop(held);
        if result.is_err() {
            for s in sessions.iter_mut() {
                s.valid = false;
            }
        }
        let ids: Vec<_> = result?.into_iter().map(|v| v.0).collect();
        let mut at = 0;
        Ok(tokens
            .iter()
            .map(|t| {
                let row = ids[at..at + t.len()].to_vec();
                at += t.len();
                row
            })
            .collect())
    }
    pub fn verify_mtp(&mut self, b: &Backend<'a>, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        Self::verify_batch(b, &mut [self], &[tokens])?;
        self.decode_logits(b, tokens.len())
    }
    /// The verifier is unchanged; only its readback is a compact exact shortlist.
    pub fn verify_mtp_top_k(&mut self, b: &Backend<'a>, tokens: &[u32], k: usize) -> Result<Vec<Vec<(u32, f32)>>> {
        Self::verify_batch(b, &mut [self], &[tokens])?;
        self.decode_top_k(b, tokens.len(), k)
    }
    /// Reproject only the head from session-owned normalized inputs. The target
    /// layers/KV are never replayed, and later cohorts cannot overwrite this input.
    pub(super) fn canonical_bound_logits(&self, b: &Backend<'a>) -> Result<Option<Vec<Vec<f32>>>> {
        let Some(d) = self.draft.as_ref() else { return Ok(None); };
        let Some((position, count)) = d.bound_at.get() else { return Ok(None); };
        if position != self.pos || !self.valid || !self.logits_ready { return Ok(None); }
        let hidden = d.bound_input.as_ref().ok_or("missing retained canonical head inputs")?;
        let model = self.model;
        let mut work = model.take_work(b, self.capacity, count)?;
        let result = (|| {
            Head::project_next_rows(b, model.weights.device_address("lm_head.weight")?, model.vocabulary(), count,
                hidden, &mut work.head_logits, &mut work.head_partials, &mut work.head_result)?;
            Ok(b.read_vector_range(&work.head_logits, 0, count * model.vocabulary())?
                .chunks_exact(model.vocabulary()).map(|v| v.to_vec()).collect())
        })();
        model.return_work(work);
        result.map(Some)
    }
    pub fn decode_top_k(&mut self, b: &Backend<'a>, rows: usize, k: usize) -> Result<Vec<Vec<(u32, f32)>>> {
        if let Some(logits) = self.canonical_bound_logits(b)? {
            let count = logits.len();
            let d = self.draft.as_mut().ok_or("MTP disabled")?;
            let values: Vec<_> = logits.into_iter().flatten().collect();
            b.write_vector(&mut d.verification.slice(0..values.len())?, &values)?;
            self.head.select_row(b, &d.verification, count - 1)?;
            d.bound_at.set(None);
        }
        if rows == 1 { return self.head.top_k(b, k).map(|v| vec![v]); }
        if rows != self.retained || !self.valid { return Err("missing MTP logits".into()); }
        self.head.top_k_rows(b, &self.draft.as_ref().ok_or("MTP disabled")?.verification, rows, k)
    }
    pub fn decode_logits(&self, b: &Backend<'a>, rows: usize) -> Result<Vec<Vec<f32>>> {
        if rows == 1 {
            return self.logits(b).map(|v| vec![v]);
        }
        if rows != self.retained || !self.valid {
            return Err("missing MTP logits".into());
        }
        if let Some(logits) = self.canonical_bound_logits(b)? { return Ok(logits); }
        Ok(b.read_vector_range(
            &self.draft.as_ref().ok_or("MTP disabled")?.verification,
            0,
            rows * self.model.vocabulary(),
        )?
        .chunks_exact(self.model.vocabulary())
        .map(|r| r.to_vec())
        .collect())
    }
    pub fn commit_mtp(&mut self, b: &Backend<'a>, position: usize) -> Result<()> {
        if !self.valid {
            return Err("MTP session needs reset".into());
        }
        self.flush_draft_feed(b)?;
        if position == self.pos {
            self.retained = 0;
            return Ok(());
        }
        let prefix = self.saved[..self.retained]
            .iter()
            .position(|s| s.position == position)
            .ok_or("MTP prefix was not retained")?;
        self.valid = false;
        let rollback = self.rollback.as_mut().ok_or("MTP rollback missing")?;
        // Restore by ownership exchange, moving zero state bytes. Pool graphs
        // resolve the republished addresses after the accepted GDN prefix replays.
        if rollback.shallow {
            for (live, entry) in self.gdn_state.iter_mut().zip(rollback.gdn.iter_mut()) {
                if let (Some(live), Some(entry)) = (live.as_mut(), entry.as_mut()) { live.restore_history(entry); }
            }
        } else {
            std::mem::swap(&mut self.gdn_state, &mut rollback.gdn);
        }
        let mut work = self.model.take_work(b, self.capacity, 1)?;
        // Replayed rows sit at positions position - prefix .. position (chunk form).
        // VADE_W2_PP=1 front: only layers [0, split) live on this rank.
        let layers = 0..self.model.pp_front_split().unwrap_or(LAYERS);
        let replayed = work.replay_gdn(b, self.model, &mut self.gdn_state, rollback, prefix, position - prefix, layers);
        self.model.return_work(work);
        replayed?;
        let saved = &mut self.saved[prefix];
        self.ple.select_history(&mut saved.ple);
        self.hash = saved.hash.clone();
        for q in self.qsa_state.iter_mut().flatten() {
            q.rewind(position);
        }
        let d = self.draft.as_mut().ok_or("MTP disabled")?;
        b.copy_range(&saved.last, &mut d.last, 0, 0, W)?;
        d.attention.rewind(position.saturating_sub(1));
        self.pos = position;
        self.sync_pool_state(b)?;
        self.logits_ready = false;
        if let Some(d) = self.draft.as_ref() { d.bound_at.set(None); }
        self.valid = true;
        self.retained = 0;
        Ok(())
    }
}
pub(crate) fn draft_experts(
    plan: &engine_model::flash::mixed::Plan,
    mut address: impl FnMut(&engine_model::shapes::TensorEntry) -> Result<u64>,
) -> Result<Vec<spark_gpu_probe::driver::fp4_abi::Expert>> {
    use engine_model::{flash::mixed::ScaleView, safetensors::Dtype};
    use spark_gpu_probe::driver::fp4_abi::{Expert, Matrix};
    for t in &plan.model.shape.tensors {
        if t.name.starts_with("mtp.")
            && t.name.ends_with(".weight")
            && !t.name.starts_with("mtp.layers.0.mlp.experts.")
            && t.dtype != Dtype::Bf16
        {
            return Err(format!(
                "{}: unsupported draft tensor {:?}; expected BF16",
                t.name, t.dtype
            ));
        }
    }
    (0..512)
        .map(|id| {
            let mut matrices = [Matrix::default(); 3];
            for (pi, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                let name = format!("mtp.layers.0.mlp.experts.{id}.{proj}");
                let view = plan
                    .nvfp4(&name)
                    .map_err(|e| format!("{name}: unsupported draft expert: {e}"))?;
                let shape = if pi == 2 { (2560, 640) } else { (640, 2560) };
                let ScaleView::E4m3(scales) = view.scales else {
                    return Err(format!(
                        "{name}: draft expert requires stored E4M3 block scales"
                    ));
                };
                if (view.rows, view.columns) != shape
                    || view.weight.dtype != Dtype::U8
                    || scales.dtype != Dtype::F8E4M3
                    || scales.shape != [shape.0, shape.1 / 16]
                    || view.global.dtype != Dtype::F32
                    || !view.global.shape.is_empty()
                    || view.input.is_some()
                {
                    return Err(format!(
                        "{name}: unsupported draft tensor layout; expected W4A16 NVFP4 {shape:?}"
                    ));
                }
                matrices[pi] = Matrix {
                    weights: address(view.weight)?,
                    scales: address(scales)?,
                    global_scale: address(view.global)?,
                };
            }
            Ok(Expert {
                gate: matrices[0],
                up: matrices[1],
                down: matrices[2],
                width: 640,
                reserved: 0,
            })
        })
        .collect()
}
