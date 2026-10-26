//! Untied, vocabulary-sharded BF16 output projection with FP32 activations.
use super::{blocks::H, Weights};
use engine_model::{flash::plan::Selection, safetensors::Dtype};
use spark_gpu_probe::driver::{Backend, DeviceVector, TopKScratch};
use std::{marker::PhantomData, ops::Range};

type Result<T> = std::result::Result<T, String>;
pub use crate::flash_candidate::{Candidate, VOCABULARY};

/// LBF12 head: the resident head re-encoded in place by flash_next_head_lbf12_encode;
/// (weight, tile flags, row base exponents). Every head launch on that weight decodes the
/// identical BF16 fragments in flash_next_head_lbf12 (same walk, chains and argmax).
static LBF12: std::sync::RwLock<Option<(u64, u64, u64)>> = std::sync::RwLock::new(None);

/// Grid of flash_next_draft_conf_partial (VADE_MTP_CONF).
pub(super) const CONF_CTAS: usize = 96;

/// Optional full-vocabulary screen. Canonical BF16 weights remain resident.
pub(super) struct Bounds<'a> {
    twin: DeviceVector<'a>,
    norms: DeviceVector<'a>,
}
impl Bounds<'_> {
    pub(super) fn bytes(&self) -> u64 { (self.twin.len() + self.norms.len()) as u64 * 4 }
}

pub struct Head<'a, 'w> {
    weight: u64,
    range: Range<u32>,
    logits: DeviceVector<'a>,
    partials: DeviceVector<'a>,
    result: DeviceVector<'a>,
    top_k: Option<TopKScratch<'a>>,
    /// Draft-only MXFP8 image of the scored rows (flash_next_head_mx8), if enabled.
    mx8: Option<&'w DeviceVector<'a>>,
    /// Drafter-only NVFP4 image (flash_next_head_fp4_pack), VADE_FLASHNEXT_DRAFT_FP4=1|2 (unset = 2).
    fp4: Option<&'w DeviceVector<'a>>,
    /// VADE_FLASHNEXT_DRAFT_FP4=2 (the default): rescore the FP4 screen's top four 1024-row chunks from
    /// the MXFP8 twin's values (needs that twin).
    fp4_rescore: bool,
    ready: bool,
    max_ready: bool,
    _weights: PhantomData<&'w ()>,
}
impl<'a, 'w> Head<'a, 'w> {
    pub(super) fn load_bounds(b: &Backend<'a>, weight: u64, rows: usize) -> Result<Bounds<'a>> {
        let twin = Self::load_mx8_draft(b, weight, rows)?.ok_or("head bounds need complete MX8 rows")?;
        let norms = b.vector(rows * 2)?;
        let (mut w, mut t, mut n, mut o) = (weight, twin.device_address(), rows as u32, norms.device_address());
        unsafe {
            b.launch_flash("flash_head_bound_weights", rows.div_ceil(128) as u32, 128,
                &mut [(&mut w as *mut u64).cast(), (&mut t as *mut u64).cast(),
                    (&mut n as *mut u32).cast(), (&mut o as *mut u64).cast()])?;
        }
        let values = b.read_vector_values(&norms)?;
        // Nonfinite norms cannot justify pruning: the shared envelope retains
        // those rows and runs their canonical head. Negative norms are malformed.
        if values.iter().any(|v| *v < 0.0) { return Err("invalid target-head norm bounds".into()); }
        let bounds = Bounds { twin, norms };
        Ok(bounds)
    }

    /// Sparse greedy verification; the session retains inputs for canonical full-logit readers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_bound_rows(b: &Backend<'_>, weight: u64, rows: usize, count: usize,
        hidden: &DeviceVector<'_>, logits: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>,
        result: &mut DeviceVector<'_>, bounds: &Bounds<'_>) -> Result<()> {
        use spark_gpu_probe::kernel_params::{Bf16Rows, FlashBf16Input};
        if !(1..=4).contains(&count) || rows % 64 != 0 || hidden.len() < count * H
            || logits.len() < rows * count || partials.len() < count * 192 * 3 || result.len() < count * 3 {
            return Err("head bound verification shape".into());
        }
        // Existing MX8 group kernels preserve the input's canonical hi/lo terms.
        super::blocks::next_rows_group(b, hidden.device_address(), H, count,
            [(bounds.twin.device_address() | 1, logits.device_address(), rows), (0,0,0), (0,0,0), (0,0,0)])?;
        let (mut x, mut m, mut norm, mut limits, mut scores, mut n, mut part) =
            (hidden.device_address(), count as u32, bounds.norms.device_address(), result.device_address(),
                logits.device_address(), rows as u32, partials.device_address());
        unsafe {
            b.launch_flash("flash_head_bound_input", count as u32, 128,
                &mut [(&mut x as *mut u64).cast(), (&mut m as *mut u32).cast(), (&mut limits as *mut u64).cast()])?;
            b.launch_flash_rows("flash_head_bound_lower", 192, count as u32, 128, 0,
                &mut [(&mut scores as *mut u64).cast(), (&mut norm as *mut u64).cast(), (&mut limits as *mut u64).cast(),
                    (&mut n as *mut u32).cast(), (&mut part as *mut u64).cast()])?;
            b.launch_flash("flash_head_bound_lower_finish", 1, 32,
                &mut [(&mut part as *mut u64).cast(), (&mut m as *mut u32).cast(), (&mut limits as *mut u64).cast()])?;
        }
        b.prepare_flash_dense()?;
        let mut input = FlashBf16Input { input: hidden.device_address(), output: b.flash_dense_input()?,
            tokens: count as u32, cols: H as u32, hc_silu: 0, lanes: 0 };
        let mut p = Bf16Rows { weight, output: logits.device_address(), rows: rows as u32, stride: rows as u32,
            weight2: 0, output2: 0, rows2: 0, stride2: 0, input: input.output, cols: H as u32,
            tokens: count as u32, input_bf16: 1, hc_silu: 0, input_plane_words: 0 };
        let (mut flags, mut bases) = LBF12.read().map_err(|_| "LBF12 registry poisoned")?
            .filter(|&(w, _, _)| w == weight).map_or((0,0), |(_, f, a)| (f,a));
        let mut blocks = 192u32;
        unsafe {
            b.launch_flash("flash_next_head_input", (count.div_ceil(8)*(H/16)*32).div_ceil(128) as u32, 128,
                &mut [(&mut input as *mut FlashBf16Input).cast()])?;
            b.launch_flash("flash_head_bound_exact", blocks, 128,
                &mut [(&mut p as *mut Bf16Rows).cast(), (&mut part as *mut u64).cast(),
                    (&mut flags as *mut u64).cast(), (&mut bases as *mut u64).cast(),
                    (&mut norm as *mut u64).cast(), (&mut limits as *mut u64).cast()])?;
            b.launch_flash_rows("logits_argmax_finish", 1, count as u32, 128, 0,
                &mut [(&mut part as *mut u64).cast(), (&mut limits as *mut u64).cast(), (&mut blocks as *mut u32).cast()])?;
        }
        Ok(())
    }
    fn arena_lengths(rows: usize) -> [usize; 3] {
        [rows.max(1), (rows.div_ceil(1024) * 3).max(1), 3]
    }
    pub(super) fn reservation_bytes(rows: usize) -> usize {
        Self::arena_lengths(rows).into_iter()
            .map(|n| (n * 4).div_ceil(64) * 64).sum::<usize>()
            + Self::top_k_reservation_bytes(rows)
    }
    pub(super) fn top_k_reservation_bytes(rows: usize) -> usize {
        (192 * rows.div_ceil(1024)).next_power_of_two() * 4 + 1024
    }
    pub fn load(b: &Backend<'a>, w: &'w Weights<'_>) -> Result<Self> {
        b.require_layer_kernels()?;
        let range = &w.plan.vocabulary;
        let rows = (range.end - range.start) as usize;
        let weight = if rows == 0 { 0 } else {
            let t = w.tensor("lm_head.weight")?;
            if t.tensor.dtype != Dtype::Bf16 || t.tensor.shape != [VOCABULARY as u64, H as u64]
                || t.bytes != rows * H * 2
                || t.selection != Selection::Rows(vec![range.clone()])
                    && !(t.selection == Selection::All && range.start == 0 && rows == VOCABULARY as usize) {
                return Err("lm_head.weight must contain this rank's official BF16 vocabulary rows".into());
            }
            w.device_address("lm_head.weight")?
        };
        let mut vectors = b.vector_arena(&Self::arena_lengths(rows))?.into_iter();
        Ok(Self { weight, range: range.start as u32..range.end as u32,
            logits: vectors.next().unwrap(), partials: vectors.next().unwrap(),
            result: vectors.next().unwrap(), top_k: None, mx8: None, fp4: None, fp4_rescore: false, ready: false, max_ready: false, _weights: PhantomData })
    }
    pub fn vocabulary(&self) -> Range<u32> { self.range.clone() }
    /// Draft proposals only: score the first `rows` vocabulary ids (a multiple of 8; the
    /// resident layout keeps rows in groups of eight, so a row prefix is a weight prefix).
    /// Verification keeps the full head, so outputs never depend on this.
    pub(super) fn limit_rows(&mut self, rows: u32) {
        if self.range.start == 0 && rows > 0 && rows % 8 == 0 && rows < self.range.end {
            self.range.end = rows;
        }
    }
    pub(super) fn draft_rows(vocabulary: usize, requested: Option<u32>) -> usize {
        requested.filter(|&n| n > 0 && n % 8 == 0 && (n as usize) < vocabulary)
            .map_or(vocabulary, |n| n as usize)
    }
    pub(super) fn mx8_draft_bytes(rows: usize) -> usize {
        if rows <= 640 || rows % 64 != 0 { return 0; }
        ((rows * H).div_ceil(256) * 256 + rows / 8 * (H / 32) * 8).div_ceil(64) * 64
    }
    /// Optional draft-only head: build an MXFP8 copy of the scored rows and project
    /// through flash_mx8_mma_latency at M <= 4 (half the head bytes per drafted token).
    /// Verification keeps the BF16 head, so output bytes never depend on it; only which
    /// tokens are proposed can change (accept rate is the measured cost).
    pub(super) fn load_mx8_draft(b: &Backend<'a>, weight: u64, rows: usize) -> Result<Option<DeviceVector<'a>>> {
        use spark_gpu_probe::kernel_params::FlashBf16Pack;
        let bytes = Self::mx8_draft_bytes(rows);
        if bytes == 0 { return Ok(None); }
        let twin = b.vector(bytes.div_ceil(4))?;
        let mut pack = FlashBf16Pack { source: weight, weight: twin.device_address(), rows: rows as u32, cols: H as u32 };
        unsafe {
            b.launch_flash("flash_next_head_mx8", (rows * (H / 32)).div_ceil(128) as u32, 128,
                &mut [(&mut pack as *mut FlashBf16Pack).cast()])?;
        }
        b.synchronize()?;
        Ok(Some(twin))
    }
    /// Bytes of the drafter-only NVFP4 image of `rows` head rows: E2M1 pairs, then at the
    /// next 256-byte boundary one E4M3 scale per 16 columns.
    pub(super) fn fp4_draft_bytes(rows: usize) -> usize {
        if rows == 0 || rows % 2 != 0 { return 0; }
        ((rows * H / 2).div_ceil(256) * 256 + rows * (H / 16)).div_ceil(64) * 64
    }
    /// Drafter head (VADE_FLASHNEXT_DRAFT_FP4=1|2, unset = 2, =0 off; crate::flash_draft_stack):
    /// an NVFP4 image of the scored rows read by flash_next_head_fp4 for one-chat proposals
    /// (0.36 GB per drafted token instead of the MXFP8 twin's 0.66 GB). Verification keeps
    /// the full head, so output bytes never depend on it; only which tokens are proposed can
    /// change (accept rate is the measured cost). Build before Head::encode_lbf12.
    pub(super) fn load_fp4_draft(b: &Backend<'a>, weight: u64, rows: usize) -> Result<Option<DeviceVector<'a>>> {
        use spark_gpu_probe::kernel_params::FlashBf16Pack;
        let bytes = Self::fp4_draft_bytes(rows);
        if bytes == 0 || weight == 0 { return Ok(None); }
        if !b.has_flash("flash_next_head_fp4_pack") || !b.has_flash("flash_next_head_fp4") {
            return Err("VADE_FLASHNEXT_DRAFT_FP4=1|2 (unset = 2) needs flash_next_head_fp4 in the PTX; =0 turns it off".into());
        }
        let twin = b.vector(bytes.div_ceil(4))?;
        let mut pack = FlashBf16Pack { source: weight, weight: twin.device_address(), rows: rows as u32, cols: H as u32 };
        unsafe {
            b.launch_flash("flash_next_head_fp4_pack", (rows * (H / 16)).div_ceil(128) as u32, 128,
                &mut [(&mut pack as *mut FlashBf16Pack).cast()])?;
        }
        b.synchronize()?;
        Ok(Some(twin))
    }
    pub(super) fn use_fp4_draft(&mut self, twin: Option<&'w DeviceVector<'a>>) {
        self.fp4 = twin;
        self.fp4_rescore = twin.is_some() && crate::flash_draft_stack::draft_fp4() == Ok(2);
    }
    /// Bytes of the LBF12 tile flags and row base exponents for `rows` vocabulary rows.
    pub(super) fn lbf12_bytes(rows: usize) -> usize { rows / 8 * 16 + rows / 8 * (H / 32) * 8 }
    /// Re-encode the resident head in place as LBF12 (VADE_HEAD_LBF12=0 keeps BF16). Lossless:
    /// a slot whose rows' exponents span <= 14 becomes 12 bits per weight, any other stays raw,
    /// and the head launch decodes the identical BF16 bits, so logits and ids are unchanged.
    /// Call after every reader of the BF16 head (the MXFP8 drafter twin) has run.
    pub(super) fn encode_lbf12(b: &Backend<'a>, weight: u64, rows: usize) -> Result<Option<DeviceVector<'a>>> {
        use spark_gpu_probe::kernel_params::FlashBf16Pack;
        if std::env::var("VADE_HEAD_LBF12").is_ok_and(|v| v == "0") || weight == 0 || rows == 0 || rows % 8 != 0
            || !b.has_flash("flash_next_head_lbf12_encode") {
            return Ok(None);
        }
        let flag_bytes = rows / 8 * 16;
        let mut v = b.vector(Self::lbf12_bytes(rows) / 4)?;
        b.write_vector_bytes(&mut v, &vec![0u8; Self::lbf12_bytes(rows)])?;
        let (mut flags, mut bases) = (v.device_address(), v.device_address() + flag_bytes as u64);
        let mut pack = FlashBf16Pack { source: 0, weight, rows: rows as u32, cols: H as u32 };
        unsafe {
            b.launch_flash("flash_next_head_lbf12_encode", (rows / 8 * (H / 32) * 32).div_ceil(128) as u32, 128,
                &mut [(&mut pack as *mut FlashBf16Pack).cast(), (&mut flags as *mut u64).cast(), (&mut bases as *mut u64).cast()])?;
        }
        b.synchronize()?;
        *LBF12.write().map_err(|_| "LBF12 registry poisoned")? = Some((weight, flags, bases));
        Ok(Some(v))
    }
    pub(super) fn use_mx8_draft(&mut self, twin: Option<&'w DeviceVector<'a>>) {
        self.mx8 = twin;
    }
    /// VADE_FP8_DRAFT_MX8 (unset = 1, 0 = off): the FP8 rank path's drafts score this rank's
    /// vocabulary rows from an MXFP8 image (load_mx8_draft), half the BF16 head's bytes per
    /// drafted token. Verification keeps the BF16 head: only proposals can change, never ids.
    /// The image's pack reads the Flash-Next head layout, so the resident rows are placed in
    /// that layout in a temporary copy first.
    pub(super) fn fp8_draft_twin(b: &Backend<'a>, w: &Weights<'_>) -> Result<Option<DeviceVector<'a>>> {
        let on = match std::env::var("VADE_FP8_DRAFT_MX8") {
            Err(std::env::VarError::NotPresent) => true,
            Ok(v) if v == "0" => false, Ok(v) if v == "1" => true,
            _ => return Err("VADE_FP8_DRAFT_MX8 must be 0 or 1".into()),
        };
        let rows = (w.plan.vocabulary.end - w.plan.vocabulary.start) as usize;
        if !on || Self::mx8_draft_bytes(rows) == 0 { return Ok(None); }
        let mut placed = vec![0u8; rows * H * 2];
        w.read_bf16_matrix("lm_head.weight", H, |r, c| spark_gpu_probe::next_math::head_weight(r, c, H), &mut placed)?;
        let mut copy = b.vector(rows * H / 2)?;
        b.write_vector_bytes(&mut copy, &placed)?;
        let twin = Self::load_mx8_draft(b, copy.device_address(), rows);
        // The pack has completed (load_mx8_draft drains); return the staging heap to the OS.
        drop((copy, placed));
        b.trim_host_memory();
        twin
    }
    /// A drafted token's logits: the MXFP8 twin when one is lent, else the BF16 head.
    pub fn project_draft(&mut self, b: &Backend<'_>, hidden: &DeviceVector<'_>) -> Result<()> {
        let Some(twin) = self.mx8.map(|v| v.device_address()) else { return self.project(b, hidden) };
        self.ready = false; self.max_ready = false;
        if hidden.len() != H || hidden.is_segmented() {
            return Err("Flash draft head requires a contiguous 2560-wide FP32 mixture".into());
        }
        if !self.range.is_empty() {
            super::blocks::mx8_rows(b, twin, hidden.device_address(), self.logits.device_address(), self.range.len(), H, 1)?;
        }
        self.ready = true;
        Ok(())
    }
    pub fn weight_bytes(&self) -> usize { self.range.len() * H * 2 }
    pub fn device_bytes(&self) -> usize {
        let mut allocations = Vec::new();
        if let Some(scratch) = &self.top_k { scratch.allocations(&mut allocations); }
        allocations.sort_unstable_by_key(|v| v.0);
        allocations.dedup_by_key(|v| v.0);
        // The immutable MXFP8 twin is owned and accounted once by NextModel.
        [self.logits.len(), self.partials.len(), self.result.len()].into_iter()
            .map(|n| (n * 4).div_ceil(64) * 64).sum::<usize>() + allocations.iter().map(|v| v.1).sum::<usize>()
    }
    /// Input is the final HC mixture; there is no separate final RMSNorm.
    pub fn project(&mut self, b: &Backend<'_>, hidden: &DeviceVector<'_>) -> Result<()> {
        self.ready = false; self.max_ready = false;
        if hidden.len() != H || hidden.is_segmented() {
            return Err("Flash output head requires a contiguous 2560-wide FP32 mixture".into());
        }
        if !self.range.is_empty() {
            b.profile_output_head(|| super::blocks::bf16_rows(b, self.weight, hidden.device_address(), self.logits.device_address(),
                self.range.len(), H, 1, self.range.len()))?;
        }
        self.ready = true;
        Ok(())
    }
    /// Project independent mixtures together, then retain each session's own
    /// logits for the existing greedy or sampled output path.
    pub(super) fn project_batch(b: &Backend<'_>, heads: &mut [&mut Self],
        hidden: &DeviceVector<'_>, logits: &mut DeviceVector<'_>) -> Result<()> {
        let first = heads.first().ok_or("empty Flash output batch")?;
        let (weight, range) = (first.weight, first.range.clone());
        let count = heads.len();
        let rows = range.len();
        if hidden.len() != count * H || hidden.is_segmented() || logits.is_segmented()
            || logits.len() < count * rows || heads.iter().any(|h| h.weight != weight || h.range != range) {
            return Err("Flash output batch needs one vocabulary shard and contiguous mixtures".into());
        }
        for head in heads.iter_mut() { head.ready = false; head.max_ready = false; }
        if rows != 0 {
            b.profile_output_head(|| super::blocks::bf16_rows(b, weight, hidden.device_address(), logits.device_address(), rows, H, count, rows))?;
            for (i, head) in heads.iter_mut().enumerate() {
                b.copy_range(logits, &mut head.logits, i * rows, 0, rows)?;
            }
        }
        for head in heads { head.ready = true; }
        Ok(())
    }
    /// Mixed-checkpoint rows share one M16 weight pass, including M=1. Full
    /// logits remain available to the serving sampler; greedy tie rules stay unchanged.
    pub(super) fn project_next_batch(b: &Backend<'_>, heads: &mut [&mut Self],
        hidden: &DeviceVector<'_>, logits: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, result: &mut DeviceVector<'_>) -> Result<()> {
        let first = heads.first().ok_or("empty Flash-Next output batch")?;
        let (weight, range, count) = (first.weight, first.range.clone(), heads.len());
        if count > 16 || hidden.len() != count*H || logits.len() < count*range.len()
            || heads.iter().any(|h| h.weight != weight || h.range != range) { return Err("Flash-Next head row shape".into()); }
        for head in heads.iter_mut() { head.ready = false; head.max_ready = false; }
        if let (Some(twin), 1, 0) = (heads[0].fp4.map(|v| v.device_address()), count, range.start) {
            // VADE_FLASHNEXT_DRAFT_FP4=1 (drafter only): the NVFP4 image, then main's argmax call.
            let head = &mut *heads[0];
            let rows = range.len();
            let (mut w, mut x, mut y, mut n) = (twin, hidden.device_address(), head.logits.device_address(), rows as u32);
            unsafe {
                b.launch_flash("flash_next_head_fp4", 192, 256, &mut [(&mut w as *mut u64).cast(),
                    (&mut x as *mut u64).cast(), (&mut y as *mut u64).cast(), (&mut n as *mut u32).cast()])?;
            }
            let chunks = rows.div_ceil(1024) * 3;
            if chunks < head.partials.len() {
                b.enqueue_argmax_device(&head.logits.slice(0..rows)?, &mut head.partials.slice(0..chunks)?, result, 0)?;
            } else {
                b.enqueue_argmax_device(&head.logits.slice(0..rows)?, &mut head.partials, result, 0)?;
            }
            // =2: the screen's four best chunks, rescored from the MXFP8 twin's values (FP32),
            // replace the FP4 pick; `logits` (this call's scratch) holds 4096 candidates + ids.
            if let (true, Some(mx8)) = (head.fp4_rescore, head.mx8.map(|v| v.device_address())) {
                if rows % 64 != 0 || logits.len() < 4096 + 4 || logits.is_segmented() || result.len() < 3 {
                    return Err("FP4 drafter screen shape".into());
                }
                let (mut p, mut c, mut sel) = (head.partials.device_address(), rows.div_ceil(1024) as u32, logits.device_address() + 4096 * 4);
                let (mut t, mut n, mut x, mut cand, mut res) = (mx8, rows as u32, hidden.device_address(), logits.device_address(), result.device_address());
                unsafe {
                    b.launch_flash("flash_next_head_fp4_select", 1, 32, &mut [(&mut p as *mut u64).cast(), (&mut c as *mut u32).cast(), (&mut sel as *mut u64).cast()])?;
                    b.launch_flash("flash_next_head_fp4_rescore", 64, 256, &mut [(&mut t as *mut u64).cast(), (&mut n as *mut u32).cast(),
                        (&mut x as *mut u64).cast(), (&mut sel as *mut u64).cast(), (&mut cand as *mut u64).cast()])?;
                    b.launch_flash("flash_next_head_fp4_final", 1, 256, &mut [(&mut cand as *mut u64).cast(), (&mut sel as *mut u64).cast(), (&mut res as *mut u64).cast()])?;
                }
            }
            b.copy_range(result, &mut head.result, 0, 0, 3)?;
            head.ready = true; head.max_ready = true;
            return Ok(());
        }
        if let (Some(twin), true) = (first_mx8(heads), count <= 4) {
            super::blocks::mx8_rows(b, twin, hidden.device_address(), logits.device_address(), range.len(), H, count)?;
            // A drafter limited to a vocabulary prefix (VADE_FLASHNEXT_DRAFT_VOCAB) argmaxes over
            // that prefix's chunks only; the full head keeps main's call on the whole partials vector.
            let chunks = range.len().div_ceil(1024) * 3;
            for (row, head) in heads.iter_mut().enumerate() {
                b.copy_range(logits, &mut head.logits, row*range.len(), 0, range.len())?;
                if chunks < head.partials.len() {
                    b.enqueue_argmax_device(&head.logits.slice(0..range.len())?, &mut head.partials.slice(0..chunks)?, result, row)?;
                } else {
                    b.enqueue_argmax_device(&head.logits.slice(0..range.len())?, &mut head.partials, result, row)?;
                }
                b.copy_range(result, &mut head.result, row*3, 0, 3)?;
                head.ready = true; head.max_ready = true;
            }
            return Ok(());
        }
        Self::project_next_rows(b,weight,range.len(),count,hidden,logits,partials,result)?;
        for (row, head) in heads.iter_mut().enumerate() {
            b.copy_range(logits, &mut head.logits, row*range.len(), 0, range.len())?;
            b.copy_range(result, &mut head.result, row*3, 0, 3)?;
            head.ready = true; head.max_ready = true;
        }
        Ok(())
    }
    /// Cohort drafter head (VADE_MTP_BATCH): `count` mixtures scored by this drafter head,
    /// argmax row i into result row i. The MXFP8 twin is read once per 8 rows on the latency
    /// entry a chat's own head runs alone (one per-row K16 graph at every M; every drafter
    /// twin is the same image of the same rows); without a twin the BF16 rows keep their
    /// one-row folds (project_next_rows).
    pub(super) fn project_draft_rows(&self, b: &Backend<'_>, hidden: &DeviceVector<'_>, count: usize,
        logits: &mut DeviceVector<'_>, partials: &mut DeviceVector<'_>, result: &mut DeviceVector<'_>) -> Result<()> {
        let rows = self.range.len();
        if count == 0 || count > 16 || rows == 0 || hidden.len() < count * H || logits.len() < count * rows
            || result.len() < count * 3 {
            return Err("Flash-Next draft head row shape".into());
        }
        let Some(twin) = self.mx8.as_ref().map(|v| v.device_address()) else {
            return Self::project_next_rows(b, self.weight, rows, count, hidden, logits, partials, result);
        };
        for start in (0..count).step_by(8) {
            super::blocks::mx8_rows(b, twin, hidden.device_address() + (start * H * 4) as u64,
                logits.device_address() + (start * rows * 4) as u64, rows, H, (count - start).min(8))?;
        }
        // The one-row argmax takes partials of exactly one row's chunks (as head.partials
        // is alone); the cohort buffer is wider, so each row reuses its prefix in stream order.
        let mut partials = partials.slice(0..rows.div_ceil(1024) * 3)?;
        for row in 0..count {
            b.enqueue_argmax_device(&logits.slice(row * rows..(row + 1) * rows)?, &mut partials, result, row)?;
        }
        Ok(())
    }
    pub(super) fn project_next_rows(
        b: &Backend<'_>,
        weight: u64,
        rows: usize,
        count: usize,
        hidden: &DeviceVector<'_>,
        logits: &mut DeviceVector<'_>,
        partials: &mut DeviceVector<'_>,
        result: &mut DeviceVector<'_>,
    ) -> Result<()> {
        use spark_gpu_probe::kernel_params::{Bf16Rows, FlashBf16Input};
        let tiles = crate::flash_next_rows::head_tiles(count)?;
        if hidden.is_segmented()
            || logits.is_segmented()
            || partials.is_segmented()
            || result.is_segmented()
            || hidden.len() < count * H
            || logits.len() < count * rows
            || rows == 0
            || !rows.is_multiple_of(8)
            || partials.len() < count * 192 * 3
            || result.len() < count * 3
        {
            return Err("Flash-Next head selection extent".into());
        }
        b.prepare_flash_dense()?;
        let lbf12 = (*LBF12.read().map_err(|_| "LBF12 registry poisoned")?).filter(|&(w, _, _)| w == weight);
        // Each slice packs its own complete groups of eight. Weight addresses,
        // K-range folds and tie order are identical to the one-row projection.
        for tile in tiles {
            let width = tile.len();
            let mut input = FlashBf16Input {
                input: hidden.device_address() + (tile.start * H * 4) as u64,
                output: b.flash_dense_input()?,
                tokens: width as u32,
                cols: H as u32,
                hc_silu: 0,
                lanes: 0,
            };
            let mut p = Bf16Rows {
                weight,
                output: logits.device_address() + (tile.start * rows * 4) as u64,
                rows: rows as u32,
                stride: rows as u32,
                weight2: 0,
                output2: 0,
                rows2: 0,
                stride2: 0,
                input: input.output,
                cols: H as u32,
                tokens: width as u32,
                input_bf16: 1,
                hc_silu: 0,
                input_plane_words: 0,
            };
            let (mut address, mut output, mut blocks) = (
                partials.device_address() + (tile.start * 192 * 3 * 4) as u64,
                result.device_address() + (tile.start * 3 * 4) as u64,
                192u32,
            );
            unsafe {
                b.launch_flash(
                    "flash_next_head_input",
                    (width.div_ceil(8) * (H / 16) * 32).div_ceil(128) as u32,
                    128,
                    &mut [(&mut input as *mut FlashBf16Input).cast()],
                )?;
                if let Some((_, mut flags, mut bases)) = lbf12 {
                    b.launch_flash(
                        if width > 16 { "flash_next_head32_lbf12" } else { "flash_next_head_lbf12" },
                        blocks,
                        128,
                        &mut [
                            (&mut p as *mut Bf16Rows).cast(),
                            (&mut address as *mut u64).cast(),
                            (&mut flags as *mut u64).cast(),
                            (&mut bases as *mut u64).cast(),
                        ],
                    )?;
                } else {
                    b.launch_flash(
                        if width > 16 { "flash_next_head32" } else { "flash_next_head" },
                        blocks,
                        128,
                        &mut [
                            (&mut p as *mut Bf16Rows).cast(),
                            (&mut address as *mut u64).cast(),
                        ],
                    )?;
                }
                b.launch_flash_rows(
                    "logits_argmax_finish",
                    1,
                    width as u32,
                    128,
                    0,
                    &mut [
                        (&mut address as *mut u64).cast(),
                        (&mut output as *mut u64).cast(),
                        (&mut blocks as *mut u32).cast(),
                    ],
                )?;
            }
        }
        Ok(())
    }
    pub(super) fn publish_next(&mut self,b: &Backend<'_>,logits: &DeviceVector<'_>,result: &DeviceVector<'_>,row: usize) -> Result<()> {
        self.ready=false; self.max_ready=false;
        b.copy_range(logits,&mut self.logits,row*self.range.len(),0,self.range.len())?;
        b.copy_range(result,&mut self.result,row*3,0,3)?;
        self.ready=true; self.max_ready=true; Ok(())
    }
    /// Consecutive verified positions share the vocabulary weight pass.
    pub(super) fn project_rows(&mut self, b: &Backend<'_>, hidden: &DeviceVector<'_>,
        logits: &mut DeviceVector<'_>, count: usize) -> Result<()> {
        self.ready = false; self.max_ready = false;
        if count == 0 || hidden.len() < count * H || logits.len() < count * self.range.len() {
            return Err("invalid Flash verification head rows".into());
        }
        if !self.range.is_empty() {
            b.profile_output_head(|| super::blocks::bf16_rows(b, self.weight, hidden.device_address(), logits.device_address(),
                self.range.len(), H, count, self.range.len()))?;
        }
        Ok(())
    }
    pub(super) fn select_row(&mut self, b: &Backend<'_>, logits: &DeviceVector<'_>, row: usize) -> Result<()> {
        if !self.range.is_empty() { b.copy_range(logits, &mut self.logits, row * self.range.len(), 0, self.range.len())?; }
        self.ready = true; self.max_ready = false;
        Ok(())
    }
    /// Transfer only the GPU winner; full rows are downloaded only on request.
    pub fn local_max(&mut self, b: &Backend<'_>) -> Result<Option<Candidate>> {
        if !self.ready { return Err("Flash head has no projected logits".into()); }
        if self.range.is_empty() { return Ok(None); }
        let (token, logit) = if self.max_ready { b.read_argmax_results(&self.result, 1)?.remove(0) }
            else { b.argmax_device(&self.logits, &mut self.partials, &mut self.result)? };
        Ok(Some(Candidate { token: self.range.start + token, logit }))
    }
    /// VADE_MTP_CONF (drafter policy only): this one-row draft's argmax plus its softmax
    /// probability over the scored rows, read back in the argmax's place (one 16-byte read).
    /// `partials` holds >= CONF_CTAS floats, `out` exactly 4.
    pub(super) fn draft_confidence(&mut self, b: &Backend<'_>, partials: &DeviceVector<'_>, out: &DeviceVector<'_>)
        -> Result<(Candidate, f32)> {
        if !self.ready || !self.max_ready || self.range.is_empty() || partials.len() < CONF_CTAS || out.len() != 4 {
            return Err("draft confidence needs a projected one-row draft".into());
        }
        let rows = self.range.len();
        let (mut l, mut n, mut r) = (self.logits.device_address(), rows as u32, self.result.device_address());
        let (mut p, mut c, mut o) = (partials.device_address(), CONF_CTAS as u32, out.device_address());
        unsafe {
            b.launch_flash("flash_next_draft_conf_partial", CONF_CTAS as u32, 256, &mut [(&mut l as *mut u64).cast(),
                (&mut n as *mut u32).cast(), (&mut r as *mut u64).cast(), (&mut p as *mut u64).cast()])?;
            b.launch_flash("flash_next_draft_conf_final", 1, 32, &mut [(&mut p as *mut u64).cast(),
                (&mut c as *mut u32).cast(), (&mut r as *mut u64).cast(), (&mut o as *mut u64).cast()])?;
        }
        let v = b.read_vector_values(out)?;
        if v[2] != 1.0 || !v[0].is_finite() || !v[1].is_finite() { return Err("nonfinite logits".into()); }
        Ok((Candidate { token: self.range.start + v[1] as u32, logit: v[0] }, if v[3].is_finite() { v[3] } else { 0.0 }))
    }
    pub(super) fn enqueue_max(&mut self, b: &Backend<'_>, result: &mut DeviceVector<'_>, row: usize) -> Result<()> {
        if !self.ready || self.range.is_empty() { return Err("Flash head has no projected vocabulary".into()); }
        b.enqueue_argmax_device(&self.logits, &mut self.partials, result, row)
    }
    /// Stage this row's projected shard at `offset` of a batch readback buffer.
    pub(super) fn copy_logits(&self, b: &Backend<'_>, destination: &mut DeviceVector<'_>, offset: usize) -> Result<()> {
        if !self.ready { return Err("Flash head has no projected logits".into()); }
        if self.range.is_empty() { return Ok(()); }
        b.copy_range(&self.logits, destination, 0, offset, self.range.len())
    }
    pub fn logits(&self, b: &Backend<'_>) -> Result<Vec<f32>> {
        if !self.ready { return Err("Flash head has no projected logits".into()); }
        if self.range.is_empty() { return Ok(Vec::new()); }
        b.read_vector(&self.logits)
    }
    pub fn top_k(&mut self, b: &Backend<'a>, k: usize) -> Result<Vec<(u32, f32)>> {
        if !self.ready { return Err("Flash head has no projected logits".into()); }
        if self.range.is_empty() { return Ok(Vec::new()); }
        let values = b.top_k_device(&self.logits, &mut self.top_k, 1, k.min(self.range.len()))?
            .pop().ok_or("missing Flash top-k row")??;
        Ok(values.into_iter().map(|(id, value)| (self.range.start + id, value)).collect())
    }
    /// Reuse the existing one-row scratch for retained canonical verifier rows.
    pub(super) fn top_k_rows(&mut self, b: &Backend<'a>, logits: &DeviceVector<'a>,
        rows: usize, k: usize) -> Result<Vec<Vec<(u32, f32)>>> {
        let width = self.range.len();
        (0..rows).map(|row| {
            let view = logits.slice(row * width..(row + 1) * width)?;
            let values = b.top_k_device(&view, &mut self.top_k, 1, k.min(width))?
                .pop().ok_or("missing Flash verifier top-k row")??;
            Ok(values.into_iter().map(|(id, value)| (self.range.start + id, value)).collect())
        }).collect()
    }
}
fn first_mx8(heads: &[&mut Head<'_, '_>]) -> Option<u64> {
    let twin = heads.first()?.mx8.as_ref()?.device_address();
    heads.iter().all(|h| h.mx8.as_ref().map(|v| v.device_address()) == Some(twin)).then_some(twin)
}

#[cfg(test)]
mod tests {
    use super::Head;
    /// The FP8 draft twin's placement (resident rows into the Flash-Next head layout the MXFP8
    /// pack reads) is a bijection over a whole shard: every weight lands once, none is lost.
    #[test]
    fn fp8_draft_twin_placement_is_a_bijection() {
        let (rows, cols) = (704, super::H);
        let mut seen = vec![false; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let to = spark_gpu_probe::next_math::head_weight(r, c, cols);
                assert!(to < rows * cols && !seen[to], "row {r} col {c}");
                seen[to] = true;
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                assert!(spark_gpu_probe::kernel_params::flash_bf16_element(r, c, cols, rows) < rows * cols);
            }
        }
    }
    #[test]
    fn shared_draft_twin_accounting_and_prefix_fallback() {
        assert_eq!(Head::mx8_draft_bytes(248320), 655_564_800);
        assert_eq!(Head::mx8_draft_bytes(65536), 173_015_040);
        for rows in (64..=640).step_by(64) { assert_eq!(Head::mx8_draft_bytes(rows), 0); }
        assert_eq!(Head::mx8_draft_bytes(648), 0);
        assert_eq!(Head::mx8_draft_bytes(704), 1_858_560);
        for requested in [None, Some(0), Some(7), Some(248328)] {
            assert_eq!(Head::draft_rows(248320, requested), 248320);
        }
        assert_eq!(Head::draft_rows(248320, Some(64)), 64);
        // Per-session scratch remains separate from the model's accounted twin.
        assert_eq!(Head::reservation_bytes(248320), 1_259_456);
    }
}

