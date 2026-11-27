//! Dense prompt tiles, mixed planes and one-shot offers. Combined mechanical move.
use super::*;

impl<'a> Backend<'a> {
    /// One stable activation tile, shared by projections on the ordered stream.
    /// Reserve at session creation so batch graphs never capture an allocation.
    pub fn prepare_flash_dense(&self) -> Result<(), String> {
        let mut input = self.flash_input.borrow_mut();
        if input.is_none() {
            let n = crate::flash_dense_reserve_bytes() / 4;
            let mut v = self.vector(n + 64)?;
            self.write_vector_range(&mut v, n, &[0.0; 64])?;
            *input = Some(v);
        }
        Ok(())
    }
    pub fn flash_dense_bytes(&self) -> usize {
        self.flash_input.borrow().as_ref().map_or(0, |v| v.len().saturating_sub(64) * 4)
    }
    pub fn flash_mixed_bytes(&self) -> usize {
        self.flash_mixed.borrow().as_ref().map_or(0, |v| v.len() * 4)
    }
    /// Stable operand address for a fused producer on the same ordered stream.
    pub fn flash_dense_input(&self) -> Result<u64, String> {
        self.flash_input.borrow().as_ref().map(|v| v.device_address())
            .ok_or_else(|| "Flash dense input was not reserved before capture".into())
    }
    /// Second dense tile for a producer that writes the next projections' planes while the
    /// first tile still holds its own operand. Holds HC up output for prompt rows and decode M <= 8.
    pub fn flash_mixed_planes(&self) -> Result<u64, String> {
        let mut planes = self.flash_mixed.borrow_mut();
        if planes.is_none() { *planes = Some(self.vector(crate::flash_mixed_reserve_bytes() / 4)?); }
        planes.as_ref().map(|v| v.device_address()).ok_or_else(|| "Flash mixed planes missing".into())
    }
    /// `planes` hold the dense hi/lo planes of `input` [tokens][cols]: the next dense pack of
    /// exactly that input returns them instead of packing. Every pack consumes the offer.
    pub fn offer_dense_planes(&self, input: u64, cols: usize, tokens: usize, planes: u64) {
        self.dense_offer.set(Some((input, cols, tokens, planes)));
    }
    /// Drop an offer whose input rows leave the workspace (a bank tile, swapped out before its
    /// next pack): the next pack then forms its planes from the FP32 rows.
    pub fn drop_dense_offer(&self) { self.dense_offer.set(None); }
    pub fn pack_flash_dense_input(&self, input: u64, cols: usize, tokens: usize) -> Result<u64, String> {
        self.pack_flash_input(input, cols, tokens, false, 32)
    }
    /// Pack for flash_bf16_mma_latency(_group) only: at K % 64 == 0 and M <= 16 its
    /// fast path stages flash_latency_lanes(M) lanes of each K16 tile, so only those
    /// fragments are written (same bytes; the M1 pack writes 1/8 of the tile).
    pub fn pack_flash_latency_input(&self, input: u64, cols: usize, tokens: usize) -> Result<u64, String> {
        let lanes = if tokens <= 16 && cols.is_multiple_of(64) { crate::kernel_params::flash_latency_lanes(tokens) } else { 32 };
        self.pack_flash_input(input, cols, tokens, false, lanes)
    }
    /// One decode HC block's injection + down group (flash_hc_down_latency_group), which also
    /// activates the down rows and writes their packed planes at `planes`.
    pub fn flash_hc_down_group(&self, p: &mut crate::kernel_params::Bf16Pairs, mut planes: u64) -> Result<(), String> {
        crate::kernel_params::hc_down_group_check(p, planes, self.flash_dense_input()?, self.flash_dense_bytes())?;
        let blocks = p.pairs[0].rows.div_ceil(8) + p.pairs[0].rows2.div_ceil(8);
        // SAFETY: p and planes outlive the launch's argument copy. The check above holds the entry's
        // M, K, weight, pairs[1] and planes list; this call passes its grid and shared size.
        unsafe { self.launch_flash_rows("flash_hc_down_latency_group", blocks, 1, 128, 4 * crate::kernel_params::LATENCY_RING as u32,
            &mut [(p as *mut crate::kernel_params::Bf16Pairs).cast(), (&mut planes as *mut u64).cast()]) }
    }
    /// Single-launch 48-SM 4-way split-K BF16 decode HC down + injection group for M=3..=16
    /// (164 CTAs x 32 threads, GPU-fenced atomic last-CTA merge + hc_sigmoid + hc_planes!).
    pub fn flash_hc_down_splitk_group(&self, p: &mut crate::kernel_params::Bf16Pairs, mut planes: u64) -> Result<(), String> {
        let m = p.pairs[0].tokens as usize;
        let out_plane_bytes = (2 * m.div_ceil(16) * (320 / 16) * 128 * 4) as u64;
        let mut partials = planes.checked_add(out_plane_bytes).ok_or("flash_hc_down_splitk_group: partials address overflow")?;
        let tile = self.flash_dense_input()?;
        let tile_bytes = self.flash_dense_bytes();
        crate::hc_down_splitk_check(p, planes, partials, tile, tile_bytes)?;
        let tasks = p.pairs[0].rows.div_ceil(8) + p.pairs[0].rows2.div_ceil(8);
        if tasks > 64 {
            return Err("flash_hc_down_splitk_group: tasks exceed reserved 64 counters".into());
        }
        let mut counters = tile.checked_add(tile_bytes as u64).ok_or("flash_hc_down_splitk_group: counter address overflow")?;
        let shared = if m <= 4 { 6144u32 } else { 10240u32 };
        let entry = self.chain_entry(m, "flash_hc_down_splitk_fused", "flash_hc_down_splitk_fused_chain");
        // SAFETY: hc_down_splitk_check guarantees M in 3..=16, K % 64 == 0, plain BF16 weights,
        // rows2 == 320, pairs[1] empty, and disjoint in-bounds input/planes/partials inside tile;
        // `counters` points to the 64 zero-initialized u32 words reserved past tile_bytes in prepare_flash_dense.
        unsafe {
            self.launch_flash_rows(entry, tasks * 4, 1, 32, shared,
                &mut [
                    (p as *mut crate::kernel_params::Bf16Pairs).cast(),
                    (&mut partials as *mut u64).cast(),
                    (&mut planes as *mut u64).cast(),
                    (&mut counters as *mut u64).cast(),
                ])
        }
    }
    /// Batched drafter NVFP4 half-screen + MXFP8 rescore for 1..=16 tokens (bit-identical to per-row C1D).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_next_head_fp4_draft_batch(
        &self,
        fp4: u64,
        mx8: u64,
        hidden: &DeviceVector<'_>,
        logits: &mut DeviceVector<'_>,
        partials: &mut DeviceVector<'_>,
        result: &mut DeviceVector<'_>,
        rows: usize,
        count: usize,
    ) -> Result<(), String> {
        const H: usize = 2560;
        if fp4 == 0 || mx8 == 0 || !(1..=16).contains(&count) || rows < 4096 || !rows.is_multiple_of(64)
            || rows.div_ceil(256) > 1024
            || hidden.is_segmented() || logits.is_segmented() || partials.is_segmented() || result.is_segmented()
            || hidden.len() < count * H || logits.len() < count * rows
            || partials.len() < count * 16 || result.len() < count * 3
        {
            return Err("flash_next_head_fp4_draft_batch: launch outside its safety list".into());
        }
        let mut n = rows as u32;
        for start in (0..count).step_by(8) {
            let m = (count - start).min(8);
            let mut x = hidden.device_address() + (start * H * 4) as u64;
            let mut sums = logits.device_address();
            let mut sel = partials.device_address();
            let mut cand = logits.device_address();
            let mut res = result.device_address() + (start * 3 * 4) as u64;
            let (mut w, mut t) = (fp4, mx8);
            // SAFETY: The validation above guarantees contiguous buffers covering m <= 8 rows of hidden [count][2560],
            // logits [count][rows] (reused as cand [m][4096] since rows >= 4096), partials [count][16], and
            // result [count][3], with rows % 64 == 0 and <= 1024 tiles; all stack args outlive each launch.
            unsafe {
                if m == 1 {
                    let mut half = 0b01_0101_0101u32;
                    self.launch_flash("flash_next_head_fp4", 192, 256, &mut [(&mut w as *mut u64).cast(), (&mut x as *mut u64).cast(),
                        (&mut sums as *mut u64).cast(), (&mut n as *mut u32).cast(), (&mut half as *mut u32).cast()])?;
                    self.launch_flash("flash_next_head_fp4_select", 1, 256, &mut [(&mut sums as *mut u64).cast(), (&mut n as *mut u32).cast(), (&mut sel as *mut u64).cast()])?;
                    self.launch_flash("flash_next_head_fp4_rescore", 64, 256, &mut [(&mut t as *mut u64).cast(), (&mut n as *mut u32).cast(),
                        (&mut x as *mut u64).cast(), (&mut sel as *mut u64).cast(), (&mut cand as *mut u64).cast()])?;
                    self.launch_flash("flash_next_head_fp4_final", 1, 256, &mut [(&mut cand as *mut u64).cast(), (&mut sel as *mut u64).cast(), (&mut res as *mut u64).cast()])?;
                } else {
                    let mut tokens = m as u32;
                    let shared = (5 * if m <= 4 { 4 } else { 8 } * 1024) as u32;
                    let ctas = if m <= 4 { 192 } else { 96 };
                    self.launch_flash_rows("flash_next_head_fp4_batch", ctas, 1, 256, shared, &mut [
                        (&mut w as *mut u64).cast(), (&mut x as *mut u64).cast(),
                        (&mut sums as *mut u64).cast(), (&mut n as *mut u32).cast(), (&mut tokens as *mut u32).cast(),
                    ])?;
                    self.launch_flash("flash_next_head_fp4_select_batch", m as u32, 256, &mut [
                        (&mut sums as *mut u64).cast(), (&mut n as *mut u32).cast(), (&mut sel as *mut u64).cast(),
                    ])?;
                    self.launch_flash_rows("flash_next_head_fp4_rescore_batch", 64, m as u32, 256, 0, &mut [
                        (&mut t as *mut u64).cast(), (&mut n as *mut u32).cast(),
                        (&mut x as *mut u64).cast(), (&mut sel as *mut u64).cast(), (&mut cand as *mut u64).cast(),
                    ])?;
                    self.launch_flash("flash_next_head_fp4_final_batch", m as u32, 256, &mut [
                        (&mut cand as *mut u64).cast(), (&mut sel as *mut u64).cast(), (&mut res as *mut u64).cast(),
                    ])?;
                }
            }
        }
        Ok(())
    }
    /// Pack HC up operands without materializing a separate SiLU output.
    pub fn pack_flash_hc_input(&self, input: u64, tokens: usize) -> Result<u64, String> {
        self.pack_flash_input(input, 320, tokens, true, 32)
    }
    /// FP8 prompt BF16-norm (with optional fused HC add when `p.block != 0`, supporting both
    /// full-width `right_half == 0` and 2-half column-split `right_half != 0`).
    pub fn flash_fp8_hc_add_norm_bf16(
        &self, p: &mut crate::kernel_params::FlashHcAddNorm, mut rows: u32, mut right_half: u64,
    ) -> Result<(), String> {
        // SAFETY: Caller validates `p` operands and optional `right_half`; grid (1, ceil(rows/2), 1), block 256, shared 4096.
        unsafe {
            self.launch_flash_rows("flash_fp8_hc_add_norm_bf16", 1, rows.div_ceil(2), 256,
                4096,
                &mut [
                    (p as *mut crate::kernel_params::FlashHcAddNorm).cast(),
                    (&mut rows as *mut u32).cast(),
                    (&mut right_half as *mut u64).cast(),
                ])
        }
    }
    /// On-read FP8 prompt HC-down + HC-inject directly from FP32 residual + RMS scales + BF16
    /// norm weight, fusing exact SiLU + split2_pair into K=320 packed planes.
    pub fn flash_fp8_hc_down_fused(
        &self, p: &mut crate::kernel_params::Bf16Rows, mut scales: u64, mut norm_bf16: u64,
    ) -> Result<(), String> {
        // SAFETY: Caller validates `p` (residual, down, inject, K=320 packed planes, inject_out), scales, norm_bf16.
        unsafe {
            self.launch_flash_rows("flash_fp8_hc_down_fused", p.tokens.div_ceil(64), (p.rows + p.rows2).div_ceil(128), 256,
                crate::kernel_params::FLASH_BF16_PREFILL_ROWS_SHARED as u32,
                &mut [
                    (p as *mut crate::kernel_params::Bf16Rows).cast(),
                    (&mut scales as *mut u64).cast(),
                    (&mut norm_bf16 as *mut u64).cast(),
                ])
        }
    }
    /// Fused 4-slice FP8 prompt HC-up tiled GEMM + exact stream mix + optional next-block packed planes.
    pub fn flash_fp8_hc_up_tiled(
        &self, p: &mut crate::kernel_params::Bf16Rows,
        mut scales: u64, mut planes: u64, mut residual: u64, mut weight_bf16: u64,
    ) -> Result<(), String> {
        // SAFETY: Caller validates `p` (packed K=320 input, blocked64 up weight, output), scales, planes, residual, weight_bf16.
        unsafe {
            self.launch_flash_rows("flash_fp8_hc_up_tiled", p.tokens.div_ceil(64), 2560 / 32 / 4, 256,
                crate::kernel_params::FLASH_BF16_PREFILL_TILED_SHARED as u32,
                &mut [
                    (p as *mut crate::kernel_params::Bf16Rows).cast(),
                    (&mut scales as *mut u64).cast(),
                    (&mut planes as *mut u64).cast(),
                    (&mut residual as *mut u64).cast(),
                    (&mut weight_bf16 as *mut u64).cast(),
                ])
        }
    }
    /// 16-byte coalesced word copy for aligned RDMA mailbox receive (`words4 = bytes / 16`).
    pub fn flash_fp8_copy_v4(&self, mut src: u64, mut dst: u64, mut words4: u32) -> Result<(), String> {
        // SAFETY: Caller validates 16-byte aligned `src`/`dst` covering at least `words4 * 16` bytes.
        unsafe {
            self.launch_flash_rows("flash_fp8_copy_v4", words4.div_ceil(128), 1, 128, 0,
                &mut [(&mut src as *mut u64).cast(), (&mut dst as *mut u64).cast(), (&mut words4 as *mut u32).cast()])
        }
    }
    /// 16-byte coalesced in-place FP32 add for aligned RDMA mailbox receive (`words4 = bytes / 16`).
    pub fn flash_fp8_add_v4(&self, mut dst: u64, mut src: u64, mut words4: u32) -> Result<(), String> {
        // SAFETY: Caller validates 16-byte aligned `dst`/`src` covering at least `words4 * 16` bytes.
        unsafe {
            self.launch_flash_rows("flash_fp8_add_v4", words4.div_ceil(128), 1, 128, 0,
                &mut [(&mut dst as *mut u64).cast(), (&mut src as *mut u64).cast(), (&mut words4 as *mut u32).cast()])
        }
    }
    /// Vectorized 4-column-per-thread `flash_moe_finish_v4` for aligned `columns % 4 == 0`.
    pub fn launch_flash_moe_finish_v4(
        &self, p: &mut crate::fp8_abi::MoeFinishParams, tokens: u32,
    ) -> Result<(), String> {
        // SAFETY: Caller validates `p` with `columns % 4 == 0`, `column_start % 4 == 0`, 16-byte aligned buffers.
        unsafe {
            self.launch_flash_rows("flash_moe_finish_v4", (p.columns / 4).div_ceil(256), tokens, 256, 0,
                &mut [(p as *mut crate::fp8_abi::MoeFinishParams).cast()])
        }
    }
    /// Launch a tiled prefill BF16 GEMM (`flash_bf16_mma_prefill_tiled_*`) on packed input planes.
    pub fn launch_flash_prefill_tiled(
        &self, name: &str, p: &mut crate::kernel_params::Bf16Rows,
    ) -> Result<(), String> {
        // SAFETY: Caller validates `p` packed planes, blocked64 weights, and matching grid/shared geometry.
        unsafe {
            self.launch_flash_rows(name, p.tokens.div_ceil(64), (p.rows + p.rows2).div_ceil(128), 256,
                crate::kernel_params::FLASH_BF16_PREFILL_TILED_SHARED as u32,
                &mut [(p as *mut crate::kernel_params::Bf16Rows).cast()])
        }
    }
    /// Launch fused SoA LBF12 pair decode (`flash_bf16_mma1_10240_hc_lbf12` or `flash_bf16_mma1_2560_pair_lbf12`).
    pub fn launch_flash_lbf12_pair(
        &self, name: &str, blocks: u32, threads: u32,
        p: &mut crate::kernel_params::Bf16Rows, mut t0: u64, mut t1: u64,
    ) -> Result<(), String> {
        // SAFETY: Caller validates `p`, twin addresses `t0`/`t1`, and matching launch geometry.
        unsafe {
            if t1 == 0 {
                self.launch_flash_rows(name, blocks, 1, threads, 0,
                    &mut [(p as *mut crate::kernel_params::Bf16Rows).cast(), (&mut t0 as *mut u64).cast()])
            } else {
                self.launch_flash_rows(name, blocks, 1, threads, 0,
                    &mut [(p as *mut crate::kernel_params::Bf16Rows).cast(), (&mut t0 as *mut u64).cast(), (&mut t1 as *mut u64).cast()])
            }
        }
    }
    /// Launch `flash_bf16_lbf12_fp8_twin` to build the SoA-coalesced LBF12 twin of a packed BF16 matrix.
    pub fn launch_flash_lbf12_fp8_twin(
        &self, pack: &mut crate::kernel_params::FlashBf16Pack, slots: usize,
    ) -> Result<(), String> {
        // SAFETY: Caller validates packed BF16 `pack.source`, zeroed `pack.weight`, and `slots`.
        unsafe {
            self.launch_flash("flash_bf16_lbf12_fp8_twin", (slots * 32).div_ceil(128) as u32, 128,
                &mut [(pack as *mut crate::kernel_params::FlashBf16Pack).cast()])
        }
    }
    /// Launch the 4-stage NVFP4 screen + MXFP8 rescore greedy draft head.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_flash_draft_head_fp4(
        &self, mut fp4: u64, mut mx8: u64, mut hidden: u64,
        mut logits: u64, partials: u64, mut result: u64, mut rows: u32,
    ) -> Result<(), String> {
        let (mut sums, mut sel, mut half) = (logits, partials, 0b01_0101_0101u32);
        let mut cand = logits;
        // SAFETY: Caller validates fp4/mx8 twins, hidden[2560], logits[>=rows.max(4096)], partials[>=16], result[>=3].
        unsafe {
            self.launch_flash("flash_next_head_fp4", 192, 256, &mut [(&mut fp4 as *mut u64).cast(), (&mut hidden as *mut u64).cast(),
                (&mut logits as *mut u64).cast(), (&mut rows as *mut u32).cast(), (&mut half as *mut u32).cast()])?;
            self.launch_flash("flash_next_head_fp4_select", 1, 32, &mut [(&mut sums as *mut u64).cast(), (&mut rows as *mut u32).cast(), (&mut sel as *mut u64).cast()])?;
            self.launch_flash("flash_next_head_fp4_rescore", 64, 256, &mut [(&mut mx8 as *mut u64).cast(), (&mut rows as *mut u32).cast(),
                (&mut hidden as *mut u64).cast(), (&mut sel as *mut u64).cast(), (&mut cand as *mut u64).cast()])?;
            self.launch_flash("flash_next_head_fp4_final", 1, 256, &mut [(&mut cand as *mut u64).cast(), (&mut sel as *mut u64).cast(), (&mut result as *mut u64).cast()])
        }
    }
    /// Expand one gathered BF16 vocabulary row into one FP32 hidden row (`width % 128 == 0`).
    pub fn flash_embedding_expand_row(
        &self, raw: &DeviceVector<'a>, raw_byte_offset: usize,
        out: &mut DeviceVector<'a>, out_f32_offset: usize, width: usize,
    ) -> Result<(), String> {
        if !width.is_multiple_of(128)
            || raw_byte_offset + width * 2 > raw.len() * 4
            || out_f32_offset + width > out.len()
        {
            return Err("flash_embedding_expand_row out of bounds".into());
        }
        let (mut raw_addr, mut out_addr, mut w) = (
            raw.device_address() + raw_byte_offset as u64,
            out.device_address() + (out_f32_offset * 4) as u64,
            width as u32,
        );
        // SAFETY: Bounds above cover raw[raw_byte_offset..+2*width] and out[out_f32_offset..+width],
        // and width % 128 == 0 ensures grid * 128 == width so only 0..width are written.
        unsafe {
            self.launch_flash("flash_embedding_expand", (width / 128) as u32, 128,
                &mut [(&mut raw_addr as *mut u64).cast(), (&mut out_addr as *mut u64).cast(), (&mut w as *mut u32).cast()])
        }
    }
    fn pack_flash_input(&self, input: u64, cols: usize, tokens: usize, hc_silu: bool, lanes: usize) -> Result<u64, String> {
        if let Some((x, c, t, planes)) = self.dense_offer.take() {
            // Offered planes hold the pack's words of rows below M; decode writers add the 0 words of
            // padding rows t + 8. Other rows past M stay unwritten: each MMA output row reads only its
            // own A row, and rows past M are not stored. A latency pack's lanes subset is the same words.
            if (x, c, t) == (input, cols, tokens) && !hc_silu { return Ok(planes); }
        }
        let (output, words) = self.flash_dense_words(cols, tokens, 8)?;
        let mut p = crate::kernel_params::FlashBf16Input { input, output,
            tokens: tokens as u32, cols: cols as u32, hc_silu: u32::from(hc_silu), lanes: lanes as u32 };
        // One thread per written fragment (words / 4 at all 32 lanes).
        unsafe { self.launch_flash(self.chain_entry(tokens, "flash_bf16_input", "flash_bf16_input_chain"), (words / 128 * lanes).div_ceil(128) as u32, 128,
            &mut [(&mut p as *mut crate::kernel_params::FlashBf16Input).cast()])?; }
        Ok(p.output)
    }
}
