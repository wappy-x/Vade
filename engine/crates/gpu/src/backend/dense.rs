//! Dense prompt tiles, mixed planes and one-shot offers. Combined mechanical move.
use super::*;

impl<'a> Backend<'a> {
    /// One stable activation tile, shared by projections on the ordered stream.
    /// Reserve at session creation so batch graphs never capture an allocation.
    pub fn prepare_flash_dense(&self) -> Result<(), String> {
        let mut input = self.flash_input.borrow_mut();
        if input.is_none() { *input = Some(self.vector(crate::flash_dense_reserve_bytes() / 4)?); }
        Ok(())
    }
    pub fn flash_dense_bytes(&self) -> usize {
        self.flash_input.borrow().as_ref().map_or(0, |v| v.len() * 4)
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
    /// first tile still holds its own operand (prompt HC up -> its mixed output's planes).
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
    /// Pack HC up operands without materializing a separate SiLU output.
    pub fn pack_flash_hc_input(&self, input: u64, tokens: usize) -> Result<u64, String> {
        self.pack_flash_input(input, 320, tokens, true, 32)
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
