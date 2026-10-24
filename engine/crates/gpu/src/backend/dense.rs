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
    /// Pack HC up operands without materializing a separate SiLU output.
    pub fn pack_flash_hc_input(&self, input: u64, tokens: usize) -> Result<u64, String> {
        self.pack_flash_input(input, 320, tokens, true, 32)
    }
    fn pack_flash_input(&self, input: u64, cols: usize, tokens: usize, hc_silu: bool, lanes: usize) -> Result<u64, String> {
        if let Some((x, c, t, planes)) = self.dense_offer.take() {
            // Offered planes hold every lane's fragments; a latency pack's lanes subset is the same words.
            if (x, c, t) == (input, cols, tokens) && !hc_silu { return Ok(planes); }
        }
        let (output, words) = self.flash_dense_words(cols, tokens, 8)?;
        let mut p = crate::kernel_params::FlashBf16Input { input, output,
            tokens: tokens as u32, cols: cols as u32, hc_silu: u32::from(hc_silu), lanes: lanes as u32 };
        // One thread per written fragment (words / 4 at all 32 lanes).
        unsafe { self.launch_flash("flash_bf16_input", (words / 128 * lanes).div_ceil(128) as u32, 128,
            &mut [(&mut p as *mut crate::kernel_params::FlashBf16Input).cast()])?; }
        Ok(p.output)
    }
}
