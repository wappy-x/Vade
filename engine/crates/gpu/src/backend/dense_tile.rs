//! The dense input tile of flash_bf16_mma_prefill_tiled: its checked size, and
//! the entry for fused producers that write flash_bf16_input's words themselves.
use super::*;

impl<'a> Backend<'a> {
    /// The reserved tile's address and words per plane for `tokens` x `cols`
    /// (`cols` a multiple of `align`), checked against the reservation.
    pub(super) fn flash_dense_words(&self, cols: usize, tokens: usize, align: usize) -> Result<(u64, usize), String> {
        self.prepare_flash_dense()?;
        let scratch = self.flash_input.borrow();
        let scratch = scratch.as_ref().ok_or("Flash dense input was not reserved before capture")?;
        let words = tokens.div_ceil(16).checked_mul(cols.div_ceil(16)).and_then(|n| n.checked_mul(128)).ok_or("Flash dense tile overflow")?;
        let total = words.checked_mul(crate::kernel_params::TERMS).ok_or("Flash dense tile overflow")?;
        if tokens == 0 || cols == 0 || cols % align != 0 || total > scratch.len() { return Err("Flash dense input exceeds its reserved tile".into()); }
        Ok((scratch.device_address(), words))
    }
    /// The dense input tile for a fused producer that writes flash_bf16_input's
    /// words itself (`tokens` x `cols`): checks the size and clears the last
    /// 16-row tile row of every plane when `tokens` is not a multiple of 16 (the
    /// pack writes zeros for the padding rows there). Returns the tile address.
    pub fn flash_dense_tile(&self, cols: usize, tokens: usize) -> Result<u64, String> {
        let (tile, words) = self.flash_dense_words(cols, tokens, 16)?;
        if !tokens.is_multiple_of(16) {
            let row = tokens / 16 * (cols / 16) * 128;
            self.context.activate()?;
            for plane in 0..crate::kernel_params::TERMS {
                unsafe { check(cuMemsetD8Async(tile + ((plane * words + row) * 4) as u64, 0, (cols / 16) * 128 * 4,
                    self.stream.raw), "dense tile padding clear")?; }
            }
            self.context.pending.set(true);
        }
        Ok(tile)
    }
}
