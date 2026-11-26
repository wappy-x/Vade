//! BF16 bit conversion shared by Rust GPU kernels.

#[inline(always)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
