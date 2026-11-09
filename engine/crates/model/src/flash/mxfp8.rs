//! CPU oracle: E4M3FN weights times one UE8M0 power-of-two scale per 32 values.
use crate::{Result, invalid};
pub const BLOCK: usize = 32;
pub fn dequant_value(weight: u8, scale: u8) -> f32 {
    if scale == 255 { return f32::NAN; }
    // FP64 holds the exact product even at the exponent extremes. Round once
    // to FP32, preserving subnormals, signed zero and overflow semantics.
    (super::nvfp4::e4m3(weight) as f64 * 2f64.powi(scale as i32 - 127)) as f32
}
pub fn dequantize(weights: &[u8], scales: &[u8], output: &mut [f32]) -> Result<()> {
    if weights.len() != output.len() || !weights.len().is_multiple_of(BLOCK) || scales.len() != weights.len() / BLOCK {
        return Err(invalid("MXFP8 dequantization lengths disagree"));
    }
    for (i, out) in output.iter_mut().enumerate() { *out = dequant_value(weights[i], scales[i / BLOCK]); }
    Ok(())
}
/// Exact load-time expansion for the BF16 projection path: the FP32 oracle
/// value rounded once to BF16 (nearest-even). Non-finite inputs fail closed.
pub fn expand_to_bf16(weights: &[u8], scales: &[u8], output: &mut [u16]) -> Result<()> {
    if weights.len() != output.len() || !weights.len().is_multiple_of(BLOCK) || scales.len() != weights.len() / BLOCK {
        return Err(invalid("MXFP8 dequantization lengths disagree")); }
    for (i, out) in output.iter_mut().enumerate() {
        let value = dequant_value(weights[i], scales[i / BLOCK]);
        if !value.is_finite() { return Err(invalid("MXFP8 dequantization is not finite")); }
        *out = super::nvfp4::f32_to_bf16(value);
    }
    Ok(())
}
