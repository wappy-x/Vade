//! Host reference for a configurable SwiGLU epilogue. No model-name dispatch.
use crate::{Result, invalid, flash::nvfp4::{bf16_to_f32, f32_to_bf16, exp_ref}};

/// A missing bound preserves the existing uncapped arithmetic.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SwiGlu { limit: Option<f32> }
impl SwiGlu {
    /// The bound must remain positive and finite at both supported precisions.
    pub fn new(limit: Option<f64>) -> Result<Self> {
        let limit = limit.map(|v| {
            let f = v as f32;
            let b = bf16_to_f32(f32_to_bf16(f));
            if !v.is_finite() || f <= 0.0 || !f.is_finite() || b <= 0.0 || !b.is_finite() {
                return Err(invalid("swiglu_limit must be positive and finite in FP32 and BF16"));
            }
            Ok(f)
        }).transpose()?;
        Ok(Self { limit })
    }
    pub fn limit(self) -> Option<f32> { self.limit }
    /// Projection outputs enter in FP32. BF16 mode rounds each projection,
    /// the bound, SiLU, and the final product. Clamp precedes SiLU and product.
    /// FP32 mode has no BF16 boundaries. The gate has no lower bound.
    pub fn reference(self, gate: f32, up: f32, bf16: bool) -> f32 {
        self.reference_with_exp(gate, up, bf16, exp_ref)
    }
    /// An uncapped device uses its existing exponential. A host check can
    /// supply the same exponential without changing the rounding contract.
    pub fn reference_with_exp(self, gate: f32, up: f32, bf16: bool, exp: impl Fn(f32) -> f32) -> f32 {
        let round = |x| if bf16 { bf16_to_f32(f32_to_bf16(x)) } else { x };
        let (mut gate, mut up) = (round(gate), round(up));
        if let Some(limit) = self.limit {
            let limit = round(limit);
            if gate > limit { gate = limit; }
            if up > limit { up = limit; }
            if up < -limit { up = -limit; }
        }
        let activated = round(gate / (1.0 + exp(-gate)));
        round(activated * up)
    }
}
