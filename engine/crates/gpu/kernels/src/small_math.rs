//! Arithmetic helpers of the small-count kernels, as private always-inline copies. The shared versions (`expert_ops::round`, `latent_ops::sigmoid`) are
//! also called from the existing kernels; a new caller changes how LLVM inlines them there, which changed the SASS of four existing functions (a device
//! cubin comparison caught it). The expressions are the shared ones, token for token.
/// `expert_ops::round`: the BF16 round of a float, kept in FP32.
#[inline(always)]
pub(crate) fn round(x:f32)->f32 {f32::from_bits(crate::next_math::bf16(x)<<16)}
/// `latent_ops::sigmoid`.
#[inline(always)]
pub(crate) fn sigmoid(x:f32)->f32 {1.0/(1.0+crate::next_math::exp(-x))}
