//! Four small host functions that several operations share: the weighted
//! norm, the plain norm, the length clip and the soft cap. Fixed order, FP32.
//!
//! Source (read for the math, no code copied): the public `transformers`
//! release v5.24.0, `models/inkling2_loop/modeling_inkling2_loop.py`.
//! Plain norm: lines 67-76. Length clip: lines 176-187. Weighted norm:
//! lines 730-745. Soft cap: lines 751-757.
//!
//! Number contract: every function takes FP32 values and returns FP32
//! values. The caller widens BF16 inputs first and rounds the result with
//! [`round_bf16`] where its step asks for BF16. Nothing here reads a config
//! or a model name. Sizes come from the slices.
//!
//! Fixed orders of this reference, so that a run always gives the same bits.
//! They are the orders of the device kernels that run the same steps:
//! - [`norm_sum_squares`], for [`weighted_norm`] and [`plain_norm`]: the tree
//!   of the resident norm kernel. Chunks of 128 values, 32 lanes a chunk, and
//!   the chunk sums joined the same way.
//! - [`tree_sum_squares`], for [`length_clip`] and [`tree_plain_norm`]: one
//!   32-lane tree over the whole slice. The device runs the plain norm of a
//!   mix, which stays FP32, on this tree.
//! - A product and a sum are two rounded steps. No fused multiply-add.
//! - The inverse root is `1.0 / sqrt(v)`: two correctly rounded operations.
//! - A division of values by one number is a product with [`reciprocal`]. The
//!   soft cap uses it. One number divided by a value is the number times the
//!   rounded reciprocal of the value ([`clip_ratio`]). A value divided by a
//!   value is a true division.
//! - [`exp`], [`sigmoid`], [`tanh`] and [`round_bf16`] come from the shared
//!   math file `fixed_math`: one source for the host and the device.
//!
//! These orders are this reference's contract. The source leaves them to its
//! library, so last-bit parity with the source is not claimed.
use crate::{Result, invalid};

/// The length clip adds this to the length before the divide. Source line 187.
pub const CLIP_EPSILON: f32 = 1e-6;
/// Lanes of a fixed sum tree. A constant of the sum, not of a model.
pub const SUM_LANES: usize = 32;
/// Values in one chunk of the norm tree. A constant of the sum, not of a model.
pub const NORM_CHUNK: usize = 128;

/// Every math call of the host references goes through the one shared math
/// file that the device code also builds. `round_bf16` is its BF16 round.
pub use crate::fixed_math::{exp, rn as round_bf16, sigmoid, tanh};

/// The factor that replaces a division of FP32 values by one number: the
/// FP64 value `1 / divisor`, rounded one time to FP32. The source divides a
/// tensor by a number in this way on its GPU. A run of the source measured it
/// (receipt `divform.json` of the reference run): 4,000,000 of 4,000,000
/// values equal this form for each of 4 divisors. True division equals fewer.
/// A division of a tensor by a tensor is a true division and does not use this.
pub fn reciprocal(divisor: f64) -> f32 { (1.0 / divisor) as f32 }

/// A soft cap: `tanh(x / cap) * cap`, with the division by the cap as a
/// product with [`reciprocal`]. Source lines 751-757.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftCap { cap: f32, reciprocal: f32 }
impl SoftCap {
    /// The cap must be positive and finite in FP32. `name` goes into the refusal text.
    pub fn new(cap: f64, name: &str) -> Result<Self> {
        let (value, reciprocal) = (check_cap(cap, name)?, reciprocal(cap));
        if !reciprocal.is_finite() || reciprocal <= 0.0 { return Err(invalid(format!("{name} has no finite FP32 reciprocal"))); }
        Ok(Self { cap: value, reciprocal })
    }
    pub fn cap(self) -> f32 { self.cap }
    /// FP32 in, FP32 out. Not rounded.
    pub fn apply(self, x: f32) -> f32 { tanh(x * self.reciprocal) * self.cap }
}

/// A cap or a clip threshold must be positive and finite in FP32.
pub fn check_cap(cap: f64, name: &str) -> Result<f32> {
    let value = cap as f32;
    if !cap.is_finite() || !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!("{name} must be positive and finite in FP32")));
    }
    Ok(value)
}

/// Join 32 lane sums: lanes 16 apart, then 8, 4, 2, 1. The result is lane 0.
fn join(mut lanes: [f32; SUM_LANES]) -> f32 {
    let mut distance = SUM_LANES / 2;
    while distance > 0 {
        for lane in 0..distance { lanes[lane] += lanes[lane + distance]; }
        distance /= 2;
    }
    lanes[0]
}

/// Sum of squares on one 32-lane tree. Lane `l` adds the squares of
/// `x[l], x[l + 32], ...` in rising index from 0.0. Then the lanes are joined.
pub fn tree_sum_squares(x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; SUM_LANES];
    for (i, &v) in x.iter().enumerate() { lanes[i % SUM_LANES] += v * v; }
    join(lanes)
}

/// Sum of squares on the norm tree. Each chunk of 128 values gives one sum on
/// the 32-lane tree. Lane `l` then adds the chunk sums `l, l + 32, ...` in
/// rising index from 0.0, and the lanes are joined.
pub fn norm_sum_squares(x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; SUM_LANES];
    for (chunk, values) in x.chunks(NORM_CHUNK).enumerate() { lanes[chunk % SUM_LANES] += tree_sum_squares(values); }
    join(lanes)
}

fn inverse_rms(x: &[f32], epsilon: f32, sum_squares: fn(&[f32]) -> f32) -> Result<f32> {
    if x.is_empty() || !epsilon.is_finite() || epsilon < 0.0 { return Err(invalid("norm needs values and a finite epsilon that is not negative")); }
    let scale = 1.0 / (sum_squares(x) / x.len() as f32 + epsilon).sqrt();
    if !scale.is_finite() { return Err(invalid("norm input is not finite or its mean square is zero with no epsilon")); }
    Ok(scale)
}

/// Weighted norm: `weight[i] * (x[i] * 1/sqrt(mean(x^2) + epsilon))`.
/// The mean is over the whole slice. The scale multiplies first, then the weight.
pub fn weighted_norm(x: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    if weight.len() != x.len() { return Err(invalid("weighted norm needs one weight for each value")); }
    let scale = inverse_rms(x, epsilon, norm_sum_squares)?;
    finite(x.iter().zip(weight).map(|(&v, &w)| (v * scale) * w).collect(), "weighted norm")
}

/// Plain norm: `x[i] * 1/sqrt(mean(x^2) + epsilon)`. No learned weight.
/// The squares are summed on the norm tree, as in the weighted norm. Use it
/// where the device runs the norm kernel with no weight.
pub fn plain_norm(x: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    let scale = inverse_rms(x, epsilon, norm_sum_squares)?;
    finite(x.iter().map(|&v| v * scale).collect(), "plain norm")
}

/// The same plain norm with the squares summed on one 32-lane tree. Use it
/// for the plain norm of a mix, whose result stays FP32. For a slice of 128
/// values or fewer the two plain norms give the same bits.
pub fn tree_plain_norm(x: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    let scale = inverse_rms(x, epsilon, tree_sum_squares)?;
    finite(x.iter().map(|&v| v * scale).collect(), "plain norm")
}

/// `threshold / (length + 1e-6)` in the form that the source's library gives
/// to a number divided by a tensor: two rounded FP32 steps. First
/// `1 / (length + 1e-6)`, then the FP32 threshold times it. A run of the
/// source measured it (receipt `divform2.json` of the reference run):
/// 4,000,000 of 4,000,000 values equal this form, on its GPU and on its CPU.
/// One true division equals about 72 in 100. This function is the one place
/// that holds the form.
pub fn clip_ratio(threshold: f32, length: f32) -> f32 { threshold * (1.0 / (length + CLIP_EPSILON)) }

/// Length clip: `x[i] * min(1, threshold / (length(x) + 1e-6))`.
/// `length` is the square root of the sum of squares of the whole slice.
/// A vector that is shorter than the threshold does not change.
pub fn length_clip(x: &[f32], threshold: f32) -> Result<Vec<f32>> {
    if x.is_empty() || !threshold.is_finite() || threshold <= 0.0 { return Err(invalid("length clip needs values and a positive finite threshold")); }
    let ratio = clip_ratio(threshold, tree_sum_squares(x).sqrt());
    if ratio.is_nan() { return Err(invalid("length clip input is not finite")); }
    let factor = if ratio > 1.0 { 1.0 } else { ratio };
    finite(x.iter().map(|&v| v * factor).collect(), "length clip")
}

fn finite(values: Vec<f32>, what: &str) -> Result<Vec<f32>> {
    if values.iter().any(|v| !v.is_finite()) { return Err(invalid(format!("{what} result is not finite"))); }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ulps(a: f32, exact: f64) -> f64 {
        let step = (f32::from_bits(a.abs().to_bits() + 1) - a.abs()) as f64;
        ((a as f64) - exact).abs() / step
    }
    #[test]
    fn weighted_norm_hand_values() {
        // x = [3, 4]: mean square 12.5. With no epsilon the scale is 1/sqrt(12.5).
        // 3 * scale = 0.8485281, 4 * scale = 1.1313709. Weights [2, 0.5].
        let y = weighted_norm(&[3.0, 4.0], &[2.0, 0.5], 0.0).unwrap();
        let scale = 1.0 / 12.5f32.sqrt();
        assert_eq!(y, [(3.0 * scale) * 2.0, (4.0 * scale) * 0.5]);
        assert!((y[0] - 1.697_056_3).abs() < 1e-6 && (y[1] - 0.565_685_4).abs() < 1e-6);
        // x = [2, 2, 2, 2]: mean square 4, scale exactly 1/2. Exact results.
        assert_eq!(weighted_norm(&[2.0; 4], &[1.0, -1.0, 4.0, 0.25], 0.0).unwrap(), [1.0, -1.0, 4.0, 0.25]);
        // The epsilon is inside the root: [1, 1] with epsilon 3 gives 1/sqrt(4).
        assert_eq!(weighted_norm(&[1.0, 1.0], &[1.0, 6.0], 3.0).unwrap(), [0.5, 3.0]);
    }
    #[test]
    fn plain_norm_is_the_weighted_norm_with_weight_one() {
        let x = [0.25f32, -1.5, 3.0, 0.0, 7.0];
        assert_eq!(plain_norm(&x, 1e-6).unwrap(), weighted_norm(&x, &[1.0; 5], 1e-6).unwrap());
        assert_eq!(plain_norm(&[2.0; 4], 0.0).unwrap(), [1.0; 4]);
        // A zero vector stays zero when the epsilon is positive.
        assert_eq!(plain_norm(&[0.0; 3], 1e-6).unwrap(), [0.0; 3]);
        // Up to 128 values the two sum trees are one tree, so the two plain norms agree bit for bit.
        let short: Vec<f32> = (0..128).map(|i| (i as f32 - 60.5) * 0.37).collect();
        assert_eq!(plain_norm(&short, 1e-6).unwrap(), tree_plain_norm(&short, 1e-6).unwrap());
        // Above 128 values the trees differ. Sums here: 2^24 + 2 on the norm tree, 2^24 on the one tree.
        let mut long = [0.0f32; 130];
        (long[0], long[128], long[129]) = (4096.0, 1.0, 1.0);
        let (a, b) = (plain_norm(&long, 0.0).unwrap(), tree_plain_norm(&long, 0.0).unwrap());
        assert_eq!(a[0], 4096.0 * (1.0 / (16_777_218.0f32 / 130.0).sqrt()));
        assert_eq!(b[0], 4096.0 * (1.0 / (16_777_216.0f32 / 130.0).sqrt()));
        assert_ne!(a[0], b[0]);
    }
    #[test]
    fn length_clip_hand_values() {
        // Length of [3, 4] is 5. Threshold 2.5: factor 2.5 / (5 + 1e-6), just under 1/2.
        // The factor is two rounded steps: the reciprocal first, then the product.
        let y = length_clip(&[3.0, 4.0], 2.5).unwrap();
        let factor = 2.5f32 * (1.0f32 / (5.0f32 + 1e-6));
        assert_eq!(clip_ratio(2.5, 5.0), factor);
        assert_eq!(y, [3.0 * factor, 4.0 * factor]);
        // One division gives another last bit at this point. A threshold that is a power of 2 gives the same bits.
        assert_ne!(factor, 2.5f32 / (5.0f32 + 1e-6));
        assert_eq!(clip_ratio(8.0, 5.0), 8.0f32 / (5.0f32 + 1e-6));
        assert!(factor < 0.5 && (y[0] - 1.5).abs() < 1e-6 && (y[1] - 2.0).abs() < 1e-6);
        // A shorter vector is unchanged, bit for bit.
        assert_eq!(length_clip(&[3.0, 4.0], 6.0).unwrap(), [3.0, 4.0]);
        // At the threshold the epsilon makes the factor less than 1.
        let edge = length_clip(&[3.0, 4.0], 5.0).unwrap();
        assert!(edge[0] <= 3.0 && edge[1] <= 4.0);
        assert_eq!(length_clip(&[0.0; 4], 1.0).unwrap(), [0.0; 4]);
    }
    #[test]
    fn softcap_hand_values() {
        let cap = |c: f64| SoftCap::new(c, "cap").unwrap();
        // tanh(ln 3) = (3 - 1/3) / (3 + 1/3) = 0.8. So the cap 4 of 4 ln 3 is 3.2.
        assert!((cap(4.0).apply(4.0 * 3.0f32.ln()) - 3.2).abs() < 2e-6);
        assert!((cap(25.0).apply(-25.0 * 3.0f32.ln()) + 20.0).abs() < 1e-5);
        // Small values pass nearly unchanged. Large values stop at the cap.
        assert!((cap(30.0).apply(0.001) - 0.001).abs() < 1e-9);
        assert_eq!((cap(30.0).apply(1000.0), cap(30.0).apply(-1000.0)), (30.0, -30.0));
        assert_eq!(cap(4.0).apply(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(cap(30.0).cap(), 30.0);
    }
    #[test]
    fn division_by_a_number_is_a_product_with_the_rounded_reciprocal() {
        // 1/4 is an FP32 value, so the product and the division agree. 1/30 is not.
        assert_eq!(reciprocal(4.0), 0.25);
        assert_eq!(reciprocal(30.0), (1.0f64 / 30.0) as f32);
        // 30 * fl(1/30) = 1 in FP32, and 3 * fl(1/30) is not 3 / 30: the two forms differ in the last bit.
        let r = reciprocal(30.0);
        assert_eq!(30.0f32 * r, 1.0);
        assert_ne!(3.0f32 * r, 3.0f32 / 30.0);
        // The cap uses the product: tanh(x * r) * cap.
        let cap = SoftCap::new(30.0, "cap").unwrap();
        assert_eq!(cap.apply(3.0), tanh(3.0 * r) * 30.0);
        // The reciprocal comes from the FP64 number, not from its FP32 rounding.
        assert_eq!(reciprocal(7.0f64.sqrt()), (1.0 / 7.0f64.sqrt()) as f32);
        assert!(SoftCap::new(0.0, "cap").is_err() && SoftCap::new(f64::NAN, "cap").is_err() && SoftCap::new(1e60, "cap").is_err() && SoftCap::new(1e-60, "cap").is_err());
    }
    #[test]
    fn tanh_comes_from_the_shared_math_file() {
        // The full sweep of tanh is a test of the shared file. Here: the points the soft cap relies on.
        assert_eq!(tanh(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(tanh(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!((tanh(20.0), tanh(-20.0)), (1.0, -1.0));
        for x in [1e-30f32, 1e-6, 0.01, 0.2, 0.25, 0.3, 1.0, 3.0, 9.0] {
            let y = tanh(x);
            assert!(ulps(y, (x as f64).tanh()) <= 4.0, "tanh({x})");
            assert_eq!(tanh(-x).to_bits(), (-y).to_bits());
        }
    }
    #[test]
    fn sigmoid_uses_the_shared_exponential() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!((sigmoid(3.0f32.ln()) - 0.75).abs() < 1e-6);
        assert_eq!(sigmoid(100.0), 1.0);
        assert!(sigmoid(-100.0) > 0.0 && sigmoid(-100.0) < 1e-37);
        assert_eq!(exp(0.0), 1.0);
    }
    #[test]
    fn round_bf16_is_nearest_even() {
        // BF16 keeps 8 significant bits. Near 1 the step is 2^-7.
        assert_eq!(round_bf16(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(round_bf16(1.0 + 3.0 / 256.0), 1.0 + 2.0 / 128.0);
        assert_eq!(round_bf16(1.0 + 1.0 / 128.0), 1.0 + 1.0 / 128.0);
        assert_eq!(round_bf16(-0.3), -0.300_781_25);
    }
    #[test]
    fn sum_orders_are_the_two_fixed_trees() {
        // 2^24 + 1 is not an FP32 value, and 2^24 + 2 is. 4096^2 = 2^24.
        // One tree, 4 values [4096, 1, 0, 1]: lanes 1 and 3 join first (1 + 1 = 2), then lane 0: 2^24 + 2.
        // A rising chain would give 2^24 + 1 + 0 + 1 = 2^24.
        assert_eq!(tree_sum_squares(&[4096.0, 1.0, 0.0, 1.0]), 16_777_218.0);
        // Values 32 apart share a lane. Lane 0: 2^24 + 1 = 2^24. Lane 1: 1 + 1 = 2. Joined: 2^24 + 2.
        let mut x = [0.0f32; 34];
        (x[0], x[1], x[32], x[33]) = (4096.0, 1.0, 1.0, 1.0);
        assert_eq!(tree_sum_squares(&x), 16_777_218.0);
        // The norm tree first sums each chunk of 128. Chunk 0: 2^24. Chunk 1 holds two ones: 2. Joined: 2^24 + 2.
        // One tree over all 130 values puts values 0 and 128 in lane 0 (2^24 + 1 = 2^24) and value 129 in lane 1: 2^24.
        let mut y = [0.0f32; 130];
        (y[0], y[128], y[129]) = (4096.0, 1.0, 1.0);
        assert_eq!(norm_sum_squares(&y), 16_777_218.0);
        assert_eq!(tree_sum_squares(&y), 16_777_216.0);
        // Chunk sums 32 apart share a lane: chunks 0 and 32 here. 2^24 + 1 = 2^24, then chunk 1 adds 1 more: 2^24.
        let mut z = vec![0.0f32; 33 * 128];
        (z[0], z[128], z[32 * 128]) = (4096.0, 1.0, 1.0);
        assert_eq!(norm_sum_squares(&z), 16_777_216.0);
        z[129] = 1.0;
        assert_eq!(norm_sum_squares(&z), 16_777_218.0);
        // Short slices: both trees are the plain sum when no rounding happens.
        assert_eq!((tree_sum_squares(&[3.0, 4.0]), norm_sum_squares(&[3.0, 4.0]), norm_sum_squares(&[])), (25.0, 25.0, 0.0));
    }
    #[test]
    fn bad_inputs_are_refused() {
        assert!(weighted_norm(&[1.0], &[1.0, 2.0], 1e-6).is_err());
        assert!(weighted_norm(&[], &[], 1e-6).is_err());
        assert!(plain_norm(&[0.0; 2], 0.0).is_err());
        assert!(plain_norm(&[f32::NAN, 1.0], 1e-6).is_err());
        assert!(plain_norm(&[1.0], -1.0).is_err());
        assert!(length_clip(&[1.0], 0.0).is_err());
        assert!(length_clip(&[f32::INFINITY], 1.0).is_err());
        assert!(length_clip(&[f32::NAN], 1.0).is_err());
        assert!(check_cap(0.0, "cap").is_err() && check_cap(f64::NAN, "cap").is_err() && check_cap(1e60, "cap").is_err());
        assert_eq!(check_cap(30.0, "cap").unwrap(), 30.0);
    }
}
