//! Fixed FP32 math for host forms and GPU arms: `exp`, `sigmoid`, `tanh`, `ln` and the BF16 round.
//! One source, three builds: this crate, the kernel crate and the model crate include this file by path.
//! The file has safe code only and calls no platform math function. Each step is one correctly rounded
//! IEEE 754 operation (add, subtract, multiply, divide or fused multiply-add) in a fixed order. Thus each
//! function gives the same bits on every CPU and on the GPU.
//!
//! `exp` and `rn` are copies of `next_math::exp` and of the round in `causal_ops::round`. The model crate
//! forbids unsafe code and `next_math.rs` holds an unsafe function, so the model crate cannot include that
//! file. `fixed_math_tests.rs` compares each copy with its origin, bit for bit, on every 32-bit pattern.

#[inline(always)]
fn madd(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")] { core::f32::math::mul_add(a, b, c) }
    #[cfg(not(target_arch = "nvptx64"))] { a.mul_add(b, c) }
}
/// The high and the low part of ln 2. The high part has 16 significant bits, so its product with a small
/// integer is exact.
#[allow(clippy::excessive_precision)] // The literal stays as `next_math::exp` writes it.
const LN2_HIGH: f32 = 0.693145751953125;
const LN2_LOW: f32 = 0.0000014286068;

/// Round to the nearest BF16 value, ties to even, kept in an FP32 word. Same bits as `causal_ops::round`.
#[inline(always)]
pub fn rn(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
}
/// The engine's exponential. Same bits as `next_math::exp` for every input.
#[allow(clippy::manual_clamp)] // `clamp` treats a NaN in another way; the lines stay as in `next_math::exp`.
#[inline(always)]
pub fn exp(t: f32) -> f32 {
    let t = if t > 87.0 { 87.0 } else if t < -87.0 { -87.0 } else { t };
    let n = t * core::f32::consts::LOG2_E + 12_582_912.0 - 12_582_912.0;
    let r = madd(-n, LN2_HIGH, t);
    let r = madd(-n, LN2_LOW, r);
    // The 7 steps of `next_math::exp`, one by one: its loop over a list of the constants is the same steps in the
    // same order. Written out, the GPU build keeps the constants in the code and stores no list in thread memory.
    let p = madd(1.0 / 5040.0f32, r, 1.0 / 720.0);
    let p = madd(p, r, 1.0 / 120.0);
    let p = madd(p, r, 1.0 / 24.0);
    let p = madd(p, r, 1.0 / 6.0);
    let p = madd(p, r, 0.5);
    let p = madd(p, r, 1.0);
    let p = madd(p, r, 1.0);
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}
/// Same expression as `causal_ops::sigmoid`.
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + exp(-x)) }

/// Below this magnitude `tanh` uses its odd series. At and above it `tanh` uses `exp`.
pub const TANH_SERIES_BELOW: f32 = 0.5625;
/// Coefficients of `tanh(x) / x - 1` as a series in `x * x`, highest power first: the Taylor terms
/// -1/3, 2/15, -17/315, 62/2835, -1382/155925, 21844/6081075, -929569/638512875, 6404582/10854718875,
/// -443861162/1856156927625.
const TANH_SERIES: [f32; 9] = [-443861162.0 / 1856156927625.0, 6404582.0 / 10854718875.0, -929569.0 / 638512875.0,
    21844.0 / 6081075.0, -1382.0 / 155925.0, 62.0 / 2835.0, -17.0 / 315.0, 2.0 / 15.0, -1.0 / 3.0];
/// Hyperbolic tangent. Odd: `tanh(-x)` has the bits of `-tanh(x)`. A NaN gives a NaN.
/// Small inputs: `x + x * z * P(z)` with `z = x * x`, so no bit is lost near zero and a tiny `x` gives `x`.
/// Other inputs: with `t = exp(-2 * |x|)`, the quotient `(1 - t) / (1 + t)` and one correction step that
/// takes away the rounding errors of the two sums and of the division. The result is 1 from about 9.02.
/// Measured on every finite input: the largest error is 0.8830 units of the last place, and the function never
/// falls (`fixed_math_tests.rs`, the ignored test `tanh_error_on_every_input`).
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    let a = f32::from_bits(x.to_bits() & 0x7fff_ffff);
    let sign = x.to_bits() & 0x8000_0000;
    let magnitude = if a < TANH_SERIES_BELOW {
        let z = a * a;
        let mut p = TANH_SERIES[0];
        for c in [TANH_SERIES[1], TANH_SERIES[2], TANH_SERIES[3], TANH_SERIES[4], TANH_SERIES[5], TANH_SERIES[6], TANH_SERIES[7], TANH_SERIES[8]] { p = madd(p, z, c); }
        madd(a * z, p, a)
    } else {
        let t = exp(-2.0 * a);
        // Each low part is the exact rounding error of its sum: 1 is the larger operand of both.
        let n = 1.0 - t; let n_low = (1.0 - n) - t;
        let d = 1.0 + t; let d_low = t - (d - 1.0);
        let q = n / d;
        // The exact remainder of the division, then the two low parts.
        let remainder = madd(-q, d, n);
        q + (remainder + (n_low - q * d_low)) / d
    };
    f32::from_bits(magnitude.to_bits() | sign)
}

/// Natural logarithm of a positive number. Zero gives negative infinity, a negative number or a NaN gives
/// a NaN, and positive infinity gives itself.
/// With `x = m * 2^e` and `m` in [sqrt(1/2), sqrt(2)): `f = m - 1` is exact, `s = f / (2 + f)`, and
/// `ln(m) = f - (f*f/2 - s * (f*f/2 + R))`, where `R` is the series 2/3 z + 2/5 z^2 + ... in `z = s * s`.
/// Measured on every positive finite input: the largest error is 0.8348 units of the last place, and the function
/// never falls (the ignored test `ln_error_on_every_input`).
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    if x.to_bits() >= 0x7f80_0000 || x == 0.0 {
        return if x == 0.0 { f32::NEG_INFINITY } else if x.to_bits() == 0x7f80_0000 { x } else { f32::NAN };
    }
    // A subnormal input is scaled by 2^25 first. The scale is exact.
    let (bits, shift) = if x.to_bits() < 0x0080_0000 { ((x * 33_554_432.0).to_bits(), -25) } else { (x.to_bits(), 0) };
    let mut e = (bits >> 23) as i32 - 127 + shift;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if m > core::f32::consts::SQRT_2 { m *= 0.5; e += 1; }
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let mut r = 2.0 / 13.0f32;
    for c in [2.0 / 11.0, 2.0 / 9.0, 2.0 / 7.0, 2.0 / 5.0, 2.0 / 3.0] { r = madd(r, z, c); }
    let r = r * z;
    let half_square = 0.5 * f * f;
    let k = e as f32;
    k * LN2_HIGH - ((half_square - (s * (half_square + r) + k * LN2_LOW)) - f)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The distance of `got` from the true value `want`, in units of the last place of the FP32 value at `want`.
    fn ulps(got: f32, want: f64) -> f64 {
        let nearest = want as f32;
        let unit = f64::from(f32::from_bits(nearest.to_bits() & 0x7fff_ffff | 1)) - f64::from(f32::from_bits(nearest.to_bits() & 0x7fff_fffe));
        (f64::from(got) - want).abs() / unit
    }
    #[test] fn tanh_keeps_zero_sign_tiny_values_and_its_limits() {
        assert_eq!(tanh(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(tanh(-0.0).to_bits(), (-0.0f32).to_bits());
        for bits in [1u32, 0x0080_0000, 0x3000_0000, 0x3800_0000] {
            let x = f32::from_bits(bits);
            assert_eq!(tanh(x).to_bits(), bits, "a tiny input is its own tangent");
            assert_eq!(tanh(-x).to_bits(), bits | 0x8000_0000);
        }
        for x in [9.02f32, 10.0, 43.5, 44.0, 1.0e30, f32::INFINITY] {
            assert_eq!(tanh(x), 1.0); assert_eq!(tanh(-x), -1.0);
        }
        assert!(tanh(f32::NAN).is_nan());
        assert!(tanh(9.0) < 1.0);
    }
    #[test] fn tanh_is_odd_and_close_to_the_true_value_on_both_sides_of_the_branch() {
        // 64-bit tanh is the reference: its error is far below one FP32 unit.
        let mut worst = 0.0f64;
        let mut bits = 0x3300_0000u32;
        while bits < 0x4120_0000 {
            let x = f32::from_bits(bits);
            assert_eq!(tanh(-x).to_bits(), tanh(x).to_bits() | 0x8000_0000);
            worst = worst.max(ulps(tanh(x), f64::from(x).tanh()));
            bits += 4099;
        }
        for x in [TANH_SERIES_BELOW, f32::from_bits(TANH_SERIES_BELOW.to_bits() - 1), f32::from_bits(TANH_SERIES_BELOW.to_bits() + 1)] {
            worst = worst.max(ulps(tanh(x), f64::from(x).tanh()));
        }
        assert!(worst < 1.0, "largest tanh error {worst} units");
    }
    #[test] fn tanh_does_not_fall_across_the_branch() {
        let mut bits = TANH_SERIES_BELOW.to_bits() - 4096;
        let mut last = tanh(f32::from_bits(bits));
        while bits < TANH_SERIES_BELOW.to_bits() + 4096 {
            bits += 1;
            let next = tanh(f32::from_bits(bits));
            assert!(next >= last, "tanh falls at {bits:#x}");
            last = next;
        }
    }
    #[test] fn ln_handles_its_edges_and_exact_powers_of_two() {
        assert_eq!(ln(1.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(ln(0.0), f32::NEG_INFINITY); assert_eq!(ln(-0.0), f32::NEG_INFINITY);
        assert!(ln(-1.0).is_nan() && ln(f32::NAN).is_nan() && ln(f32::NEG_INFINITY).is_nan());
        assert_eq!(ln(f32::INFINITY), f32::INFINITY);
        for power in -149i32..=127 {
            let exact = 2.0f64.powi(power) as f32;
            assert!(ulps(ln(exact), f64::from(power) * std::f64::consts::LN_2) < 1.0, "ln(2^{power})");
        }
    }
    #[test] fn ln_is_close_to_the_true_value() {
        let mut worst = 0.0f64;
        let mut bits = 1u32;
        while bits < 0x7f80_0000 {
            let x = f32::from_bits(bits);
            worst = worst.max(ulps(ln(x), f64::from(x).ln()));
            bits += 65_521;
        }
        // The range the position factor of a full attention layer uses: 1 to 128.
        let mut bits = 1.0f32.to_bits();
        while bits <= 128.0f32.to_bits() {
            let x = f32::from_bits(bits);
            worst = worst.max(ulps(ln(x), f64::from(x).ln()));
            bits += 257;
        }
        assert!(worst < 1.0, "largest ln error {worst} units");
    }
    #[test] fn round_and_sigmoid_keep_their_definitions() {
        // BF16 keeps the high 16 bits. A tie goes to the even pattern; one bit above a tie goes up.
        for (bits, want) in [(0x3f80_0000u32, 0x3f80_0000u32), (0x3f80_8000, 0x3f80_0000), (0x3f81_8000, 0x3f82_0000),
            (0x3f80_8001, 0x3f81_0000), (0xbf80_8001, 0xbf81_0000), (0x3f80_7fff, 0x3f80_0000), (0, 0), (0x8000_0000, 0x8000_0000)] {
            assert_eq!(rn(f32::from_bits(bits)).to_bits(), want, "rn of {bits:#x}");
        }
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(exp(0.0), 1.0);
    }
}
