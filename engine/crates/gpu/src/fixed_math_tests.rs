//! Compare the copies in `fixed_math.rs` with their origins in this crate, bit for bit.
//! Measure `tanh` and `ln` on every input pattern. An error bound of a shared math function is a sweep of every
//! pattern, never a sample of points.
use crate::{fixed_math, next_math};

/// Count the patterns from `first`, in steps of `step`, where a copy differs from its origin.
fn differences(first: u64, step: u64) -> u64 {
    let mut differ = 0;
    let mut bits = first;
    while bits < 1 << 32 {
        let x = f32::from_bits(bits as u32);
        let (copy, origin) = (fixed_math::exp(x), next_math::exp(x));
        if copy.to_bits() != origin.to_bits() && !(copy.is_nan() && origin.is_nan()) { differ += 1; }
        if fixed_math::rn(x).to_bits() != crate::causal_ops::round(x).to_bits() { differ += 1; }
        let (copy, origin) = (fixed_math::sigmoid(x), crate::causal_ops::sigmoid(x));
        if copy.to_bits() != origin.to_bits() && !(copy.is_nan() && origin.is_nan()) { differ += 1; }
        bits += step;
    }
    differ
}
#[test] fn copies_equal_their_origins_on_a_spread_of_patterns() {
    // 4,099 is prime, so the steps visit every exponent and many low-bit patterns: 1,047,809 inputs.
    assert_eq!(differences(0, 4099), 0);
    // Every pattern near the clamps of exp, near zero, near one and at the specials.
    for center in [0x42ae_0000u64, 0xc2ae_0000, 0, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x7f80_0000, 0xff80_0000] {
        let first = center.saturating_sub(4096);
        let mut differ = 0;
        for bits in first..(center + 4096).min(1 << 32) { differ += differences(bits, 1 << 32); }
        assert_eq!(differ, 0, "near {center:#x}");
    }
}
/// The full proof: every 32-bit pattern. About 1 minute on one core, so it does not run by default.
#[test] #[ignore = "all 4,294,967,296 patterns; run it when fixed_math.rs or next_math::exp changes"]
fn copies_equal_their_origins_on_every_pattern() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get() as u64);
    let differ: u64 = std::thread::scope(|scope| (0..threads).map(|first| scope.spawn(move || differences(first, threads)))
        .collect::<Vec<_>>().into_iter().map(|handle| handle.join().unwrap()).sum());
    assert_eq!(differ, 0);
}

/// The distance of `got` from the true value `want`, in units of the last place of the FP32 value at `want`.
fn ulps(got: f32, want: f64) -> f64 {
    let nearest = want as f32;
    let unit = f64::from(f32::from_bits(nearest.to_bits() & 0x7fff_ffff | 1)) - f64::from(f32::from_bits(nearest.to_bits() & 0x7fff_fffe));
    (f64::from(got) - want).abs() / unit
}
/// Run `f` on every pattern from `first` up to `end`, on all cores. The 64-bit `exact` is the reference: its own
/// error is far below one FP32 unit. Returns the largest error in units of the last place, and the count of
/// patterns where the result is below the result of the pattern before.
fn sweep(first: u32, end: u32, f: impl Fn(f32) -> f32 + Copy + Send, exact: impl Fn(f64) -> f64 + Copy + Send) -> (f64, u64) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get() as u64);
    let (first, end) = (u64::from(first), u64::from(end));
    let step = (end - first).div_ceil(threads);
    std::thread::scope(|scope| (0..threads).map(|part| scope.spawn(move || {
        let (from, to) = ((first + part * step).min(end), (first + (part + 1) * step).min(end));
        let (mut worst, mut falls) = (0.0f64, 0u64);
        let mut last = if from == first { f32::NEG_INFINITY } else { f(f32::from_bits(from as u32 - 1)) };
        for bits in from..to {
            let x = f32::from_bits(bits as u32);
            let y = f(x);
            worst = worst.max(ulps(y, exact(f64::from(x))));
            if y < last { falls += 1; }
            last = y;
        }
        (worst, falls)
    })).collect::<Vec<_>>().into_iter().map(|handle| handle.join().unwrap()).fold((0.0f64, 0u64), |all, part| (all.0.max(part.0), all.1 + part.1)))
}
/// The error bound of `tanh`: every non-negative finite pattern. A negative input has the same bits with the sign
/// set, so this is every finite input. Measured: 0.8830, and the function never falls.
#[test] #[ignore = "2,139,095,040 inputs on all cores; run it when tanh or exp changes, with --nocapture for the numbers"]
fn tanh_error_on_every_input() {
    let (worst, falls) = sweep(0, 0x7f80_0000, fixed_math::tanh, f64::tanh);
    println!("tanh: inputs 2139095040, largest error {worst:.4} units of the last place, places where it falls {falls}");
    assert!(worst < 0.884, "largest tanh error {worst}");
    assert_eq!(falls, 0);
}
/// The error bound of `ln`: every positive finite pattern. Measured: 0.8348, and the function never falls.
#[test] #[ignore = "2,139,095,039 inputs on all cores; run it when ln changes, with --nocapture for the numbers"]
fn ln_error_on_every_input() {
    let (worst, falls) = sweep(1, 0x7f80_0000, fixed_math::ln, f64::ln);
    println!("ln: inputs 2139095039, largest error {worst:.4} units of the last place, places where it falls {falls}");
    assert!(worst < 0.836, "largest ln error {worst}");
    assert_eq!(falls, 0);
}
