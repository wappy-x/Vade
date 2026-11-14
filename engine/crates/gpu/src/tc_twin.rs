//! Kernel kit, part 1: the CPU twin (reference) of the tensor-core contracts, with hard inputs.
//!
//! This is the INDEPENDENT model: wide integers, written from the instruction's measured behavior, sharing no code with
//! `tc_contract` (the fast form that kernels run). Every kernel of the family is tested against this file: on the Mac
//! by the unit tests of `tc_contract`, on a box by `kernel-bench --fp4-cohort`. Std only.
//!
//! NVFP4 step = one `mma.sync m16n8k64 kind::mxf4nvf4 block_scale scale_vec::4X` for one output word.
//! Measured on GB10 (a lane box): lane multi-row 4,194,304 words; FP4-COHORT run 2 and night-1 ticket 1 (calibration
//! fixtures A and B, 8,192 raw words each; 474,368 gate word checks).

#[derive(Clone, Copy)]
pub struct Val { pub neg: bool, pub mag: u128, pub exp: i32, pub top: i32 }
/// E2M1 magnitude in half units.
pub const HALF: [i64; 8] = [0, 1, 2, 3, 4, 6, 8, 12];
pub fn half(code: u8) -> i64 { let h = HALF[(code & 7) as usize]; if code & 8 != 0 { -h } else { h } }
/// UE4M3 (sign bit ignored, as the instruction reads it): significand with 3 fraction bits and the format exponent.
pub fn ue4m3(byte: u8) -> (u128, i32) { let b = byte & 0x7f; let (e, m) = ((b >> 3) as i32, (b & 7) as u128); if e == 0 { (m, -6) } else { (8 + m, e - 7) } }
pub fn ue4m3_f32(byte: u8) -> f32 { let (m, e) = ue4m3(byte); m as f32 * 2f32.powi(e - 3) }
pub fn f32_val(bits: u32) -> Val {
    let (e, m) = ((bits >> 23 & 255) as i32, (bits & 0x7fffff) as u128);
    let neg = bits >> 31 != 0;
    if e == 0 { Val { neg, mag: m, exp: -126 - 23, top: -126 } } else { Val { neg, mag: (1 << 23) + m, exp: e - 127 - 23, top: e - 127 } }
}
/// sign * mag * 2^exp to FP32 bits, cut toward zero.
pub fn to_f32(neg: bool, mag: u128, exp: i32) -> u32 {
    let sign = (neg as u32) << 31;
    if mag == 0 { return sign; }
    let top = 127 - mag.leading_zeros() as i32;
    let e = top + exp;
    let low = if e < -126 { -126 - 23 } else { e - 23 };
    let shift = low - exp;
    let kept = if shift <= 0 { mag << (-shift) as u32 } else if shift >= 128 { 0 } else { mag >> shift as u32 };
    if e < -126 { return sign | kept as u32; }
    if e > 127 { return sign | 0x7f80_0000; }
    sign | ((e + 127) as u32) << 23 | (kept as u32 & 0x7f_ffff)
}
/// The model family the calibration fixtures separate. `DEVICE_MODEL` is the one the device fits.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Twin { pub zero_terms_align: bool, pub products: bool }
/// A 16-column block whose sum is zero still counts in the alignment when at least one of its 16 products is nonzero
/// (a sum cancelled to zero), and does not count when every product is zero (all-zero codes on either side).
pub const DEVICE_MODEL: Twin = Twin { zero_terms_align: true, products: true };
/// One instruction for one output word: `w`, `a` = the 64 codes of the step (one per byte), `sw`, `sa` = the four
/// scale codes, `c` = the old accumulator bits.
pub fn step(t: Twin, w: &[u8], a: &[u8], sw: &[u8], sa: &[u8], c: u32) -> u32 {
    if c & 0x7f80_0000 == 0x7f80_0000 { return c; }
    let mut terms = [Val { neg: false, mag: 0, exp: 0, top: 0 }; 4];
    let mut live = [false; 4];
    for q in 0..4 {
        let mut sum = 0i64;
        for e in 0..16 { let x = half(w[q * 16 + e]) * half(a[q * 16 + e]); sum += x; live[q] |= x != 0; }
        let ((mw, ew), (ma, ea)) = (ue4m3(sw[q]), ue4m3(sa[q]));
        terms[q] = Val { neg: sum < 0, mag: sum.unsigned_abs() as u128 * mw * ma, exp: ew + ea - 8, top: ew + ea };
    }
    let cv = f32_val(c);
    let mut top = i32::MIN / 4;
    for (q, x) in terms.iter().enumerate() { if x.mag != 0 || (t.zero_terms_align && (!t.products || live[q])) { top = top.max(x.top); } }
    top = top.max(cv.top);
    let unit = top - 35;
    let mut sum: i128 = 0;
    for x in terms.iter().chain(std::iter::once(&cv)) {
        if x.mag == 0 { continue; }
        let shift = x.exp - unit;
        let kept = if shift >= 0 { (x.mag as i128) << shift as u32 } else if -shift >= 127 { 0 } else { (x.mag >> (-shift) as u32) as i128 };
        sum += if x.neg { -kept } else { kept };
    }
    to_f32(sum < 0, sum.unsigned_abs(), unit)
}
pub fn bf16_round(x: f32) -> f32 { f32::from_bits(x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) >> 16 << 16) }
/// The raw accumulator of one word: the chain over all 64-column steps of a row (codes one a byte, scales one a block).
pub fn chain(w: &[u8], a: &[u8], sw: &[u8], sa: &[u8]) -> u32 {
    let mut acc = 0u32;
    for k in 0..w.len().div_ceil(64) {
        // A last step with fewer than four blocks: the absent blocks have zero codes (no product, so they never count).
        let n = (w.len() - k * 64).min(64);
        let (mut ws, mut as_, mut sws, mut sas) = ([0u8; 64], [0u8; 64], [0u8; 4], [0u8; 4]);
        ws[..n].copy_from_slice(&w[k * 64..][..n]);
        as_[..n].copy_from_slice(&a[k * 64..][..n]);
        sws[..n / 16].copy_from_slice(&sw[k * 4..][..n / 16]);
        sas[..n / 16].copy_from_slice(&sa[k * 4..][..n / 16]);
        acc = step(DEVICE_MODEL, &ws, &as_, &sws, &sas, acc);
    }
    acc
}
/// The raw accumulator of one word of a SECOND matrix: four chains, one a quarter of the columns, each from +0.0,
/// joined fl(fl(a0 + a1) + fl(a2 + a3)) with round-to-nearest-even FP32 adds.
pub fn chain_quarters(w: &[u8], a: &[u8], sw: &[u8], sa: &[u8]) -> u32 {
    let q = w.len() / 4;
    assert!(q > 0 && q.is_multiple_of(64), "a quarter must be whole 64-column steps");
    let part = |i: usize| f32::from_bits(chain(&w[i * q..][..q], &a[i * q..][..q], &sw[i * q / 16..][..q / 16], &sa[i * q / 16..][..q / 16]));
    ((part(0) + part(1)) + (part(2) + part(3))).to_bits()
}
/// The raw accumulator for `chains` = 1 (first matrices) or 4 (second matrices).
pub fn accumulator(chains: u32, w: &[u8], a: &[u8], sw: &[u8], sa: &[u8]) -> u32 {
    match chains { 1 => chain(w, a, sw, sa), 4 => chain_quarters(w, a, sw, sa), _ => panic!("chains must be 1 or 4") }
}
/// The final word: bf16(fl(acc * fl(global * activation))).
pub fn word(acc: u32, global: f32, activation: f32) -> u32 { bf16_round(f32::from_bits(acc) * (global * activation)).to_bits() }

// ------------------------------------------------------------------ hard inputs
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 { self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15); let mut z = self.0; z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9); z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb); z ^ (z >> 31) }
    pub fn below(&mut self, n: u64) -> u64 { self.next_u64() % n }
}
/// One step with its old accumulator, and the name of its class.
pub struct StepCase { pub class: &'static str, pub w: [u8; 64], pub a: [u8; 64], pub sw: [u8; 4], pub sa: [u8; 4], pub acc: u32 }
/// The hard classes of one step. Every run of a twin test covers each of them `n / CLASSES` times:
/// random; all-zero block under the largest scale; cancelled block (products nonzero, sum zero) under the largest
/// scale; zero and subnormal scale codes; extreme scale spread; old accumulator that nearly cancels the step;
/// subnormal old accumulator; signed-zero old accumulator; infinite and NaN old accumulator (passes through);
/// sparse codes (most activations +-0); every code pair in one block; largest magnitudes everywhere.
pub const CLASSES: usize = 12;
pub fn hard_steps(n: usize, seed: u64) -> Vec<StepCase> {
    let mut r = Rng(seed);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut c = StepCase { class: "random", w: [0; 64], a: [0; 64], sw: [0; 4], sa: [0; 4], acc: 0 };
        for e in 0..64 { c.w[e] = r.below(16) as u8; c.a[e] = r.below(16) as u8; }
        for q in 0..4 { c.sw[q] = r.below(0x7f) as u8; c.sa[q] = r.below(0x7f) as u8; }
        c.acc = if r.below(4) == 0 { 0 } else { random_f32(&mut r) };
        let big = r.below(4) as usize;
        match i % CLASSES {
            0 => {}
            1 => { c.class = "zero block under the largest scale"; for e in 0..16 { c.w[big * 16 + e] &= 8; } small_scales(&mut c, &mut r, big); }
            2 => { c.class = "cancelled block under the largest scale"; for e in 0..16 { c.w[big * 16 + e] = [2, 10][e % 2]; c.a[big * 16 + e] = 2; } small_scales(&mut c, &mut r, big); }
            3 => { c.class = "zero and subnormal scales"; for q in 0..4 { c.sw[q] = r.below(9) as u8; c.sa[q] = r.below(9) as u8; } }
            4 => { c.class = "scale spread"; for q in 0..4 { c.sw[q] = if r.below(2) == 0 { 1 + r.below(8) as u8 } else { 0x70 + r.below(15) as u8 }; c.sa[q] = if r.below(2) == 0 { 1 + r.below(8) as u8 } else { 0x70 + r.below(15) as u8 }; } }
            5 => { c.class = "old accumulator nearly cancels the step"; let s = step(DEVICE_MODEL, &c.w, &c.a, &c.sw, &c.sa, 0); c.acc = (s ^ 0x8000_0000).wrapping_add(r.below(5) as u32).wrapping_sub(2); if c.acc & 0x7f80_0000 == 0x7f80_0000 { c.acc = 0; } }
            6 => { c.class = "subnormal old accumulator"; c.acc = (r.next_u64() as u32 & 0x807f_ffff) | 1; for q in 0..4 { c.sw[q] = r.below(4) as u8; c.sa[q] = r.below(4) as u8; } }
            7 => { c.class = "signed-zero old accumulator"; c.acc = (r.below(2) as u32) << 31; if r.below(2) == 0 { for e in 0..64 { c.a[e] &= 8; } } }
            8 => { c.class = "infinite or NaN old accumulator"; c.acc = 0x7f80_0000 | ((r.below(2) as u32) << 31) | if r.below(2) == 0 { 0 } else { r.next_u64() as u32 & 0x7f_ffff }; }
            9 => { c.class = "sparse codes"; for e in 0..64 { if r.below(10) < 8 { c.a[e] &= 8; } } }
            10 => { c.class = "every code pair"; let base = (i / CLASSES) % 4; for e in 0..64 { let k = (i / CLASSES / 4 * 64 + e) % 256; c.w[e] = (k >> 4) as u8; c.a[e] = (k & 15) as u8; } let _ = base; }
            _ => { c.class = "largest magnitudes"; let sign = r.below(2) as u8 * 8; for e in 0..64 { c.w[e] = 7 | (r.below(2) as u8 * 8); c.a[e] = 7 | sign; } for q in 0..4 { c.sw[q] = 0x7e; c.sa[q] = 0x7e; } }
        }
        out.push(c);
    }
    out
}
fn small_scales(c: &mut StepCase, r: &mut Rng, big: usize) {
    for q in 0..4 {
        if q == big { c.sw[q] = 0x7e; c.sa[q] = 0x7e; } else { c.sw[q] = 0x08 + r.below(8) as u8; c.sa[q] = 1 + r.below(7) as u8; }
    }
}
fn random_f32(r: &mut Rng) -> u32 {
    // Finite, any sign, exponents around the range a step can reach (about 2^-30 .. 2^40) and some far ones.
    let e = if r.below(8) == 0 { r.below(254) as u32 + 1 } else { 97 + r.below(70) as u32 };
    ((r.below(2) as u32) << 31) | (e << 23) | (r.next_u64() as u32 & 0x7f_ffff)
}
/// One whole row (weight codes, activation codes, one scale a block) that mixes the hard classes step by step.
pub struct Row { pub w: Vec<u8>, pub a: Vec<u8>, pub sw: Vec<u8>, pub sa: Vec<u8> }
pub fn hard_row(cols: usize, seed: u64) -> Row {
    let steps = hard_steps(cols / 64, seed);
    let mut row = Row { w: vec![], a: vec![], sw: vec![], sa: vec![] };
    for s in &steps { row.w.extend(s.w); row.a.extend(s.a); row.sw.extend(s.sw); row.sa.extend(s.sa); }
    row
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_small_sums() {
        // 64 products of 1.0 x 1.0 with unit scales.
        assert_eq!(f32::from_bits(step(DEVICE_MODEL, &[2; 64], &[2; 64], &[0x38; 4], &[0x38; 4], 0)), 64.0);
        // An infinite accumulator passes through with its payload.
        assert_eq!(step(DEVICE_MODEL, &[2; 64], &[2; 64], &[0x38; 4], &[0x38; 4], 0x7f80_0001), 0x7f80_0001);
        // All-zero codes and a zero accumulator give +0.
        assert_eq!(step(DEVICE_MODEL, &[8; 64], &[2; 64], &[0x7e; 4], &[0x7e; 4], 0), 0);
    }
    /// The three calibration models differ on the two fixture kinds exactly as the device run showed:
    /// an all-zero block under the largest scale separates "always counts"; a cancelled block separates "never counts".
    #[test]
    fn the_model_variants_are_separated_by_the_hard_classes() {
        let always = Twin { zero_terms_align: true, products: false };
        let never = Twin { zero_terms_align: false, products: false };
        let (mut zero_differs, mut cancel_differs) = (0, 0);
        for c in hard_steps(24_000, 9) {
            let d = step(DEVICE_MODEL, &c.w, &c.a, &c.sw, &c.sa, c.acc);
            if c.class == "zero block under the largest scale" && step(always, &c.w, &c.a, &c.sw, &c.sa, c.acc) != d { zero_differs += 1; }
            if c.class == "cancelled block under the largest scale" && step(never, &c.w, &c.a, &c.sw, &c.sa, c.acc) != d { cancel_differs += 1; }
        }
        assert!(zero_differs > 100 && cancel_differs > 100, "{zero_differs} {cancel_differs}");
    }
    #[test]
    fn every_class_is_present() {
        let mut names: Vec<&str> = hard_steps(CLASSES * 3, 1).iter().map(|c| c.class).collect();
        names.sort(); names.dedup();
        assert_eq!(names.len(), CLASSES);
    }
}
