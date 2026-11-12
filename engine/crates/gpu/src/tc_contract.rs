//! Contract (b) of the NVFP4 routed-expert product, as integer code. Host and device share this file.
//!
//! The word of one weight row and one token row: `acc = +0.0`; for each 64-column step in ascending column order
//! `acc = chain_step(acc, ..)`; `word = bf16(fl(acc * fl(global * activation)))`. `chain_step` is the integer function
//! that GB10's `mma.sync m16n8k64 kind::mxf4nvf4 block_scale scale_vec::4X` computes for one output word (measured:
//! every word of 4,194,304 on lane multi-row, and every word of the FP4-COHORT gates): each 16-column block sum is
//! exact; times its two UE4M3 block scales, exact; the four terms and the old accumulator are lined up on the largest
//! exponent, where a block counts only when one of its 16 products is not zero; bits under 35 fraction bits are cut
//! toward zero, term by term; the sum is exact; the result is cut toward zero to FP32.
//!
//! TWO WORD FUNCTIONS (runner ruling 2026-11-11 02:16, one rebase):
//! - FIRST matrices (expert gate and up; q, k, v; routers): `chains` = 1. One chain over all columns, as above.
//! - SECOND matrices (expert down, shared down, dense down, the o projections): `chains` = 4, the quarter chain.
//!   Quarter q = columns [q * cols/4, (q+1) * cols/4); cols/4 is a multiple of 64. acc_q = one ascending chain of
//!   64-column steps over quarter q, anchored at the quarter's first column, from +0.0. Then
//!   acc = fl(fl(acc_0 + acc_1) + fl(acc_2 + acc_3)), where fl is one IEEE round-to-nearest-even FP32 add (never
//!   fused, never reordered), and the epilogue runs on acc. The join order is fixed by the column index only, so 1, 2
//!   and 4 ranks give the same bits when a later step splits these matrices by columns (a rank then owns whole
//!   quarters and sends its partial join).
//!
//! The tensor-core kernels (`fp4_cohort*`) run the instruction. The CUDA-core kernels (`nv_cols_b`, `fp4_cohort_cc`)
//! run this file. `tc_twin` is the independent reference both are tested against. No unsafe code, no float operation.

/// Chains a word of a first matrix and of a second matrix.
pub const CHAINS_FIRST: u32 = 1;
pub const CHAINS_SECOND: u32 = 4;
/// 64-column steps in one chain of a word, when the shape admits the word function.
#[inline(always)]
pub fn chain_steps(cols: usize, chains: u32) -> Option<usize> {
    match chains {
        1 if cols > 0 && cols.is_multiple_of(64) => Some(cols / 64),
        4 if cols > 0 && cols.is_multiple_of(256) => Some(cols / 256),
        _ => None,
    }
}
/// The quarter join on the host: fl(fl(a0 + a1) + fl(a2 + a3)). Kernels use two `add.rn.f32` per pair in this order.
#[cfg(not(target_arch = "nvptx64"))]
pub fn join4(a: [u32; 4]) -> u32 {
    let f = |x: u32| f32::from_bits(x);
    ((f(a[0]) + f(a[1])) + (f(a[2]) + f(a[3]))).to_bits()
}

/// E2M1 magnitudes in half units (0, 0.5, 1, 1.5, 2, 3, 4, 6 -> 0, 1, 2, 3, 4, 6, 8, 12), one nibble a code.
pub const E2M1_HALF: u32 = 0xc864_3210;

/// Signed half units of an E2M1 code (bit 3 = sign).
#[inline(always)]
pub fn half_units(code: u32) -> i32 {
    let mag = ((E2M1_HALF >> ((code & 7) * 4)) & 15) as i32;
    if code & 8 != 0 { -mag } else { mag }
}

/// Signed half units of a decoded E2M1 value given as FP32 bits (the activation quantizer writes these values:
/// +-0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6). Any other input is outside the contract.
#[inline(always)]
pub fn half_units_f32(bits: u32) -> i32 {
    let x = bits & 0x7fff_ffff;
    if x == 0 { return 0; }
    // value * 2 = (2 + m) * 2^e, m = the first fraction bit, e = the exponent (-1 ..= 2).
    let e = (x >> 23) as i32 - 127;
    let mag = (((2 + ((x >> 22) & 1)) << ((e + 1) as u32 & 3)) >> 1) as i32;
    if bits >> 31 != 0 { -mag } else { mag }
}

/// UE4M3 scale code to (significand with 3 fraction bits, format exponent). The sign bit is ignored, as the
/// instruction ignores it. Code 0x7f (NaN) is outside the contract: loaders and the quantizer never produce it.
#[inline(always)]
pub fn ue4m3_parts(byte: u32) -> (u32, i32) {
    let b = byte & 0x7f;
    let (e, m) = ((b >> 3) as i32, b & 7);
    if e == 0 { (m, -6) } else { (8 + m, e - 7) }
}

/// The same parts from a decoded UE4M3 value given as FP32 bits (word 16 of an activation block).
#[inline(always)]
pub fn ue4m3_parts_f32(bits: u32) -> (u32, i32) {
    let x = bits & 0x7fff_ffff;
    if x == 0 { return (0, -6); }
    let e = (x >> 23) as i32 - 127;
    if e >= -6 { (8 + ((x >> 20) & 7), e) } else { ((((x & 0x7f_ffff) | 0x80_0000) >> ((14 - e) as u32 & 31)), -6) }
}

/// E2M1 code (bit 3 = sign, -0.0 keeps its sign) of a decoded E2M1 value given as FP32 bits.
#[inline(always)]
pub fn e2m1_code_f32(bits: u32) -> u32 {
    let x = bits & 0x7fff_ffff;
    let sign = (bits >> 31) << 3;
    if x == 0 { return sign; }
    let e = (x >> 23) as i32 - 127;
    sign | if e < 0 { 1 } else { (2 * (e as u32 + 1) + ((x >> 22) & 1)) & 7 }
}
/// UE4M3 code of a decoded UE4M3 value given as FP32 bits.
#[inline(always)]
pub fn ue4m3_code_f32(bits: u32) -> u32 {
    let (m, e) = ue4m3_parts_f32(bits);
    if m >= 8 { (((e + 7) as u32) << 3) | (m - 8) } else { m }
}

/// FP16 bits of a decoded E2M1 value given as FP32 bits (exact: +-0, +-0.5 .. +-6 are FP16 values).
#[inline(always)]
pub fn f16_bits_e2m1(bits: u32) -> u32 {
    let x = bits & 0x7fff_ffff;
    let sign = (bits >> 31) << 15;
    if x == 0 { return sign; }
    sign | ((((x >> 23) + 15 - 127) & 31) << 10) | (((x >> 22) & 1) << 9)
}

/// (significand, exponent) in one word, for a shared-memory slot.
#[inline(always)]
pub fn pack_parts(p: (u32, i32)) -> u32 { p.0 | (((p.1 + 16) as u32) << 8) }
#[inline(always)]
pub fn unpack_parts(w: u32) -> (u32, i32) { (w & 0xff, (w >> 8) as i32 - 16) }

/// sign * mag * 2^exp to FP32 bits, cut toward zero.
#[inline(always)]
pub fn cut_f32(neg: bool, mag: u64, exp: i32) -> u32 {
    let sign = (neg as u32) << 31;
    if mag == 0 { return sign; }
    let top = 63 - mag.leading_zeros() as i32;
    let e = top + exp;
    let low = if e < -126 { -149 } else { e - 23 };
    let shift = low - exp;
    let kept = if shift <= 0 { mag << ((-shift) as u32) } else if shift >= 64 { 0 } else { mag >> (shift as u32) };
    if e < -126 { return sign | kept as u32; }
    if e > 127 { return sign | 0x7f80_0000; }
    sign | (((e + 127) as u32) << 23) | (kept as u32 & 0x7f_ffff)
}

/// One 64-column step for one output word. `sum[q]` = the exact sum over the 16 columns of block q of
/// (weight half units * activation half units); `live[q]` = one of those 16 products is not zero; `w[q]`, `a[q]` = the
/// UE4M3 parts of the block's weight and activation scales; `acc` = the old accumulator bits.
#[inline(always)]
pub fn chain_step(acc: u32, sum: [i32; 4], live: [bool; 4], w: [(u32, i32); 4], a: [(u32, i32); 4]) -> u32 {
    if acc & 0x7f80_0000 == 0x7f80_0000 { return acc; }
    // A block term is mag * 2^(e - 8) with mag = |sum| * mw * ma below 2^20 and e = ew + ea.
    let mut mag = [0u32; 4];
    let mut e = [0i32; 4];
    let mut top = i32::MIN / 4;
    let mut q = 0;
    while q < 4 {
        mag[q] = sum[q].unsigned_abs() * (w[q].0 * a[q].0);
        e[q] = w[q].1 + a[q].1;
        if (mag[q] != 0 || live[q]) && e[q] > top { top = e[q]; }
        q += 1;
    }
    let (ce, cm) = (((acc >> 23) & 255) as i32, acc & 0x7f_ffff);
    let (cmag, ctop) = if ce == 0 { (cm, -126) } else { (cm | 1 << 23, ce - 127) };
    if ctop > top { top = ctop; }
    // Everything in units of 2^(top - 35), cut toward zero: a block term is (mag << 27) >> (top - e); the old
    // accumulator, cmag * 2^(ctop - 23), is (cmag << 12) >> (top - ctop). A distance of 64 or more leaves nothing
    // (a term that did not count has mag 0, whatever its distance).
    let mut total: i64 = 0;
    let mut q = 0;
    while q < 5 {
        let (neg, wide, d) = if q < 4 { (sum[q] < 0, (mag[q] as u64) << 27, (top - e[q]) as u32) } else { (acc >> 31 != 0, (cmag as u64) << 12, (top - ctop) as u32) };
        let kept = if d < 64 { (wide >> d) as i64 } else { 0 };
        total += if neg { -kept } else { kept };
        q += 1;
    }
    cut_f32(total < 0, total.unsigned_abs(), top - 35)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tc_twin as twin;

    #[test]
    fn code_and_value_decoders_agree_with_the_formats() {
        for code in 0..16u32 {
            let value = crate::expert_ops::e2m1(code as u8);
            assert_eq!(half_units(code), (value * 2.0) as i32, "code {code}");
            assert_eq!(half_units_f32(value.to_bits()), half_units(code), "value of code {code}");
            assert_eq!(half_units(code) as i64, twin::half(code as u8));
            // FP16: 0, 0.5, 1, 1.5, 2, 3, 4, 6 are 0x0000, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400, 0x4600.
            let f16 = [0x0000u32, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400, 0x4600][(code & 7) as usize] | ((code & 8) << 12);
            assert_eq!(f16_bits_e2m1(value.to_bits()), f16, "FP16 bits of the value of code {code}");
            let signed = if code & 8 != 0 { -crate::expert_ops::e2m1(code as u8 & 7) } else { value };
            assert_eq!(e2m1_code_f32(signed.to_bits()), code, "code of the value of {code}");
        }
        for byte in 0..127u32 {
            let value = crate::expert_ops::e4m3(byte as u8);
            let (m, e) = ue4m3_parts(byte);
            assert_eq!(m as f32 * 2f32.powi(e - 3), value, "scale code {byte}");
            assert_eq!(ue4m3_parts_f32(value.to_bits()), if m == 0 { (0, -6) } else { (m, e) }, "scale value of code {byte}");
            assert_eq!(unpack_parts(pack_parts((m, e))), (m, e));
            assert_eq!(ue4m3_code_f32(value.to_bits()), byte, "scale code of the value of {byte}");
            let (tm, te) = twin::ue4m3(byte as u8);
            assert_eq!((tm as u32, te), (m, e));
        }
        // The sign bit of a scale code is ignored.
        assert_eq!(ue4m3_parts(0x80 | 0x3a), ue4m3_parts(0x3a));
    }

    /// The shared integer step equals the independent reference on every hard class and on random steps.
    #[test]
    fn chain_step_equals_the_reference_on_hard_inputs() {
        let mut checked = 0usize;
        for case in twin::hard_steps(20_000, 0x7c) {
            let mut sum = [0i32; 4];
            let mut live = [false; 4];
            let (mut w, mut a) = ([(0u32, 0i32); 4], [(0u32, 0i32); 4]);
            for q in 0..4 {
                for e in 0..16 {
                    let p = half_units(case.w[q * 16 + e] as u32) * half_units(case.a[q * 16 + e] as u32);
                    sum[q] += p;
                    live[q] |= p != 0;
                }
                w[q] = ue4m3_parts(case.sw[q] as u32);
                a[q] = ue4m3_parts(case.sa[q] as u32);
            }
            let want = twin::step(twin::DEVICE_MODEL, &case.w, &case.a, &case.sw, &case.sa, case.acc);
            assert_eq!(chain_step(case.acc, sum, live, w, a), want, "class {} acc {:08x}", case.class, case.acc);
            checked += 1;
        }
        assert!(checked >= 20_000);
    }

    #[test]
    fn the_quarter_word_is_four_chains_and_the_fixed_join() {
        assert_eq!(chain_steps(2048, 4), Some(8));
        assert_eq!(chain_steps(4096, 1), Some(64));
        assert_eq!(chain_steps(320, 4), None);
        for (seed, cols) in [(11u64, 256usize), (12, 2048), (13, 4096)] {
            let fx = twin::hard_row(cols, seed);
            let quarter = cols / 4;
            let parts: [u32; 4] = std::array::from_fn(|q| twin::chain(&fx.w[q * quarter..][..quarter], &fx.a[q * quarter..][..quarter], &fx.sw[q * quarter / 16..][..quarter / 16], &fx.sa[q * quarter / 16..][..quarter / 16]));
            assert_eq!(join4(parts), twin::chain_quarters(&fx.w, &fx.a, &fx.sw, &fx.sa));
            // The two word functions are different functions: they must not be mixed for one matrix.
            let _ = twin::chain(&fx.w, &fx.a, &fx.sw, &fx.sa);
        }
        // Join order: (a0 + a1) + (a2 + a3), not a left-to-right sum.
        let a = [1.0f32, 2f32.powi(-24), 2f32.powi(-24), -1.0].map(f32::to_bits);
        assert_eq!(f32::from_bits(join4(a)), (1.0f32 + 2f32.powi(-24)) + (2f32.powi(-24) + -1.0f32));
        assert_eq!(f32::from_bits(join4(a)), 2f32.powi(-24));
        assert_eq!(((1.0f32 + 2f32.powi(-24)) + 2f32.powi(-24)) + -1.0f32, 0.0);
    }

    /// A whole word through the shared code (FP32 activation rows, as the decode kernel reads them) equals the reference.
    #[test]
    fn a_word_from_decoded_activation_rows_equals_the_reference() {
        for (seed, cols) in [(1u64, 64usize), (2, 256), (3, 1024), (4, 4096)] {
            let fx = twin::hard_row(cols, seed);
            let mut acc = 0u32;
            for k in 0..cols / 64 {
                let mut sum = [0i32; 4];
                let mut live = [false; 4];
                let (mut w, mut a) = ([(0u32, 0i32); 4], [(0u32, 0i32); 4]);
                for q in 0..4 {
                    let b = k * 4 + q;
                    for e in 0..16 {
                        let value = crate::expert_ops::e2m1(fx.a[b * 16 + e]);
                        let p = half_units(fx.w[b * 16 + e] as u32) * half_units_f32(value.to_bits());
                        sum[q] += p;
                        live[q] |= p != 0;
                    }
                    w[q] = ue4m3_parts(fx.sw[b] as u32);
                    a[q] = unpack_parts(pack_parts(ue4m3_parts_f32(crate::expert_ops::e4m3(fx.sa[b] & 0x7f).to_bits())));
                }
                acc = chain_step(acc, sum, live, w, a);
            }
            assert_eq!(acc, twin::chain(&fx.w, &fx.a, &fx.sw, &fx.sa), "cols {cols}");
        }
    }
}
