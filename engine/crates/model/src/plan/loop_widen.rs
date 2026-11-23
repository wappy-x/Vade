//! Turn stored 4-bit and 8-bit expert matrices into BF16 at load. Host arithmetic only.
//!
//! This file is the one place that makes the numbers of an expert weight for the form that the
//! reference code runs: the weight is widened and rounded to BF16 one time, at load. A form that
//! keeps the stored 4-bit and 8-bit weights exact replaces the caller of this file
//! (`loop_load::LoadForm::WidenedExperts`). Nothing else reads these functions.
//!
//! Reference: transformers v5.24.0, `core_model_loading.py` lines 772-816. This file follows
//! those lines and holds its own code.
//! - Line 786: the 16 values of a 4-bit code. Lines 802-804: the code pair of a byte, and the
//!   factor of one expert, `amax / outer_scale_denominator`, a 32-bit division.
//! - Lines 808-810: an 8-bit value times `2 ^ (scale - 127)`.
//! - Lines 812-814: a 4-bit code value times its block scale, then times the factor, in 32-bit.
//! - Line 815: the cast of the product to the type of the model parameter (BF16).
//! - Line 791 (comment): the loader casts every stored source tensor to the parameter type first.
//!   So the 32-bit `amax` is rounded to BF16 before the division. Measured on the loaded
//!   reference model: all 2,038,431,744 values of two 4-bit layers equal this rule, and 76.5
//!   percent equal the rule with the 32-bit `amax`. The receipt is outside this repository
//!   (the widened-expert dump index of the reference run of 2026-11-22).
use super::storage_groups::{ExpertRows, NibbleOrder};
use crate::{Result, flash::nvfp4::{bf16_to_f32, e2m1, e4m3, f32_to_bf16}, invalid};

/// Values of one row that share one scale.
pub const NVFP4_BLOCK: usize = 16;
pub const MXFP8_BLOCK: usize = 32;
/// An 8-bit scale byte `s` is the power of two `2 ^ (s - MXFP8_BIAS)`.
const MXFP8_BIAS: i32 = 127;

/// The reference loader rounds a stored 32-bit source to the parameter type before the widen step.
fn source_cast(value: f32) -> f32 { bf16_to_f32(f32_to_bf16(value)) }

/// The factor of one 4-bit expert: `amax / denominator`, both cast to BF16 first, divided in 32-bit.
pub fn expert_factor(amax: f32, denominator: f32) -> Result<f32> {
    if !(amax.is_finite() && amax > 0.0 && denominator.is_finite() && denominator > 0.0) {
        return Err(invalid("expert scale: amax and denominator must be positive and finite"));
    }
    let factor = source_cast(amax) / source_cast(denominator);
    if !(factor.is_finite() && factor > 0.0) { return Err(invalid("expert scale: the factor of an expert is not positive and finite")); }
    Ok(factor)
}
/// One 4-bit weight of the widened form: code value times block scale (exact in 32-bit), times the
/// factor of the expert (one 32-bit product), rounded to BF16.
pub fn nvfp4_weight(code: u8, scale: u8, factor: f32) -> f32 { e2m1(code) * e4m3(scale) * factor }
/// One 8-bit weight: the stored value times a power of two, in 32-bit.
pub fn mxfp8_weight(value: u8, scale: u8) -> f32 { e4m3(value) * power_of_two(scale) }
/// `2 ^ (scale - MXFP8_BIAS)` as a 32-bit float, made from its bits. The bias of the scale byte is the
/// exponent bias of a 32-bit float, so the byte is the exponent field. Byte 0 is below the normal
/// range and byte 255 is above the largest finite power.
fn power_of_two(scale: u8) -> f32 {
    const { assert!(MXFP8_BIAS == f32::MAX_EXP - 1) };
    match scale { 0 => f32::from_bits(1 << 22), 255 => f32::INFINITY, s => f32::from_bits(u32::from(s) << 23) }
}

/// Stored row that supplies row `row` of the widened matrix. With `Interleaved` the widened matrix
/// holds all gate rows, then all up rows: stored row `2i` is gate row `i`, `2i + 1` is up row `i`.
pub(super) fn stored_row(order: Option<ExpertRows>, row: usize, rows: usize) -> usize {
    match order {
        None => row,
        Some(ExpertRows::Interleaved) => { let half = rows / 2; if row < half { 2 * row } else { 2 * (row - half) + 1 } }
    }
}
fn extents(rows: usize, columns: usize, block: usize, per_byte: usize, order: Option<ExpertRows>,
    weights: usize, scales: usize, out: usize) -> Result<()> {
    let ok = rows > 0 && columns > 0 && columns.is_multiple_of(block)
        && (order.is_none() || rows.is_multiple_of(2))
        && rows.checked_mul(columns / per_byte) == Some(weights)
        && rows.checked_mul(columns / block) == Some(scales)
        && rows.checked_mul(columns).and_then(|n| n.checked_mul(2)) == Some(out);
    if ok { Ok(()) } else { Err(invalid("expert widen: stored bytes, scale bytes or output bytes differ from rows and columns")) }
}

/// Widen one 4-bit expert matrix of `rows` by `columns` into little-endian BF16 bytes, row-major.
/// `codes` holds two codes a byte. `scales` holds one byte for each `NVFP4_BLOCK` values of a row.
pub fn widen_nvfp4(codes: &[u8], scales: &[u8], factor: f32, rows: usize, columns: usize,
    order: Option<ExpertRows>, nibbles: NibbleOrder, out: &mut [u8]) -> Result<()> {
    extents(rows, columns, NVFP4_BLOCK, 2, order, codes.len(), scales.len(), out.len())?;
    // The widened weight is a fixed function of the scale byte and the code for one expert.
    let mut table = [[0u8; 2]; 256 * 16];
    let mut usable = [true; 256];
    for scale in 0..=255u8 {
        for code in 0..16u8 {
            let value = nvfp4_weight(code, scale, factor);
            usable[scale as usize] &= value.is_finite();
            table[scale as usize * 16 + code as usize] = f32_to_bf16(value).to_le_bytes();
        }
    }
    if scales.iter().any(|&s| !usable[s as usize]) { return Err(invalid("expert widen: a 4-bit block scale gives a weight that is not finite")); }
    let (code_stride, scale_stride) = (columns / 2, columns / NVFP4_BLOCK);
    let NibbleOrder::LowFirst = nibbles;
    for (row, out) in out.chunks_exact_mut(columns * 2).enumerate() {
        let from = stored_row(order, row, rows);
        let codes = &codes[from * code_stride..(from + 1) * code_stride];
        let scales = &scales[from * scale_stride..(from + 1) * scale_stride];
        for ((codes, &scale), out) in codes.as_chunks::<{ NVFP4_BLOCK / 2 }>().0.iter().zip(scales).zip(out.as_chunks_mut::<{ NVFP4_BLOCK * 2 }>().0) {
            let base = scale as usize * 16;
            for (&byte, out) in codes.iter().zip(out.as_chunks_mut::<4>().0) {
                // The low half of a byte is the even column.
                out[..2].copy_from_slice(&table[base + (byte & 15) as usize]);
                out[2..].copy_from_slice(&table[base + (byte >> 4) as usize]);
            }
        }
    }
    Ok(())
}
/// The BF16 pattern of every 8-bit value under one scale byte, or `None` when a value is not finite.
fn mxfp8_row(scale: u8) -> Option<[[u8; 2]; 256]> {
    let mut row = [[0u8; 2]; 256];
    for value in 0..=255u8 {
        let weight = mxfp8_weight(value, scale);
        // The stored 8-bit form has no infinity. Its one NaN pattern for each sign is refused where it occurs.
        if !weight.is_finite() { if e4m3(value).is_nan() { continue; } return None; }
        row[value as usize] = f32_to_bf16(weight).to_le_bytes();
    }
    Some(row)
}
/// Widen one 8-bit expert matrix of `rows` by `columns` into little-endian BF16 bytes, row-major.
/// `scales` holds one byte for each `MXFP8_BLOCK` values of a row.
pub fn widen_mxfp8(values: &[u8], scales: &[u8], rows: usize, columns: usize,
    order: Option<ExpertRows>, out: &mut [u8]) -> Result<()> {
    extents(rows, columns, MXFP8_BLOCK, 1, order, values.len(), scales.len(), out.len())?;
    if values.iter().any(|&v| v & 0x7f == 0x7f) { return Err(invalid("expert widen: an 8-bit weight is not a number")); }
    let mut tables: Vec<Option<Box<[[u8; 2]; 256]>>> = (0..256).map(|_| None).collect();
    for &scale in scales {
        if tables[scale as usize].is_none() {
            tables[scale as usize] = Some(Box::new(mxfp8_row(scale).ok_or_else(|| invalid("expert widen: an 8-bit block scale gives a weight that is not finite"))?));
        }
    }
    let scale_stride = columns / MXFP8_BLOCK;
    for (row, out) in out.chunks_exact_mut(columns * 2).enumerate() {
        let from = stored_row(order, row, rows);
        let values = &values[from * columns..(from + 1) * columns];
        let scales = &scales[from * scale_stride..(from + 1) * scale_stride];
        for ((values, &scale), out) in values.as_chunks::<MXFP8_BLOCK>().0.iter().zip(scales).zip(out.as_chunks_mut::<{ MXFP8_BLOCK * 2 }>().0) {
            let table = tables[scale as usize].as_ref().ok_or_else(|| invalid("expert widen: scale table is absent"))?;
            for (&value, out) in values.iter().zip(out.as_chunks_mut::<2>().0) { *out = table[value as usize]; }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bf16(bytes: &[u8]) -> Vec<f32> { bytes.as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes(*b))).collect() }

    #[test]
    fn the_factor_rounds_amax_to_bf16_before_the_division() {
        // 0.3 is not a BF16 value. Its BF16 neighbor is 0.30078125 (pattern 0x3e9a).
        let amax = 0.3f32;
        assert_eq!(source_cast(amax), 0.30078125);
        assert_eq!(expert_factor(amax, 2688.0).unwrap(), 0.30078125f32 / 2688.0);
        assert_ne!(expert_factor(amax, 2688.0).unwrap(), amax / 2688.0);
        for (a, d) in [(0.0, 1.0), (-1.0, 1.0), (f32::NAN, 1.0), (1.0, 0.0), (1.0, f32::INFINITY)] { assert!(expert_factor(a, d).is_err()); }
    }
    #[test]
    fn four_bit_codes_use_the_low_half_first_and_one_scale_for_sixteen_values() {
        // One row of 32 columns: two blocks. Scale 0x38 is 1.0 and 0x40 is 2.0 in the 8-bit float form.
        assert_eq!((e4m3(0x38), e4m3(0x40)), (1.0, 2.0));
        let mut codes = vec![0u8; 16];
        codes[0] = 0x72;          // column 0 = code 2 (1.0), column 1 = code 7 (6.0)
        codes[8] = 0x9f;          // column 16 = code 15 (-6.0), column 17 = code 9 (-0.5)
        let mut out = vec![0u8; 64];
        widen_nvfp4(&codes, &[0x38, 0x40], 0.5, 1, 32, None, NibbleOrder::LowFirst, &mut out).unwrap();
        let v = bf16(&out);
        assert_eq!(&v[..2], &[0.5, 3.0]);
        assert_eq!(&v[16..18], &[-6.0, -0.5]);
        assert!(v[2..16].iter().chain(&v[18..]).all(|&x| x == 0.0));
    }
    #[test]
    fn a_four_bit_weight_is_rounded_to_bf16_one_time() {
        // code 3 (1.5) times scale 0x3b (1.375) times a factor with many bits.
        // The source 0.7 is 179/256 after its cast: 179 x 33 / 21 is no whole number, so the product has more than 8 bits.
        let factor = expert_factor(0.7, 2688.0).unwrap();
        let exact = 1.5f32 * 1.375 * factor;
        let mut out = vec![0u8; 32];
        widen_nvfp4(&[3, 0, 0, 0, 0, 0, 0, 0], &[0x3b], factor, 1, 16, None, NibbleOrder::LowFirst, &mut out).unwrap();
        let got = bf16(&out)[0];
        assert_eq!(got, bf16_to_f32(f32_to_bf16(exact)));
        assert_ne!(got, exact);
    }
    #[test]
    fn gate_rows_come_first_and_up_rows_second() {
        // Four stored rows of 16 columns. Stored row r holds code r + 1 in column 0.
        let mut codes = vec![0u8; 4 * 8];
        for r in 0..4 { codes[r * 8] = r as u8 + 1; }
        let mut out = vec![0u8; 4 * 32];
        widen_nvfp4(&codes, &[0x38; 4], 1.0, 4, 16, Some(ExpertRows::Interleaved), NibbleOrder::LowFirst, &mut out).unwrap();
        let first: Vec<f32> = bf16(&out).chunks(16).map(|row| row[0]).collect();
        // Stored rows 0 and 2 are the gate rows. Stored rows 1 and 3 are the up rows.
        assert_eq!(first, [e2m1(1), e2m1(3), e2m1(2), e2m1(4)]);
        let mut plain = vec![0u8; 4 * 32];
        widen_nvfp4(&codes, &[0x38; 4], 1.0, 4, 16, None, NibbleOrder::LowFirst, &mut plain).unwrap();
        assert_eq!(bf16(&plain).chunks(16).map(|row| row[0]).collect::<Vec<_>>(), [e2m1(1), e2m1(2), e2m1(3), e2m1(4)]);
    }
    #[test]
    fn eight_bit_weights_take_a_power_of_two_for_thirty_two_values() {
        // Value 0x38 is 1.0 and 0xb8 is -1.0. Scale 127 is 1, 126 is one half, 130 is 8.
        let mut values = vec![0u8; 64];
        values[0] = 0x38; values[31] = 0xb8; values[32] = 0x3c; values[63] = 0x38;
        let mut out = vec![0u8; 128];
        widen_mxfp8(&values, &[126, 130], 1, 64, None, &mut out).unwrap();
        let v = bf16(&out);
        assert_eq!((v[0], v[31], v[32], v[63]), (0.5, -0.5, 12.0, 8.0));
        // Every 8-bit value is exact in BF16: the widened number is the stored number.
        for value in 0..=255u8 {
            if e4m3(value).is_nan() { continue; }
            let exact = mxfp8_weight(value, 120);
            assert_eq!(bf16_to_f32(f32_to_bf16(exact)), exact);
        }
    }
    #[test]
    fn bad_extents_and_bad_numbers_are_refused() {
        let mut out = vec![0u8; 32];
        assert!(widen_nvfp4(&[0; 8], &[0x38], 1.0, 1, 16, None, NibbleOrder::LowFirst, &mut out[..30]).is_err());
        assert!(widen_nvfp4(&[0; 7], &[0x38], 1.0, 1, 16, None, NibbleOrder::LowFirst, &mut out).is_err());
        assert!(widen_nvfp4(&[0; 8], &[0x38], 1.0, 1, 16, Some(ExpertRows::Interleaved), NibbleOrder::LowFirst, &mut out).is_err());
        // Scale byte 0x7f is not a number in the 8-bit float form.
        assert!(widen_nvfp4(&[0; 8], &[0x7f], 1.0, 1, 16, None, NibbleOrder::LowFirst, &mut out).is_err());
        let mut out = vec![0u8; 64];
        assert!(widen_mxfp8(&[0x7f; 32], &[127], 1, 32, None, &mut out).is_err());
        // Scale 255 is 2 ^ 128: the weight is not finite.
        assert!(widen_mxfp8(&[0x38; 32], &[255], 1, 32, None, &mut out).is_err());
        assert!(widen_mxfp8(&[0x38; 32], &[127], 1, 32, None, &mut out).is_ok());
    }
}
