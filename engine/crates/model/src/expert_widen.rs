//! Expert weight decode for two stored formats, widened to BF16 at load.
//! This file is the one seam where expert numbers are made.
//!
//! Source (read for the math, no code copied): the public `transformers`
//! release v5.24.0, `core_model_loading.py`. The decode step: lines 772-816.
//! The cast of each stored tensor to the model type before that step:
//! lines 1909-1955 and the note at line 792.
//!
//! Two stored formats, told apart by the tensors that are present:
//! - Packed 4-bit: one byte holds two E2M1 codes. One E4M3 scale serves 16
//!   values that follow each other in one row. A second-level scale for the
//!   whole expert is `amax / denominator`.
//! - 8-bit float: one E4M3 value for each weight. One exponent byte serves 32
//!   values that follow each other in one row: the scale is `2^(byte - 127)`.
//!
//! The widened weight (the form with BF16 weights and BF16 inputs):
//! - Packed 4-bit: `round_bf16((code value * block scale) * factor)`. The
//!   first product is exact in FP32. [`expert_factor`] gives the factor.
//! - 8-bit float: `round_bf16(value * 2^(byte - 127))`.
//!
//! Each layout fact is one named item here: [`LOW_NIBBLE_IS_EVEN_COLUMN`] and
//! [`SECOND_SCALE_ROUNDED_FIRST`]. The ignored test `capped_moe_trace` runs
//! real stored experts against a saved reference trace. It fails when one
//! of the two is changed.
//!
//! A later form that keeps the stored precision replaces [`widen`] and the
//! two products that read its result. No other file decodes expert codes.
use crate::{Result, invalid, flash::{mxfp8, nvfp4::{self, bf16_to_f32, e2m1, e4m3}}, norm_cap::round_bf16};

/// In a packed byte the low 4 bits are the value of the even column and the
/// high 4 bits are the value of the next column. Source line 801.
pub const LOW_NIBBLE_IS_EVEN_COLUMN: bool = true;
/// The source loader casts each stored tensor to the BF16 model type before
/// the decode step. So `amax` and the denominator are rounded to BF16 before
/// the divide. The decode step alone, on the stored FP32 `amax`, gives other weights.
pub const SECOND_SCALE_ROUNDED_FIRST: bool = true;
/// Values that share one block scale in the packed 4-bit format.
pub const PACKED4_BLOCK: usize = nvfp4::BLOCK;
/// Values that share one exponent byte in the 8-bit float format.
pub const FLOAT8_BLOCK: usize = mxfp8::BLOCK;

/// The stored bytes of one matrix of one expert, row-major.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stored<'a> {
    /// `codes`: `rows * columns / 2` bytes. `scales`: `rows * columns / 16` E4M3 bytes.
    Packed4 { codes: &'a [u8], scales: &'a [u8], amax: f32, denominator: f32 },
    /// `values`: `rows * columns` E4M3 bytes. `scales`: `rows * columns / 32` exponent bytes.
    Float8 { values: &'a [u8], scales: &'a [u8] },
}
/// One stored matrix with its logical shape. Blocks do not cross rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StoredMatrix<'a> { pub rows: usize, pub columns: usize, pub data: Stored<'a> }

/// The stored tensors of all experts of one layer for one matrix kind.
/// The first axis of each tensor is the expert number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StoredStack<'a> {
    Packed4 { codes: &'a [u8], scales: &'a [u8], amax: &'a [f32], denominator: f32 },
    Float8 { values: &'a [u8], scales: &'a [u8] },
}
impl<'a> StoredStack<'a> {
    /// Slice one expert out of the stack. Sizes come from the caller's shapes.
    pub fn expert(&self, experts: usize, expert: usize, rows: usize, columns: usize) -> Result<StoredMatrix<'a>> {
        let cells = rows.checked_mul(columns).filter(|&n| n > 0).ok_or_else(|| invalid("expert matrix shape is empty or too large"))?;
        if expert >= experts { return Err(invalid("expert number is outside the stack")); }
        let part = |bytes: &'a [u8], each: usize, what: &str| -> Result<&'a [u8]> {
            if each.checked_mul(experts) != Some(bytes.len()) { return Err(invalid(format!("{what} length does not match the expert stack"))); }
            Ok(&bytes[expert * each..(expert + 1) * each])
        };
        let data = match *self {
            Self::Packed4 { codes, scales, amax, denominator } => {
                if !columns.is_multiple_of(PACKED4_BLOCK) || amax.len() != experts { return Err(invalid("packed 4-bit stack shape is invalid")); }
                Stored::Packed4 { codes: part(codes, cells / 2, "code")?, scales: part(scales, cells / PACKED4_BLOCK, "block scale")?, amax: amax[expert], denominator }
            }
            Self::Float8 { values, scales } => {
                if !columns.is_multiple_of(FLOAT8_BLOCK) { return Err(invalid("8-bit float stack shape is invalid")); }
                Stored::Float8 { values: part(values, cells, "value")?, scales: part(scales, cells / FLOAT8_BLOCK, "exponent")? }
            }
        };
        Ok(StoredMatrix { rows, columns, data })
    }
}

/// The 4-bit code of one column of a packed row.
#[allow(clippy::manual_is_multiple_of)] // The remainder step stays as written. The lint must not change a step.
pub fn packed_code(codes: &[u8], column: usize) -> u8 {
    let byte = codes[column / 2];
    if (column % 2 == 0) == LOW_NIBBLE_IS_EVEN_COLUMN { byte & 15 } else { byte >> 4 }
}

/// The second-level scale of one expert: `amax / denominator` in FP32.
/// With [`SECOND_SCALE_ROUNDED_FIRST`] both operands are rounded to BF16 first.
pub fn expert_factor(amax: f32, denominator: f32) -> Result<f32> {
    let cast = |v: f32| if SECOND_SCALE_ROUNDED_FIRST { round_bf16(v) } else { v };
    let factor = cast(amax) / cast(denominator);
    if !amax.is_finite() || !denominator.is_finite() || !factor.is_finite() { return Err(invalid("expert second-level scale is not finite")); }
    Ok(factor)
}

/// The FP32 value of one packed 4-bit weight before the BF16 rounding.
/// `code * scale` is exact in FP32. Only the product with the factor rounds.
pub fn packed4_value(code: u8, scale: u8, factor: f32) -> f32 { (e2m1(code) * e4m3(scale)) * factor }

/// Widen one stored matrix to BF16 bits, row-major `[rows, columns]`.
/// A value or a scale that is not finite is refused.
pub fn widen(matrix: &StoredMatrix<'_>) -> Result<Vec<u16>> {
    let (rows, columns) = (matrix.rows, matrix.columns);
    let cells = rows.checked_mul(columns).filter(|&n| n > 0).ok_or_else(|| invalid("expert matrix shape is empty or too large"))?;
    let mut output = vec![0u16; cells];
    match matrix.data {
        Stored::Packed4 { codes, scales, amax, denominator } => {
            if !columns.is_multiple_of(PACKED4_BLOCK) || codes.len() != cells / 2 || scales.len() != cells / PACKED4_BLOCK {
                return Err(invalid("packed 4-bit lengths do not match the matrix shape"));
            }
            let factor = expert_factor(amax, denominator)?;
            let (row_bytes, row_scales) = (columns / 2, columns / PACKED4_BLOCK);
            for row in 0..rows {
                let (codes, scales) = (&codes[row * row_bytes..(row + 1) * row_bytes], &scales[row * row_scales..(row + 1) * row_scales]);
                for column in 0..columns {
                    let value = packed4_value(packed_code(codes, column), scales[column / PACKED4_BLOCK], factor);
                    if !value.is_finite() { return Err(invalid("packed 4-bit weight is not finite")); }
                    output[row * columns + column] = bf16_bits(value);
                }
            }
        }
        Stored::Float8 { values, scales } => {
            if !columns.is_multiple_of(FLOAT8_BLOCK) || values.len() != cells || scales.len() != cells / FLOAT8_BLOCK {
                return Err(invalid("8-bit float lengths do not match the matrix shape"));
            }
            // A row holds a whole number of blocks, so the flat block index stays inside its row.
            for (i, out) in output.iter_mut().enumerate() {
                let value = mxfp8::dequant_value(values[i], scales[i / FLOAT8_BLOCK]);
                if !value.is_finite() { return Err(invalid("8-bit float weight is not finite")); }
                *out = bf16_bits(value);
            }
        }
    }
    if output.iter().any(|&bits| bits & 0x7f80 == 0x7f80) { return Err(invalid("widened expert weight exceeds the BF16 range")); }
    Ok(output)
}

/// The 16 stored bits of the nearest BF16 value.
fn bf16_bits(value: f32) -> u16 { (round_bf16(value).to_bits() >> 16) as u16 }

/// Widen and return FP32 copies of the BF16 values. No value changes.
pub fn widen_values(matrix: &StoredMatrix<'_>) -> Result<Vec<f32>> {
    Ok(widen(matrix)?.into_iter().map(bf16_to_f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bf16(v: f32) -> u16 { bf16_bits(v) }
    #[test]
    fn packed4_block_hand_values() {
        // One row of 16 columns: 8 code bytes and one scale byte.
        // Scale byte 0x40: sign 0, exponent field 8, fraction 0, so 2^(8-7) = 2.0.
        // amax 0.75 and denominator 1.5 are BF16 values. Factor = 0.5 exactly.
        // So each weight = code value * 2.0 * 0.5 = the code value.
        // Byte 0x21: low 4 bits 1 = +0.5 (column 0), high 4 bits 2 = +1.0 (column 1).
        // Byte 0xF7: low 7 = +6.0 (column 2), high 15 = -6.0 (column 3).
        // Byte 0x93: low 3 = +1.5, high 9 = -0.5. Byte 0x08: low 8 = -0.0, high 0 = +0.0.
        // Byte 0x54: low 4 = +2.0, high 5 = +3.0. Byte 0x6C: low 12 = -2.0, high 6 = +4.0.
        // Byte 0xAD: low 13 = -3.0, high 10 = -1.0. Byte 0xEB: low 11 = -1.5, high 14 = -4.0.
        let codes = [0x21, 0xF7, 0x93, 0x08, 0x54, 0x6C, 0xAD, 0xEB];
        let expected = [0.5, 1.0, 6.0, -6.0, 1.5, -0.5, -0.0, 0.0, 2.0, 3.0, -2.0, 4.0, -3.0, -1.0, -1.5, -4.0f32];
        let matrix = StoredMatrix { rows: 1, columns: 16, data: Stored::Packed4 { codes: &codes, scales: &[0x40], amax: 0.75, denominator: 1.5 } };
        let bits = widen(&matrix).unwrap();
        assert_eq!(bits, expected.map(bf16));
        // Signed zeros keep their sign.
        assert_eq!((bits[6], bits[7]), (0x8000, 0x0000));
        assert_eq!(widen_values(&matrix).unwrap(), expected);
        // The named constant decides which half of a byte is the even column.
        const { assert!(LOW_NIBBLE_IS_EVEN_COLUMN) };
        assert_eq!((packed_code(&[0x21], 0), packed_code(&[0x21], 1)), (1, 2));
    }
    #[test]
    fn packed4_blocks_follow_their_row_and_the_weight_rounds_once() {
        // 2 rows of 32 columns. Each row has 2 scales. All codes are 7 (+6.0) or 15 (-6.0).
        // Scale bytes: 0x38 = 1.0, 0x3C = 1.5, 0x30 = 0.5, 0x7E = 448 (the largest E4M3 value).
        let codes = [0xF7u8; 32];
        let scales = [0x38, 0x3C, 0x30, 0x7E];
        // amax 0.7 is not a BF16 value: it rounds to 179/256 = 0.69921875 first. Denominator 2688 is exact.
        let (amax, denominator) = (0.7f32, 2688.0f32);
        let factor = expert_factor(amax, denominator).unwrap();
        assert_eq!(factor, 0.699_218_75f32 / 2688.0);
        assert_ne!(factor, 0.7f32 / 2688.0);
        let matrix = StoredMatrix { rows: 2, columns: 32, data: Stored::Packed4 { codes: &codes, scales: &scales, amax, denominator } };
        let bits = widen(&matrix).unwrap();
        for (block, scale) in [1.0f32, 1.5, 0.5, 448.0].into_iter().enumerate() {
            for column in 0..16 {
                let sign = if column % 2 == 0 { 1.0 } else { -1.0 };
                assert_eq!(bits[block * 16 + column], bf16((sign * 6.0 * scale) * factor), "block {block} column {column}");
            }
        }
        // The largest code times the largest scale times amax / 2688 gives amax back: 6 * 448 = 2688.
        assert_eq!(bf16_to_f32(bits[48]), 0.699_218_75);
        // One rounding only: 9.0 * factor = 179 * 3/7 * 2^-15 has no end in binary, and BF16 keeps 8 bits.
        let exact = 9.0f32 * factor;
        assert_ne!(bf16_to_f32(bits[16]), exact);
        assert_eq!(bits[16], bf16(exact));
    }
    #[test]
    fn float8_hand_values() {
        // One row of 32 columns, then a second row with another exponent byte.
        // Exponent byte 127 is 2^0. Byte 113 is 2^-14. Byte 130 is 2^3.
        // Value byte 0x38 = 1.0. 0x3C = 1.5. 0xB8 = -1.0. 0x7E = 448. 0x01 = 2^-9. 0x00 = 0. 0x80 = -0.
        let mut values = [0x38u8; 64];
        values[..7].copy_from_slice(&[0x38, 0x3C, 0xB8, 0x7E, 0x01, 0x00, 0x80]);
        values[32..35].copy_from_slice(&[0x3C, 0xB8, 0x7E]);
        let matrix = StoredMatrix { rows: 2, columns: 32, data: Stored::Float8 { values: &values, scales: &[113, 130] } };
        let got = widen_values(&matrix).unwrap();
        let low = 2.0f32.powi(-14);
        assert_eq!(got[..7], [low, 1.5 * low, -low, 448.0 * low, 2.0f32.powi(-23), 0.0, -0.0]);
        assert_eq!(got[6].to_bits(), (-0.0f32).to_bits());
        assert_eq!(got[7..32], [low; 25]);
        assert_eq!(got[32..35], [12.0, -8.0, 3584.0]);
        assert_eq!(got[35..], [8.0; 29]);
        // The 8-bit float widen is exact: 4 significant bits fit in the 8 of BF16.
        let unit = StoredMatrix { rows: 1, columns: 32, data: Stored::Float8 { values: &values[..32], scales: &[127] } };
        assert_eq!(widen_values(&unit).unwrap()[..4], [1.0, 1.5, -1.0, 448.0]);
    }
    #[test]
    fn stack_slices_one_expert() {
        // 3 experts, each 1 row of 16 columns. Expert 1 has its own codes, scale and amax.
        let codes: Vec<u8> = [[0x11u8; 8], [0x22; 8], [0x33; 8]].concat();
        let stack = StoredStack::Packed4 { codes: &codes, scales: &[0x38, 0x40, 0x38], amax: &[1.0, 3.0, 1.0], denominator: 1.5 };
        let one = stack.expert(3, 1, 1, 16).unwrap();
        // Code 2 = 1.0. Scale 2.0. Factor 3.0 / 1.5 = 2.0. Weight 4.0.
        assert_eq!(widen_values(&one).unwrap(), [4.0; 16]);
        assert_eq!(widen_values(&stack.expert(3, 2, 1, 16).unwrap()).unwrap(), [1.0; 16]);
        assert!(stack.expert(3, 3, 1, 16).is_err());
        assert!(stack.expert(2, 1, 1, 16).is_err());
        assert!(stack.expert(3, 1, 1, 24).is_err());
        let values = [0x38u8; 96];
        let wide = StoredStack::Float8 { values: &values, scales: &[127, 128, 129] };
        assert_eq!(widen_values(&wide.expert(3, 2, 1, 32).unwrap()).unwrap(), [4.0; 32]);
        assert!(wide.expert(3, 0, 1, 16).is_err());
    }
    #[test]
    fn bad_bytes_and_shapes_are_refused() {
        let packed = |codes: &'static [u8], scales: &'static [u8], amax: f32, denominator: f32, rows, columns|
            widen(&StoredMatrix { rows, columns, data: Stored::Packed4 { codes, scales, amax, denominator } });
        assert!(packed(&[0x11; 8], &[0x38], 1.0, 1.0, 1, 16).is_ok());
        assert!(packed(&[0x11; 8], &[0x38], 1.0, 1.0, 1, 8).is_err());
        assert!(packed(&[0x11; 7], &[0x38], 1.0, 1.0, 1, 16).is_err());
        assert!(packed(&[0x11; 8], &[0x38, 0x38], 1.0, 1.0, 1, 16).is_err());
        assert!(packed(&[0x11; 8], &[0x38], 1.0, 1.0, 0, 16).is_err());
        // Scale byte 0x7F is the not-a-number code of E4M3. A zero denominator has no factor.
        assert!(packed(&[0x11; 8], &[0x7F], 1.0, 1.0, 1, 16).is_err());
        assert!(packed(&[0x11; 8], &[0x38], 1.0, 0.0, 1, 16).is_err());
        assert!(packed(&[0x11; 8], &[0x38], f32::NAN, 1.0, 1, 16).is_err());
        let float8 = |values: &'static [u8], scales: &'static [u8], columns|
            widen(&StoredMatrix { rows: 1, columns, data: Stored::Float8 { values, scales } });
        assert!(float8(&[0x38; 32], &[127], 32).is_ok());
        assert!(float8(&[0x38; 32], &[127], 16).is_err());
        assert!(float8(&[0x38; 31], &[127], 32).is_err());
        assert!(float8(&[0x7F; 32], &[127], 32).is_err());
        // Exponent byte 255 has no power of two here. A product above the BF16 range is refused.
        assert!(float8(&[0x38; 32], &[255], 32).is_err());
        assert!(float8(&[0x7E; 32], &[254], 32).is_err());
    }
}
