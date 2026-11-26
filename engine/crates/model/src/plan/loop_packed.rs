//! Form A2 of an expert product: the product reads the stored 4-bit or 8-bit codes and their block
//! scales. No widened matrix is kept. Host arithmetic only: this file is the contract and the host
//! twin. No GPU kernel reads this form yet.
//!
//! # Contract of one product
//!
//! Inputs:
//! - `codes`: `[rows, columns]` stored weights, row-major. 4-bit form: two codes a byte, and the
//!   low half of a byte is the even column. 8-bit form: one value a byte.
//! - `scales`: `[rows, columns / block]`, one byte for each block of a row. A block is 16 columns
//!   in the 4-bit form and 32 columns in the 8-bit form.
//! - 4-bit form only: the factor of the expert, one 32-bit value (`loop_widen::expert_factor`).
//! - `x`: `[columns]`, the 32-bit input row.
//! - The row order: a matrix that holds gate rows and up rows stores gate row `i` at row `2i` and
//!   up row `i` at row `2i + 1`.
//!
//! Output: `[rows]` 32-bit words. Output row `o` reads stored row `r(o)`: `r(o) = o` for a plain
//! matrix; for a gate-and-up matrix the gate rows come first, `r(o) = 2o` below half the rows and
//! `r(o) = 2(o - half) + 1` from half on.
//!
//! `out[o]` is one chain over the columns in rising order from +0.0:
//! `sum = rn(sum + rn(w(r, c) * x[c]))`. Each product is rounded before its add. The sum is not
//! rounded to BF16.
//!
//! The weight `w(r, c)` is the BF16 value that form A1 holds for that place, as a 32-bit value:
//! - 4-bit: `bf16(rn(rn(value(code) * value(scale)) * factor))` (`loop_widen::nvfp4_weight`, then
//!   one round to BF16).
//! - 8-bit: `value(code) * 2 ^ (scale - 127)` (`loop_widen::mxfp8_weight`); it is exact in BF16.
//!
//! So the output is, word for word, the output of the BF16 row product on the widened matrix of
//! form A1. No token ID moves when a product changes from form A1 to this form.
//!
//! # What a kernel can hold in place of the widened matrix
//!
//! In the 4-bit form a weight is a fixed function of its scale byte and its code for one expert:
//! 256 x 16 values. A table of these 4,096 BF16 values for each expert matrix (8,192 bytes), made
//! one time at load with the arithmetic above, keeps the three roundings of a weight out of the
//! product. In the 8-bit form a weight is a fixed function of the scale byte and the value byte
//! for the whole model: 256 x 256 values, or the stored value times a power of two in the product.
use super::{loop_widen::{self, MXFP8_BLOCK, NVFP4_BLOCK}, storage_groups::{ExpertRows, NibbleOrder}};
use crate::{Result, flash::nvfp4::{bf16_to_f32, f32_to_bf16}, invalid};

/// The stored form of the codes of one matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PackedCodes {
    /// Two 4-bit codes a byte, one 8-bit scale for 16 columns, and the factor of the expert.
    Nvfp4 { factor: f32, nibbles: NibbleOrder },
    /// One 8-bit value a byte, and one power-of-two scale for 32 columns.
    Mxfp8,
}
/// One stored expert matrix: the bytes that a product of form A2 reads.
#[derive(Debug, Clone, Copy)]
pub struct PackedMatrix<'a> {
    pub codes: &'a [u8], pub scales: &'a [u8],
    pub rows: usize, pub columns: usize,
    pub form: PackedCodes,
    /// The row order of a matrix that holds gate rows and up rows. None for a plain matrix.
    pub order: Option<ExpertRows>,
}
/// An 8-bit value byte with all these bits set is not a number, for each sign.
const NOT_A_NUMBER: u8 = 0x7f;
/// The 32-bit value of the BF16 weight that form A1 holds.
fn held(weight: f32) -> Option<f32> { weight.is_finite().then(|| bf16_to_f32(f32_to_bf16(weight))) }

impl PackedMatrix<'_> {
    /// Columns that share one scale byte.
    pub fn block(&self) -> usize { match self.form { PackedCodes::Nvfp4 { .. } => NVFP4_BLOCK, PackedCodes::Mxfp8 => MXFP8_BLOCK } }
    /// Code bytes of one row.
    fn code_stride(&self) -> usize { match self.form { PackedCodes::Nvfp4 { .. } => self.columns / 2, PackedCodes::Mxfp8 => self.columns } }
    /// Stored bytes that one product reads: the codes and the block scales of every row.
    pub fn bytes(&self) -> usize { self.codes.len() + self.scales.len() }
    fn check(&self, x: usize, out: usize) -> Result<()> {
        let (rows, columns, block) = (self.rows, self.columns, self.block());
        let ok = rows > 0 && columns > 0 && columns.is_multiple_of(block) && (self.order.is_none() || rows.is_multiple_of(2))
            && rows.checked_mul(self.code_stride()) == Some(self.codes.len()) && rows.checked_mul(columns / block) == Some(self.scales.len())
            && x == columns && out == rows;
        if ok { Ok(()) } else { Err(invalid("packed expert product: stored bytes, scale bytes, input or output differ from rows and columns")) }
    }
    /// The weights of every scale byte, by code or value: `[scale byte][code]`. A scale byte that gives a weight
    /// that is not finite is not usable; form A1 refuses a matrix that uses such a byte, and so does this form.
    fn table(&self) -> (Vec<f32>, [bool; 256], usize) {
        let kinds = match self.form { PackedCodes::Nvfp4 { .. } => 16, PackedCodes::Mxfp8 => 256 };
        let (mut table, mut usable) = (vec![0.0f32; 256 * kinds], [true; 256]);
        for scale in 0..=255u8 {
            for kind in 0..kinds {
                let weight = match self.form {
                    PackedCodes::Nvfp4 { factor, .. } => loop_widen::nvfp4_weight(kind as u8, scale, factor),
                    // The 8-bit form has one pattern for each sign that is not a number. It is refused where it occurs.
                    PackedCodes::Mxfp8 if kind as u8 & NOT_A_NUMBER == NOT_A_NUMBER => continue,
                    PackedCodes::Mxfp8 => loop_widen::mxfp8_weight(kind as u8, scale),
                };
                match held(weight) { Some(w) => table[scale as usize * kinds + kind] = w, None => usable[scale as usize] = false }
            }
        }
        (table, usable, kinds)
    }
    /// The product of this matrix with the input row `x`, by the contract of this file.
    pub fn product(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        self.check(x.len(), out.len())?;
        if self.form == PackedCodes::Mxfp8 && self.codes.iter().any(|&v| v & NOT_A_NUMBER == NOT_A_NUMBER) { return Err(invalid("packed expert product: an 8-bit weight is not a number")); }
        let (table, usable, kinds) = self.table();
        if self.scales.iter().any(|&s| !usable[s as usize]) { return Err(invalid("packed expert product: a block scale gives a weight that is not finite")); }
        let (block, code_stride, scale_stride) = (self.block(), self.code_stride(), self.columns / self.block());
        for (o, out) in out.iter_mut().enumerate() {
            let r = loop_widen::stored_row(self.order, o, self.rows);
            let codes = &self.codes[r * code_stride..(r + 1) * code_stride];
            let scales = &self.scales[r * scale_stride..(r + 1) * scale_stride];
            let mut sum = 0.0f32;
            for (b, &scale) in scales.iter().enumerate() {
                let (base, x) = (scale as usize * kinds, &x[b * block..(b + 1) * block]);
                match self.form {
                    PackedCodes::Nvfp4 { nibbles: NibbleOrder::LowFirst, .. } => {
                        // The low half of a byte is the even column.
                        for (&byte, x) in codes[b * block / 2..(b + 1) * block / 2].iter().zip(x.as_chunks::<2>().0) {
                            sum += table[base + (byte & 15) as usize] * x[0];
                            sum += table[base + (byte >> 4) as usize] * x[1];
                        }
                    }
                    PackedCodes::Mxfp8 => { for (&byte, x) in codes[b * block..(b + 1) * block].iter().zip(x) { sum += table[base + byte as usize] * x; } }
                }
            }
            *out = sum;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::storage_groups::{GATE_UP_ROWS, NVFP4_NIBBLES};

    /// Stored bytes that differ from place to place. A scale byte stays in a range that gives finite weights.
    fn made(count: usize, salt: usize, low: u8, span: u8) -> Vec<u8> { (0..count).map(|i| low + (((i + salt).wrapping_mul(2_654_435_761) >> 9) as u8) % span).collect() }
    fn input(columns: usize, salt: u32) -> Vec<f32> { (0..columns as u32).map(|k| ((k + salt).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0).collect() }
    /// The chain of the BF16 row product on a widened matrix of form A1.
    fn widened_product(bf16: &[u8], columns: usize, x: &[f32]) -> Vec<u32> {
        bf16.chunks(columns * 2).map(|row| row.as_chunks::<2>().0.iter().zip(x).fold(0.0f32, |sum, (w, x)| sum + bf16_to_f32(u16::from_le_bytes(*w)) * x).to_bits()).collect()
    }
    fn bits(words: &[f32]) -> Vec<u32> { words.iter().map(|w| w.to_bits()).collect() }

    #[test]
    fn a_four_bit_product_equals_the_product_on_the_widened_matrix() {
        let (rows, columns, factor) = (6, 48, loop_widen::expert_factor(0.7, 2688.0).unwrap());
        for order in [None, Some(GATE_UP_ROWS)] {
            // Scale bytes from 0x28 to 0x47: block scales from 0.25 to under 4.
            let (codes, scales) = (made(rows * columns / 2, 1, 0, 255), made(rows * columns / NVFP4_BLOCK, 2, 0x28, 0x20));
            let mut widened = vec![0u8; rows * columns * 2];
            loop_widen::widen_nvfp4(&codes, &scales, factor, rows, columns, order, NVFP4_NIBBLES, &mut widened).unwrap();
            let packed = PackedMatrix { codes: &codes, scales: &scales, rows, columns, form: PackedCodes::Nvfp4 { factor, nibbles: NVFP4_NIBBLES }, order };
            assert_eq!(packed.bytes(), rows * columns / 2 + rows * columns / 16);
            for salt in [3, 4] {
                let (x, mut out) = (input(columns, salt), vec![f32::NAN; rows]);
                packed.product(&x, &mut out).unwrap();
                assert_eq!(bits(&out), widened_product(&widened, columns, &x), "{order:?}");
            }
        }
    }
    #[test]
    fn an_eight_bit_product_equals_the_product_on_the_widened_matrix() {
        let (rows, columns) = (4, 64);
        for order in [None, Some(GATE_UP_ROWS)] {
            // Every value byte but the two patterns that are not numbers; scales from 2^-16 to 2^-9.
            let codes = made(rows * columns, 5, 0, 255).into_iter().map(|b| if b & 0x7f == 0x7f { b ^ 1 } else { b }).collect::<Vec<_>>();
            let scales = made(rows * columns / MXFP8_BLOCK, 6, 111, 8);
            let mut widened = vec![0u8; rows * columns * 2];
            loop_widen::widen_mxfp8(&codes, &scales, rows, columns, order, &mut widened).unwrap();
            let packed = PackedMatrix { codes: &codes, scales: &scales, rows, columns, form: PackedCodes::Mxfp8, order };
            assert_eq!(packed.bytes(), rows * columns + rows * columns / 32);
            let (x, mut out) = (input(columns, 7), vec![f32::NAN; rows]);
            packed.product(&x, &mut out).unwrap();
            assert_eq!(bits(&out), widened_product(&widened, columns, &x), "{order:?}");
        }
    }
    #[test]
    fn bad_extents_and_a_scale_that_is_not_finite_are_refused() {
        let (codes, scales, x) = (vec![0u8; 16], vec![0x38u8; 2], vec![1.0f32; 32]);
        let good = PackedMatrix { codes: &codes, scales: &scales, rows: 1, columns: 32, form: PackedCodes::Nvfp4 { factor: 0.5, nibbles: NVFP4_NIBBLES }, order: None };
        let mut out = [0.0f32; 1];
        good.product(&x, &mut out).unwrap();
        assert!(good.product(&x[..31], &mut out).is_err());
        assert!(PackedMatrix { rows: 2, ..good }.product(&x, &mut [0.0; 2]).is_err());
        assert!(PackedMatrix { order: Some(GATE_UP_ROWS), ..good }.product(&x, &mut out).is_err());
        assert!(PackedMatrix { form: PackedCodes::Mxfp8, ..good }.product(&x, &mut out).is_err());
        // The 8-bit scale byte 255 is above the largest finite power of two.
        let (values, scale) = (vec![0x38u8; 32], vec![255u8]);
        let high = PackedMatrix { codes: &values, scales: &scale, rows: 1, columns: 32, form: PackedCodes::Mxfp8, order: None };
        assert!(high.product(&x, &mut out).unwrap_err().to_string().contains("not finite"));
        let (values, scale) = (vec![0xffu8; 32], vec![120u8]);
        let no_number = PackedMatrix { codes: &values, scales: &scale, ..high };
        assert!(no_number.product(&x, &mut out).unwrap_err().to_string().contains("not a number"));
    }
}
