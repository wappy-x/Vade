//! The unit of one packed expert matrix: the bytes that a product of form A2 reads on a device (`loop_packed` is
//! the contract of that product), and a host twin that reads a unit the way the device kernel does.
//!
//! A unit is the stored codes, then the stored block scales, then a weight table. Each part starts on a multiple of
//! `PART` bytes and the unit ends on one; the fill bytes are zero. The codes and the scales are stored in tiles: for
//! each block of `ROW_BLOCK` rows, the bytes of one stage (`STAGE` columns) of its rows, one row after the other, then
//! the next stage, so that a device block reads each stage as one run (`packed_product::tiled` on the device side).
//! The rows are rounded up to a whole block and the columns to a whole stage (two in the 8-bit form); the added
//! bytes are zero. The device takes only shapes of whole stages (`packed_product::shape_admitted`); the others keep
//! a layout of the same bytes on both sides, for the load plan and the host twin. The stored bytes are not
//! changed and the rows keep their stored order, so a matrix of gate rows and up rows stays mixed in the unit; the
//! product gives the gate rows first.
//! - 4-bit form: the table is `[scale byte][code]`, 256 x 16 BF16 patterns, little-endian. Each is the weight that
//!   form A1 holds for that scale byte and code of this expert (`loop_widen::nvfp4_weight`, one round to BF16). A
//!   weight of the product is one table read.
//! - 8-bit form: the table is the 256 32-bit values of a value byte, little-endian. A weight of the product is the
//!   value times `2 ^ (scale - 127)`, rounded to BF16: the arithmetic of `loop_widen::mxfp8_weight` and the round
//!   of form A1.
//!
//! So a weight of a unit is, bit for bit, the BF16 value that form A1 holds, and the product is the product of
//! `loop_packed`. A unit is refused where form A1 refuses the matrix: a block scale that gives a weight that is not
//! finite, and an 8-bit weight that is not a number.
use super::{loop_load::ExpertCodes, loop_widen::{self, MXFP8_BLOCK, NVFP4_BLOCK}, storage_groups::ExpertRows};
use crate::{Result, flash::nvfp4::{bf16_to_f32, e4m3, f32_to_bf16}, invalid};

/// Every part of a unit starts on a multiple of this many bytes.
pub const PART: usize = 16;
/// Rows of one tile, and columns of one stage: the rows and the columns that one device block reads in one stage.
pub const ROW_BLOCK: usize = 32;
pub const STAGE: usize = 64;
/// Scale bytes of one row in one tile: one stage of 4-bit scales, or two stages of 8-bit scales.
const SCALE_PIECE: usize = 4;
/// Bytes of the weight table: 256 x 16 BF16 patterns, or 256 32-bit values.
pub const NVFP4_TABLE: usize = 256 * 16 * 2;
pub const MXFP8_TABLE: usize = 256 * 4;
/// An 8-bit value byte with all these bits set is not a number, for each sign.
const NOT_A_NUMBER: u8 = 0x7f;

/// Byte offsets inside a unit. The codes start at 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitLayout { pub scales: usize, pub table: usize, pub bytes: usize }
/// One packed matrix: its stored form, its stored rows and columns, and its row order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitShape {
    pub codes: ExpertCodes, pub rows: usize, pub columns: usize,
    /// The row order of a matrix that holds gate rows and up rows. None for a plain matrix.
    pub order: Option<ExpertRows>,
}
fn pad(bytes: usize) -> Option<usize> { bytes.checked_add(PART - 1).map(|n| n / PART * PART) }
/// The place of piece `piece` (of `width` bytes) of stored row `row` in a part stored in tiles, where a row has
/// `per_row` pieces: the pieces of the `ROW_BLOCK` rows of a block, one piece after the other.
fn tiled(row: usize, piece: usize, width: usize, per_row: usize) -> usize { ((row / ROW_BLOCK) * per_row + piece) * ROW_BLOCK * width + (row % ROW_BLOCK) * width }
/// `2 ^ (scale - 127)` from its bits, as the device kernel makes it. Byte 0 is below the normal range.
fn power(scale: u8) -> f32 { f32::from_bits(if scale == 0 { 1 << 22 } else { u32::from(scale) << 23 }) }
/// The 32-bit value of a value rounded to BF16.
fn held(value: f32) -> f32 { bf16_to_f32(f32_to_bf16(value)) }

impl UnitShape {
    /// Columns that share one scale byte.
    pub fn block(&self) -> usize { match self.codes { ExpertCodes::Nvfp4 => NVFP4_BLOCK, ExpertCodes::Mxfp8 => MXFP8_BLOCK } }
    /// Code bytes and scale bytes of one stored row.
    pub fn code_stride(&self) -> usize { match self.codes { ExpertCodes::Nvfp4 => self.columns / 2, ExpertCodes::Mxfp8 => self.columns } }
    pub fn scale_stride(&self) -> usize { self.columns / self.block() }
    pub fn table_bytes(&self) -> usize { match self.codes { ExpertCodes::Nvfp4 => NVFP4_TABLE, ExpertCodes::Mxfp8 => MXFP8_TABLE } }
    /// Code bytes of one row in one tile (one stage), and the columns of one scale piece.
    fn code_piece(&self) -> usize { match self.codes { ExpertCodes::Nvfp4 => STAGE / 2, ExpertCodes::Mxfp8 => STAGE } }
    fn scale_columns(&self) -> usize { match self.codes { ExpertCodes::Nvfp4 => STAGE, ExpertCodes::Mxfp8 => 2 * STAGE } }
    /// Code pieces and scale pieces of one row: the columns rounded up to whole scale pieces.
    fn pieces(&self) -> (usize, usize) {
        let scale_pieces = self.columns.div_ceil(self.scale_columns());
        (scale_pieces * (self.scale_columns() / STAGE), scale_pieces)
    }
    /// The place in the unit of byte `byte` of the codes of stored row `row`, and of byte `byte` of its scales
    /// (from the start of the scales). The same places as `packed_product::{code_copy, scale_copy}` on the device.
    pub fn code_at(&self, row: usize, byte: usize) -> usize { let w = self.code_piece(); tiled(row, byte / w, w, self.pieces().0) + byte % w }
    pub fn scale_at(&self, row: usize, byte: usize) -> usize { tiled(row, byte / SCALE_PIECE, SCALE_PIECE, self.pieces().1) + byte % SCALE_PIECE }
    /// The layout of the unit. A shape that is not whole blocks, or mixed rows with an odd row count, is refused.
    pub fn layout(&self) -> Result<UnitLayout> {
        let (rows, columns) = (self.rows, self.columns);
        if rows == 0 || columns == 0 || !columns.is_multiple_of(self.block()) || (self.order.is_some() && !rows.is_multiple_of(2)) {
            return Err(invalid("packed unit: the rows and columns are not whole blocks of a stored expert matrix"));
        }
        let size = || -> Option<UnitLayout> {
            let (tiled_rows, (code_pieces, scale_pieces)) = (rows.checked_next_multiple_of(ROW_BLOCK)?, self.pieces());
            let scales = pad(tiled_rows.checked_mul(code_pieces.checked_mul(self.code_piece())?)?)?;
            let table = scales.checked_add(pad(tiled_rows.checked_mul(scale_pieces.checked_mul(SCALE_PIECE)?)?)?)?;
            Some(UnitLayout { scales, table, bytes: table.checked_add(pad(self.table_bytes())?)? })
        };
        size().ok_or_else(|| invalid("packed unit: the bytes of a unit exceed the address space"))
    }
    /// Make the unit of one matrix from its stored codes and block scales. `factor` is the factor of the expert
    /// in the 4-bit form (`loop_widen::expert_factor`); the 8-bit form has none.
    pub fn fill(&self, codes: &[u8], scales: &[u8], factor: Option<f32>, out: &mut [u8]) -> Result<()> {
        let layout = self.layout()?;
        if self.rows.checked_mul(self.code_stride()) != Some(codes.len()) || self.rows.checked_mul(self.scale_stride()) != Some(scales.len()) || out.len() != layout.bytes {
            return Err(invalid("packed unit: stored bytes, scale bytes or unit bytes differ from rows and columns"));
        }
        out.fill(0);
        let table = &mut out[layout.table..layout.table + self.table_bytes()];
        let mut usable = [true; 256];
        match (self.codes, factor) {
            (ExpertCodes::Nvfp4, Some(factor)) => {
                for scale in 0..=255u8 {
                    for code in 0..16u8 {
                        let value = loop_widen::nvfp4_weight(code, scale, factor);
                        usable[scale as usize] &= value.is_finite();
                        let at = (scale as usize * 16 + code as usize) * 2;
                        table[at..at + 2].copy_from_slice(&f32_to_bf16(value).to_le_bytes());
                    }
                }
                if scales.iter().any(|&s| !usable[s as usize]) { return Err(invalid("packed unit: a 4-bit block scale gives a weight that is not finite")); }
                // Row `0xff` is E4M3 `-NaN` (always refused above). Store the expert factor tag there so SM121a
                // tensor-core kernels decode finite NVFP4 weights directly in registers without an 8 KB LUT copy.
                let tag = 0xff * 32;
                table[tag..tag + 4].copy_from_slice(&0x7fc0_4641u32.to_le_bytes());
                table[tag + 4..tag + 8].copy_from_slice(&0x7fc0_4354u32.to_le_bytes());
                table[tag + 8..tag + 12].copy_from_slice(&factor.to_bits().to_le_bytes());
            }
            (ExpertCodes::Mxfp8, None) => {
                if codes.iter().any(|&v| v & NOT_A_NUMBER == NOT_A_NUMBER) { return Err(invalid("packed unit: an 8-bit weight is not a number")); }
                for value in 0..=255u8 {
                    let number = if value & NOT_A_NUMBER == NOT_A_NUMBER { 0.0 } else { e4m3(value) };
                    table[value as usize * 4..value as usize * 4 + 4].copy_from_slice(&number.to_le_bytes());
                    if value & NOT_A_NUMBER != NOT_A_NUMBER { for scale in 0..=255u8 { usable[scale as usize] &= loop_widen::mxfp8_weight(value, scale).is_finite(); } }
                }
                if scales.iter().any(|&s| !usable[s as usize]) { return Err(invalid("packed unit: an 8-bit block scale gives a weight that is not finite")); }
            }
            _ => return Err(invalid("packed unit: the factor given differs from the stored form of the matrix")),
        }
        let (w, cs, ss) = (self.code_piece(), self.code_stride(), self.scale_stride());
        for row in 0..self.rows {
            for byte in (0..cs).step_by(w) { let n = w.min(cs - byte); out[self.code_at(row, byte)..][..n].copy_from_slice(&codes[row * cs + byte..][..n]); }
            for byte in (0..ss).step_by(SCALE_PIECE) { let n = SCALE_PIECE.min(ss - byte); out[layout.scales + self.scale_at(row, byte)..][..n].copy_from_slice(&scales[row * ss + byte..][..n]); }
        }
        Ok(())
    }
    /// The weight of stored row `row` and column `column`, read from a unit the way the device kernel reads it.
    fn weight(&self, unit: &[u8], layout: UnitLayout, row: usize, column: usize) -> f32 {
        let scale = unit[layout.scales + self.scale_at(row, column / self.block())];
        match self.codes {
            ExpertCodes::Nvfp4 => {
                // The low half of a byte is the even column.
                let byte = unit[self.code_at(row, column / 2)];
                let code = if column.is_multiple_of(2) { byte & 15 } else { byte >> 4 };
                let at = layout.table + (scale as usize * 16 + code as usize) * 2;
                bf16_to_f32(u16::from_le_bytes([unit[at], unit[at + 1]]))
            }
            ExpertCodes::Mxfp8 => {
                let at = layout.table + unit[self.code_at(row, column)] as usize * 4;
                held(f32::from_le_bytes([unit[at], unit[at + 1], unit[at + 2], unit[at + 3]]) * power(scale))
            }
        }
    }
    /// The host twin of the device product on a unit: `out[o]` is one chain over the columns in rising order from
    /// +0.0, each product rounded before its add, on the stored row of output row `o` (gate rows first).
    pub fn product(&self, unit: &[u8], x: &[f32], out: &mut [f32]) -> Result<()> {
        let layout = self.layout()?;
        if unit.len() != layout.bytes || x.len() != self.columns || out.len() != self.rows {
            return Err(invalid("packed unit: the unit, the input or the output differs from rows and columns"));
        }
        for (o, out) in out.iter_mut().enumerate() {
            let row = loop_widen::stored_row(self.order, o, self.rows);
            let mut sum = 0.0f32;
            for (column, x) in x.iter().enumerate() { sum += self.weight(unit, layout, row, column) * x; }
            *out = sum;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{loop_packed::{PackedCodes, PackedMatrix}, storage_groups::{GATE_UP_ROWS, NVFP4_NIBBLES}};

    /// Stored bytes that differ from place to place.
    fn made(count: usize, salt: usize, low: u8, span: u8) -> Vec<u8> { (0..count).map(|i| low + (((i + salt).wrapping_mul(2_654_435_761) >> 9) as u8) % span).collect() }
    fn input(columns: usize, salt: u32) -> Vec<f32> { (0..columns as u32).map(|k| ((k + salt).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0).collect() }
    fn bits(words: &[f32]) -> Vec<u32> { words.iter().map(|w| w.to_bits()).collect() }
    /// The chain of the BF16 row product on a widened matrix of form A1.
    fn widened_product(bf16: &[u8], columns: usize, x: &[f32]) -> Vec<u32> {
        bf16.chunks(columns * 2).map(|row| row.as_chunks::<2>().0.iter().zip(x).fold(0.0f32, |sum, (w, x)| sum + bf16_to_f32(u16::from_le_bytes(*w)) * x).to_bits()).collect()
    }

    #[test]
    fn the_layout_pads_each_part() {
        let shape = |codes, rows, columns, order| UnitShape { codes, rows, columns, order };
        assert_eq!(shape(ExpertCodes::Nvfp4, 2304, 2304, Some(GATE_UP_ROWS)).layout().unwrap(), UnitLayout { scales: 2_654_208, table: 2_985_984, bytes: 2_994_176 });
        assert_eq!(shape(ExpertCodes::Mxfp8, 2304, 1152, None).layout().unwrap(), UnitLayout { scales: 2_654_208, table: 2_737_152, bytes: 2_738_176 });
        // The codes and the scales hold whole tiles: 32 rows, and whole stages of columns (two in the 8-bit form).
        assert_eq!(shape(ExpertCodes::Nvfp4, 3, 64, None).layout().unwrap(), UnitLayout { scales: 1024, table: 1152, bytes: 1152 + NVFP4_TABLE });
        assert_eq!(shape(ExpertCodes::Nvfp4, 3, 48, None).layout().unwrap(), shape(ExpertCodes::Nvfp4, 32, 64, None).layout().unwrap());
        assert_eq!(shape(ExpertCodes::Mxfp8, 1, 128, None).layout().unwrap(), UnitLayout { scales: 4096, table: 4224, bytes: 4224 + MXFP8_TABLE });
        assert_eq!(shape(ExpertCodes::Mxfp8, 1, 32, None).layout().unwrap(), shape(ExpertCodes::Mxfp8, 1, 128, None).layout().unwrap());
        for bad in [shape(ExpertCodes::Nvfp4, 0, 48, None), shape(ExpertCodes::Nvfp4, 3, 0, None), shape(ExpertCodes::Nvfp4, 3, 40, None), shape(ExpertCodes::Mxfp8, 3, 48, None),
            shape(ExpertCodes::Nvfp4, 3, 48, Some(GATE_UP_ROWS)), shape(ExpertCodes::Mxfp8, usize::MAX / 2, 64, None), shape(ExpertCodes::Nvfp4, 1, usize::MAX - 15, None)] {
            assert!(bad.layout().is_err(), "{bad:?}");
        }
    }
    #[test]
    fn a_four_bit_unit_gives_the_product_on_the_widened_matrix() {
        let (rows, columns, factor) = (6, 128, loop_widen::expert_factor(0.7, 2688.0).unwrap());
        for order in [None, Some(GATE_UP_ROWS)] {
            let shape = UnitShape { codes: ExpertCodes::Nvfp4, rows, columns, order };
            // Scale bytes from 0x28 to 0x47: block scales from 0.25 to under 4.
            let (codes, scales) = (made(rows * columns / 2, 1, 0, 255), made(rows * columns / NVFP4_BLOCK, 2, 0x28, 0x20));
            let layout = shape.layout().unwrap();
            let mut unit = vec![0xa5u8; layout.bytes];
            shape.fill(&codes, &scales, Some(factor), &mut unit).unwrap();
            // The tiles: a row has two stages of 32 code bytes and 4 scale bytes; stage 0 of the 32 rows of the block
            // (rows 6 to 31 zero), then stage 1. The stored bytes are not changed, and the fill bytes are zero.
            let mut tiles = vec![0u8; layout.table];
            for row in 0..rows {
                for stage in 0..2 {
                    tiles[stage * 1024 + row * 32..][..32].copy_from_slice(&codes[row * 64 + stage * 32..][..32]);
                    tiles[layout.scales + stage * 128 + row * 4..][..4].copy_from_slice(&scales[row * 8 + stage * 4..][..4]);
                }
            }
            assert_eq!(layout.scales, 2048);
            assert!(unit[..layout.table] == tiles[..], "{order:?}");
            let mut widened = vec![0u8; rows * columns * 2];
            loop_widen::widen_nvfp4(&codes, &scales, factor, rows, columns, order, NVFP4_NIBBLES, &mut widened).unwrap();
            let packed = PackedMatrix { codes: &codes, scales: &scales, rows, columns, form: PackedCodes::Nvfp4 { factor, nibbles: NVFP4_NIBBLES }, order };
            for salt in [3, 4] {
                let (x, mut out, mut twin) = (input(columns, salt), vec![f32::NAN; rows], vec![f32::NAN; rows]);
                shape.product(&unit, &x, &mut out).unwrap();
                packed.product(&x, &mut twin).unwrap();
                assert_eq!(bits(&out), widened_product(&widened, columns, &x), "{order:?}");
                assert_eq!(bits(&out), bits(&twin), "{order:?}");
            }
        }
    }
    #[test]
    fn an_eight_bit_unit_gives_the_product_on_the_widened_matrix() {
        let (rows, columns) = (4, 128);
        for order in [None, Some(GATE_UP_ROWS)] {
            let shape = UnitShape { codes: ExpertCodes::Mxfp8, rows, columns, order };
            // Every value byte but the two patterns that are not numbers. Scales from 2^-127 up: the small ones give
            // weights below the normal range of BF16, where the round to BF16 changes the value.
            let codes = made(rows * columns, 5, 0, 255).into_iter().map(|b| if b & 0x7f == 0x7f { b ^ 1 } else { b }).collect::<Vec<_>>();
            let scales = (0..rows * columns / MXFP8_BLOCK).map(|i| [0u8, 1, 3, 9, 60, 111, 115, 127, 128, 190, 230, 240][i % 12]).collect::<Vec<_>>();
            let mut unit = vec![0xa5u8; shape.layout().unwrap().bytes];
            shape.fill(&codes, &scales, None, &mut unit).unwrap();
            let mut widened = vec![0u8; rows * columns * 2];
            loop_widen::widen_mxfp8(&codes, &scales, rows, columns, order, &mut widened).unwrap();
            let (x, mut out) = (input(columns, 7), vec![f32::NAN; rows]);
            shape.product(&unit, &x, &mut out).unwrap();
            assert_eq!(bits(&out), widened_product(&widened, columns, &x), "{order:?}");
        }
    }
    /// Every scale byte and every value byte: the weight a unit gives is the BF16 value of form A1, and a scale
    /// byte is usable exactly where form A1 takes it.
    #[test]
    fn every_eight_bit_weight_of_a_unit_is_the_weight_of_the_widened_form() {
        // One row of 256 columns under one scale byte: column `v` holds value byte `v` (0 where `v` is not a number).
        let shape = UnitShape { codes: ExpertCodes::Mxfp8, rows: 1, columns: 256, order: None };
        let layout = shape.layout().unwrap();
        let codes = (0..=255u8).map(|v| if v & NOT_A_NUMBER == NOT_A_NUMBER { 0 } else { v }).collect::<Vec<_>>();
        let (mut compared, mut refused) = (0, 0);
        for scale in 0..=255u8 {
            let (scales, mut unit, mut widened) = ([scale; 8], vec![0u8; layout.bytes], [0u8; 512]);
            let filled = shape.fill(&codes, &scales, None, &mut unit);
            let reference = loop_widen::widen_mxfp8(&codes, &scales, 1, 256, None, &mut widened);
            assert_eq!(filled.is_ok(), reference.is_ok(), "scale {scale}");
            if filled.is_err() { refused += 1; continue; }
            for value in 0..256 {
                assert_eq!(shape.weight(&unit, layout, 0, value).to_bits(), bf16_to_f32(u16::from_le_bytes([widened[2 * value], widened[2 * value + 1]])).to_bits(), "scale {scale} value {value}");
                compared += 1;
            }
        }
        assert_eq!(compared + refused * 256, 256 * 256);
        assert!(compared > 60_000 && refused > 0, "{compared} {refused}");
    }
    #[test]
    fn a_unit_is_refused_where_the_widened_form_refuses_the_matrix() {
        let shape = UnitShape { codes: ExpertCodes::Nvfp4, rows: 2, columns: 64, order: None };
        let (codes, scales) = (vec![0x21u8; 64], vec![0x38u8; 8]);
        let mut unit = vec![0u8; shape.layout().unwrap().bytes];
        shape.fill(&codes, &scales, Some(0.5), &mut unit).unwrap();
        let text = |r: Result<()>| r.unwrap_err().to_string();
        assert!(text(shape.fill(&codes[..63], &scales, Some(0.5), &mut unit)).contains("differ from rows and columns"));
        assert!(text(shape.fill(&codes, &scales[..7], Some(0.5), &mut unit)).contains("differ from rows and columns"));
        assert!(text(shape.fill(&codes, &scales, Some(0.5), &mut unit[1..])).contains("differ from rows and columns"));
        assert!(text(shape.fill(&codes, &scales, None, &mut unit)).contains("the factor given differs"));
        // A 4-bit scale byte that is not a number.
        assert!(text(shape.fill(&codes, &[0x38, 0x7f, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38], Some(0.5), &mut unit)).contains("a 4-bit block scale gives a weight that is not finite"));
        let wide = UnitShape { codes: ExpertCodes::Mxfp8, rows: 1, columns: 128, order: None };
        let mut unit = vec![0u8; wide.layout().unwrap().bytes];
        wide.fill(&[0x38; 128], &[120; 4], None, &mut unit).unwrap();
        assert!(text(wide.fill(&[0x38; 128], &[120; 4], Some(0.5), &mut unit)).contains("the factor given differs"));
        assert!(text(wide.fill(&[0x38; 128], &[120, 120, 255, 120], None, &mut unit)).contains("an 8-bit block scale gives a weight that is not finite"));
        let mut values = [0x38u8; 128];
        values[9] = 0xff;
        assert!(text(wide.fill(&values, &[120; 4], None, &mut unit)).contains("an 8-bit weight is not a number"));
        // The product checks its extents.
        wide.fill(&[0x38; 128], &[120; 4], None, &mut unit).unwrap();
        let (x, mut out) = (vec![1.0f32; 128], [0.0f32; 1]);
        wide.product(&unit, &x, &mut out).unwrap();
        assert!(wide.product(&unit[4..], &x, &mut out).is_err());
        assert!(wide.product(&unit, &x[1..], &mut out).is_err());
        assert!(wide.product(&unit, &x, &mut [0.0; 2]).is_err());
    }
}
