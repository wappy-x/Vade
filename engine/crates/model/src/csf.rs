//! Host decoding for `csf_32x2_v1`. See `../CSF.md` for the public source.
//! Components are unmodified checkpoint bytes, before any native GPU reorder.
use crate::{Result, invalid};

/// The quantization metadata selects the code width, not the model name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization { Nvfp4, Mxfp4 }
impl Quantization {
    fn bits(self) -> usize { match self { Self::Nvfp4 => 4, Self::Mxfp4 => 1 } }
}

/// One logical scale plane. `columns` counts scales, not weight elements.
/// All components are contiguous byte slices. Page bases are little-endian
/// U32 bytes. Offsets and exceptions are little-endian U16 bytes. Each slice
/// contains only this plane; page bases start at its own exception arena.
#[derive(Debug, Clone, Copy)]
pub struct Csf32<'a> {
    pub rows: usize,
    pub columns: usize,
    pub quantization: Quantization,
    pub bases: &'a [u8],
    pub codes: &'a [u8],
    pub page_bases: &'a [u8],
    pub offsets: &'a [u8],
    pub exceptions: &'a [u8],
}

#[derive(Clone, Copy)]
struct Geometry {
    rows: usize, columns: usize, bits: usize, count: usize,
    blocks: usize, pages: usize, code_bytes: usize,
}
struct Block { row: usize, column: usize, width: usize, index: usize, page: usize, code: usize }

fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("invalid csf_32x2_v1: {message}"))) }
}
fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| invalid("csf_32x2_v1 size overflow"))
}
fn sum(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| invalid("csf_32x2_v1 size overflow"))
}
fn length(name: &str, actual: usize, expected: usize) -> Result<()> {
    require(actual == expected, &format!("{name} byte length {actual}, expected {expected}"))
}
// Callers first check the complete component length against the geometry.
fn u16_at(bytes: &[u8], index: usize) -> usize {
    let at = index * 2;
    u16::from_le_bytes([bytes[at], bytes[at + 1]]) as usize
}
fn u32_at(bytes: &[u8], index: usize) -> usize {
    let at = index * 4;
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize
}

impl Geometry {
    fn new(rows: usize, columns: usize, bits: usize) -> Result<Self> {
        require(rows > 0 && rows % 32 == 0 && columns > 0,
            "scale shape needs positive columns and complete row blocks")?;
        let count = product(rows, columns)?;
        let blocks = product(rows / 32, columns.div_ceil(2))?;
        let groups = product(rows.div_ceil(128), columns.div_ceil(4))?;
        let pages = groups.div_ceil(32);
        // A complete row block makes count divisible by each packing divisor.
        let code_bytes = count / (8 / bits);
        Ok(Self { rows, columns, bits, count, blocks, pages, code_bytes })
    }

    fn visit(self, mut visit: impl FnMut(Block) -> Result<()>) -> Result<()> {
        let (mut index, mut group, mut code) = (0, 0, 0);
        for tile_row in (0..self.rows).step_by(128) {
            let row_end = tile_row + (self.rows - tile_row).min(128);
            for tile_column in (0..self.columns).step_by(4) {
                let column_end = tile_column + (self.columns - tile_column).min(4);
                for column in (tile_column..column_end).step_by(2) {
                    let width = (self.columns - column).min(2);
                    for row in (tile_row..row_end).step_by(32) {
                        visit(Block { row, column, width, index, page: group / 32, code })?;
                        // These counters cannot exceed the checked geometry totals.
                        index += 1;
                        code += 32 * width * self.bits / 8;
                    }
                }
                group += 1;
            }
        }
        Ok(())
    }
}

impl Csf32<'_> {
    fn geometry(&self) -> Result<Geometry> {
        Geometry::new(self.rows, self.columns, self.quantization.bits())
    }

    fn validate_geometry(&self, geometry: Geometry) -> Result<()> {
        length("csf32_bases", self.bases.len(), self.rows)?;
        length("csf32_codes", self.codes.len(), geometry.code_bytes)?;
        length("csf32_page_bases", self.page_bases.len(), product(geometry.pages, 4)?)?;
        length("csf32_offsets", self.offsets.len(), product(sum(geometry.blocks, geometry.pages)?, 2)?)?;
        require(self.exceptions.len() % 2 == 0 && self.exceptions.len() <= u32::MAX as usize,
            "csf32_exceptions needs complete records in the page address range")?;
        let max_base = 256 - (1 << geometry.bits);
        require(self.bases.iter().all(|&base| usize::from(base) <= max_base),
            "csf32_bases palette exceeds the scale byte range")?;
        let mut page = None;
        let (mut origin, mut records) = (0, 0);
        geometry.visit(|block| {
            let at = block.index + block.page;
            let start = u16_at(self.offsets, at);
            let stop = u16_at(self.offsets, at + 1);
            if page != Some(block.page) {
                require(start == 0, "csf32_offsets page must start at zero")?;
                require(u32_at(self.page_bases, block.page) == product(records, 2)?,
                    "csf32_page_bases must cover one consecutive exception arena")?;
                origin = records;
                page = Some(block.page);
            }
            let size = stop.checked_sub(start).ok_or_else(|| invalid(
                "invalid csf_32x2_v1: csf32_offsets must not decrease within a page"))?;
            require(size <= 32 * block.width, "csf32_offsets count exceeds the block extent")?;
            let end = sum(origin, stop)?;
            require(end <= self.exceptions.len() / 2, "csf32_exceptions is shorter than its prefixes")?;
            let mut previous = None;
            for record in sum(origin, start)?..end {
                let position = u16_at(self.exceptions, record) & 255;
                require(position < 64 && position % 2 < block.width,
                    "csf32_exceptions position is outside the block")?;
                require(previous.is_none_or(|last| last < position),
                    "csf32_exceptions positions must be sorted and unique in each block")?;
                previous = Some(position);
            }
            records = end;
            Ok(())
        })?;
        length("csf32_exceptions", self.exceptions.len(), product(records, 2)?)
    }

    /// Check every component length, page prefix, palette, and exception position.
    /// Exception bytes may contain any scale bit pattern. Arithmetic consumers
    /// must check whether those patterns are valid for their scale format.
    pub fn validate(&self) -> Result<()> { self.validate_geometry(self.geometry()?) }

    fn decode_validated(&self, geometry: Geometry, output: &mut [u8]) -> Result<()> {
        geometry.visit(|block| {
            for local_row in 0..32 {
                for local_column in 0..block.width {
                    let field = local_row * block.width + local_column;
                    let bit = field * geometry.bits;
                    let code = (self.codes[block.code + bit / 8] >> (bit % 8))
                        & ((1u8 << geometry.bits) - 1);
                    output[(block.row + local_row) * self.columns + block.column + local_column]
                        = self.bases[block.row + local_row] + code;
                }
            }
            let at = block.index + block.page;
            let origin = u32_at(self.page_bases, block.page) / 2;
            for record in origin + u16_at(self.offsets, at)..origin + u16_at(self.offsets, at + 1) {
                let word = u16_at(self.exceptions, record);
                let position = word & 255;
                output[(block.row + position / 2) * self.columns + block.column + position % 2]
                    = (word >> 8) as u8;
            }
            Ok(())
        })
    }

    /// Expand exact scale bytes in logical row-major order. No scale is rounded.
    /// On invalid input, leave the caller's output unchanged.
    pub fn decode_into(&self, output: &mut [u8]) -> Result<()> {
        let geometry = self.geometry()?;
        length("scale output", output.len(), geometry.count)?;
        self.validate_geometry(geometry)?;
        self.decode_validated(geometry, output)
    }

    /// Return logical scale bytes. For NVFP4, this is the existing E4M3 block
    /// scale layout consumed by `flash::nvfp4::dequantize`.
    pub fn expand(&self) -> Result<Vec<u8>> {
        let geometry = self.geometry()?;
        self.validate_geometry(geometry)?;
        let mut output = Vec::new();
        output.try_reserve_exact(geometry.count)
            .map_err(|_| invalid("csf_32x2_v1 scale output allocation failed"))?;
        output.resize(geometry.count, 0);
        self.decode_validated(geometry, &mut output)?;
        Ok(output)
    }

    /// Decode NVFP4 weights to FP32 with the existing host arithmetic order.
    /// Weight nibbles are row-major, low nibble first. Each scale covers the
    /// existing NVFP4 block size. `global` is the projection's weight_scale_2.
    /// This method does not admit a model or add an inference execution path.
    pub fn dequantize_nvfp4(&self, packed: &[u8], global: f32, output: &mut [f32]) -> Result<()> {
        require(self.quantization == Quantization::Nvfp4, "NVFP4 weights need NVFP4 scale metadata")?;
        let geometry = self.geometry()?;
        let values = product(geometry.count, crate::flash::nvfp4::BLOCK)?;
        length("weight output", output.len(), values)?;
        length("packed weights", packed.len(), values / 2)?;
        require(global.is_finite() && global > 0.0, "global weight scale must be positive and finite")?;
        let scales = self.expand()?;
        crate::flash::nvfp4::dequantize(packed, &scales, global, output)
    }
}
