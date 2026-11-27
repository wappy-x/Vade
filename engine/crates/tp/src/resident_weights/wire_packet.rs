//! Layout shared by the resident prompt gather and its CPU and device proofs.
use crate::Result;
use spark_gpu_probe::prompt_ops::{Op, Shape};
use std::ops::Range;

#[cfg(test)]
#[path = "wire_packet/tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Format {
    Fp32,
    Bf16,
}

impl Format {
    pub fn words(self, words: usize) -> Result<usize> {
        match self {
            Self::Fp32 => Ok(words),
            Self::Bf16 if words.is_multiple_of(2) => Ok(words / 2),
            Self::Bf16 => Err("BF16 wire part has an odd extent".into()),
        }
    }
}

/// The wire stores all parts of one rank next to each other. A scatter groups
/// each part's rank blocks before the ordinary placement or BF16 unpack.
pub struct Layout {
    pub words: usize,
    pub stride: usize,
    pub offsets: Vec<usize>,
    pub widths: Vec<usize>,
    pub sections: Vec<Range<usize>>,
    pub copies: Vec<[usize; 3]>,
}

impl Layout {
    pub fn new(ranks: usize, parts: &[(usize, Format)]) -> Result<Self> {
        let ranks = std::num::NonZeroUsize::new(ranks).ok_or("prompt wire has no owners")?.get();
        if parts.is_empty() || parts.iter().any(|&(words, _)| words == 0) {
            return Err("prompt wire layout is empty".into());
        }
        let mut widths = Vec::with_capacity(parts.len());
        let mut stride = 0usize;
        for &(words, format) in parts {
            let width = format.words(words)?;
            widths.push(width);
            stride = stride.checked_add(width).ok_or("prompt wire stride overflow")?;
        }
        let words = stride.checked_mul(ranks).ok_or("prompt wire extent overflow")?;
        let mut sections = Vec::with_capacity(parts.len());
        let mut offsets = Vec::with_capacity(parts.len());
        let mut copies = Vec::new();
        let (mut source, mut destination) = (0, 0);
        for &width in &widths {
            let end = destination + width * ranks;
            offsets.push(source);
            sections.push(destination..end);
            for rank in 0..ranks {
                copies.push([rank * stride + source, destination + rank * width, width]);
            }
            source += width;
            destination = end;
        }
        Ok(Self { words, stride, offsets, widths, sections, copies })
    }
    /// Direct slice of the raw exchange buffer covering all `ranks` copies of part `index` with rank stride `self.stride`.
    pub fn direct_range(&self, ranks: usize, index: usize) -> Range<usize> {
        let start = self.offsets[index];
        let len = (ranks - 1) * self.stride + self.widths[index];
        start..start + len
    }
}

pub fn pack_shape(words: usize) -> Result<Shape> {
    let cols = u32::try_from(words).map_err(|_| "wire pack extent exceeds ABI")?;
    Ok(Shape { cols, ..Shape::new(Op::WirePackBf16, 1).ok_or("wire pack shape")? })
}

pub fn unpack_shape(tokens: usize, width: usize, ranks: usize, transpose: bool, finite: bool) -> Result<Shape> {
    unpack_shape_strided(tokens, width, ranks, transpose, finite, 0)
}

pub fn unpack_shape_strided(tokens: usize, width: usize, ranks: usize, transpose: bool, finite: bool, stride: usize) -> Result<Shape> {
    let whole = ranks.checked_mul(width).ok_or("BF16 wire extent overflow")?;
    let rows = u32::try_from(if transpose { whole } else { ranks }).map_err(|_| "wire rows exceed ABI")?;
    let cols = u32::try_from(if transpose { tokens } else { width }).map_err(|_| "wire columns exceed ABI")?;
    let dim = u32::try_from(stride).map_err(|_| "wire stride exceeds ABI")?;
    let taps = if transpose && stride != 0 { u32::try_from(ranks).map_err(|_| "wire ranks exceed ABI")? } else { 0 };
    let shape = Shape::new(Op::WireUnpackBf16, if transpose { 1 } else { tokens }).ok_or("wire unpack shape")?;
    Ok(Shape { rows, cols, dim, taps, first: u32::from(transpose), slots: u32::from(finite), ..shape })
}

#[cfg(target_os = "linux")]
pub fn write_part<'d>(b: &spark_gpu_probe::driver::Backend<'d>, source: &spark_gpu_probe::driver::DeviceVector<'_>,
    output: &mut spark_gpu_probe::driver::DeviceVector<'d>, at: usize, format: Format) -> Result<usize> {
    let words = format.words(source.len())?;
    let end = at.checked_add(words).ok_or("prompt packet offset overflow")?;
    match format {
        Format::Fp32 => b.copy_range(source, output, 0, at, words)?,
        Format::Bf16 => {
            let shape = pack_shape(source.len())?;
            b.prompt_operation_output(shape, [Some(source), None, None, None, None], None, Some(output.resident_range(at..end)?))?;
        }
    }
    Ok(end)
}

#[cfg(any(target_os = "linux", test))]
pub(super) fn finish(ring: Option<&mut crate::transport::Ring>, result: Result<()>) -> Result<()> {
    if result.is_err() && let Some(ring) = ring {
        ring.abort();
    }
    result
}
