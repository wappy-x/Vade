//! Decode projections use four fixed K ranges and one balanced FP32 merge.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Acc { pub a: f32, pub b: f32, pub c: f32, pub d: f32 }
pub(crate) const ZERO: Acc = Acc { a: 0.0, b: 0.0, c: 0.0, d: 0.0 };

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.nvvm.mma.m16n8k16.row.col.bf16"]
    fn mma(a0: u32, a1: u32, a2: u32, a3: u32, b0: u32, b1: u32,
        c0: f32, c1: f32, c2: f32, c3: f32) -> Acc;
}

#[inline(always)]
pub(crate) unsafe fn madd(a: [u32; 4], b: [u32; 2], c: Acc) -> Acc {
    mma(a[0], a[1], a[2], a[3], b[0], b[1], c.a, c.b, c.c, c.d)
}

#[inline(always)]
pub(crate) fn add(a: Acc, b: Acc) -> Acc {
    Acc { a: a.a + b.a, b: a.b + b.b, c: a.c + b.c, d: a.d + b.d }
}

/// Boundaries count native K=16 tiles, independent of token/output tile shape.
#[inline(always)]
pub(crate) fn range(tiles: usize, part: usize) -> (usize, usize) {
    (tiles * part / 4, tiles * (part + 1) / 4)
}

/// A merge part outside a K-split shard's ranges: -0.0 is the exact FP32
/// identity, so the four-part merge equals the shard's own subtree.
pub(crate) const EMPTY: Acc = Acc { a: -0.0, b: -0.0, c: -0.0, d: -0.0 };
/// Tiles of merge part `part` in a shard with `ranges` canonical ranges (0 is
/// four, the same bounds as `range`); None outside the shard's ranges.
#[inline(always)]
pub(crate) fn span(tiles: usize, part: usize, ranges: u32) -> Option<(usize, usize)> {
    crate::kernel_params::canonical_span(tiles, part, ranges)
}
/// For walks that close ranges in tile order: the end tile of `part`, and the
/// next part the shard holds. Ranges 0 give `range` and `part + 1`.
#[inline(always)]
pub(crate) fn end(tiles: usize, part: usize, ranges: u32) -> usize {
    if ranges == 0 { return range(tiles, part).1; }
    match span(tiles, part, ranges) { Some((_, end)) => end, None => usize::MAX }
}
#[inline(always)]
pub(crate) fn next(part: usize, ranges: u32) -> usize {
    part + if ranges == 2 { 2 } else { 1 }
}

/// Packed B fragments: contiguous 64x64 blocks, eight rows per MMA tile.
#[inline(always)]
pub(crate) fn weight_tile(row: usize, col: usize, kt: usize) -> usize {
    if kt % 4 == 0 {
        (row / 8 * (kt / 4) + col / 4) * 32 + row % 8 * 4 + col % 4
    } else { row * kt + col }
}

#[inline(always)]
pub(crate) fn weight_element(row: usize, col: usize, width: usize) -> usize {
    (weight_tile(row / 8, col / 16, width / 16) * 32
        + row % 8 * 4 + col % 8 / 2) * 4 + col % 2 + col % 16 / 8 * 2
}

/// Embedding dispatch follows the resident matrix layout, including hybrid text weights.
#[inline(always)]
pub(crate) fn embedding_element(row: usize, col: usize, width: usize, layout: u32) -> usize {
    if layout == 0 { row * width + col }
    else if layout == 1 { weight_element(row, col, width) }
    else { ((row / 8 * (width / 16) + col / 16) * 32
        + row % 8 * 4 + col % 8 / 2) * 4 + col % 2 + col % 16 / 8 * 2 }
}
