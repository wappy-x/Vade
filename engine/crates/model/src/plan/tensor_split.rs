//! Checked ownership of complete tensor units. This is not model admission.
use crate::{Result, invalid, shapes::{MAX_SPARKS, MAX_TENSOR_RANK}};
use super::storage::{WeightStorage, ScaleEncoding};
use std::ops::Range;

fn need(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(invalid(why)) } }
fn mul(a: u64, b: u64) -> Result<u64> { a.checked_mul(b).ok_or_else(|| invalid("tensor split product overflow")) }
fn product(shape: &[u64]) -> Result<u64> { shape.iter().try_fold(1, |a,&b| mul(a,b)) }

/// Entries are in rank order. Coordinates refer to the original tensor axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Owner { pub rank: u32, pub interval: Range<u64> }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ownership { shape: Vec<u64>, axis: usize, unit: u64, owners: Vec<Owner>, elements: u64 }
impl Ownership {
    /// Match the logical plan's equal intervals. A unit can be a whole head.
    pub fn equal(shape: &[u64], axis: usize, unit: u64, parts: u32) -> Result<Self> {
        need((1..=MAX_SPARKS).contains(&parts), "tensor split rank count is outside the limit")?;
        let n = *shape.get(axis).ok_or_else(|| invalid("tensor split axis is outside the shape"))?;
        let stride = mul(unit, u64::from(parts))?;
        need(stride > 0 && n > 0 && n % stride == 0, "tensor split needs complete equal units")?;
        let width = n / u64::from(parts);
        let owners = (0..parts).map(|rank| Owner { rank, interval: u64::from(rank)*width..(u64::from(rank)+1)*width }).collect::<Vec<_>>();
        Self::checked(shape, axis, unit, &owners)
    }
    /// Check intervals received from a logical planner before any wire allocation.
    /// Do not sort: rank order and global coordinate order must already agree.
    pub fn checked(shape: &[u64], axis: usize, unit: u64, owners: &[Owner]) -> Result<Self> {
        need(!shape.is_empty() && shape.len() <= MAX_TENSOR_RANK && shape.iter().all(|&n| n > 0), "invalid tensor split shape")?;
        let n = *shape.get(axis).ok_or_else(|| invalid("tensor split axis is outside the shape"))?;
        need(unit > 0 && !owners.is_empty() && owners.len() <= MAX_SPARKS as usize, "invalid tensor split units or ranks")?;
        let width = n / owners.len() as u64;
        let mut end = 0;
        for (rank, owner) in owners.iter().enumerate() {
            let r = &owner.interval;
            need(owner.rank as usize == rank, "tensor split owners are not in rank order")?;
            need(r.start == end && r.start < r.end && r.end <= n, "tensor split has a gap, overlap, or invalid bound")?;
            need(r.start % unit == 0 && r.end % unit == 0 && r.end-r.start == width, "tensor split needs complete equal units")?;
            end = r.end;
        }
        need(end == n, "tensor split does not cover the global axis")?;
        let elements = product(shape)?;
        Ok(Self { shape: shape.into(), axis, unit, owners: owners.into(), elements })
    }
    pub fn shape(&self) -> &[u64] { &self.shape }
    pub fn axis(&self) -> usize { self.axis }
    pub fn unit(&self) -> u64 { self.unit }
    pub fn owners(&self) -> &[Owner] { &self.owners }
    pub fn elements(&self) -> u64 { self.elements }
    pub fn owner(&self, rank: u32) -> Result<&Owner> { self.owners.get(rank as usize).ok_or_else(|| invalid("tensor split rank is outside the owners")) }
    /// One packet contains all outer rows of one owner's contiguous inner span.
    pub fn geometry(&self) -> Result<(u64,u64,u64)> {
        Ok((product(&self.shape[..self.axis])?, self.shape[self.axis], product(&self.shape[self.axis+1..])?))
    }
}

/// A projection owns output rows only. Its input K is always complete.
/// Scale coordinates remain relative to the original tensor, never the slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionRows { rows: Range<u64>, shape: [u64;2], storage: WeightStorage, scale: Option<ScaleGrid> }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleGrid { pub block: [u64;2], pub shape: [u64;2], pub encoding: ScaleEncoding }
impl ProjectionRows {
    pub fn new(ownership: &Ownership, rank: u32, storage: WeightStorage) -> Result<Self> {
        need(ownership.axis == 0 && ownership.shape.len() == 2, "projection split must retain the complete input K")?;
        let shape = [ownership.shape[0],ownership.shape[1]];
        let rows = ownership.owner(rank)?.interval.clone();
        let grid = match storage {
            WeightStorage::Plain => None,
            WeightStorage::Mxfp8 => Some(([1,32],ScaleEncoding::Direct)),
            WeightStorage::Fp8Block { block } => Some((block,ScaleEncoding::Direct)),
            WeightStorage::Nvfp4 { encoding, .. } => {
                need(shape[1].is_multiple_of(16), "NVFP4 projection needs complete input scale groups")?;
                Some(([1,16],encoding))
            },
        };
        let scale = grid.map(|(block,encoding)| -> Result<ScaleGrid> {
            need(block.iter().all(|&v| v > 0), "projection scale block is empty")?;
            let shape = [shape[0].div_ceil(block[0]),shape[1].div_ceil(block[1])];
            mul(shape[0],shape[1])?;
            Ok(ScaleGrid { block,shape,encoding })
        }).transpose()?;
        Ok(Self { rows,shape,storage,scale })
    }
    pub fn rows(&self) -> Range<u64> { self.rows.clone() }
    pub fn shape(&self) -> [u64;2] { self.shape }
    pub fn storage(&self) -> WeightStorage { self.storage }
    pub fn scale(&self) -> Option<ScaleGrid> { self.scale }
    /// Logical index in the full original scale grid, including a shared edge block.
    /// Compressed encodings require full-plane decode before this index is used.
    pub fn scale_index(&self, local_row: u64, column: u64) -> Result<u64> {
        need(local_row < self.rows.end-self.rows.start && column < self.shape[1], "projection scale coordinate is outside the slice")?;
        let s = self.scale.ok_or_else(|| invalid("plain projection has no scale grid"))?;
        let row = (self.rows.start+local_row)/s.block[0];
        Ok(mul(row,s.shape[1])? + column/s.block[1])
    }
    /// A backend with only a rebased local grid cannot represent a shared edge
    /// block. Compressed pages cannot be sliced by a row pointer at all.
    #[allow(clippy::manual_is_multiple_of)] // A zero divisor must still panic in `%`.
    pub fn direct_scale_rows(&self) -> Result<Range<u64>> {
        let s = self.scale.ok_or_else(|| invalid("plain projection has no scale grid"))?;
        need(s.encoding == ScaleEncoding::Direct, "compressed scales need full-plane decode before row slicing")?;
        need(self.rows.start % s.block[0] == 0, "backend cannot rebase an unaligned scale origin")?;
        Ok(self.rows.start/s.block[0]..self.rows.end.div_ceil(s.block[0]))
    }
    pub fn input_features(&self) -> u64 { self.shape[1] }
}
