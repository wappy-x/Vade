//! One rank's row bank for the layer-major prompt transaction (flash/next_split_plan.rs). Per
//! row it keeps what a tile stage leaves for its next stage: the residual (W words), the mixed
//! rows (H) and the HC injection words, in one vector_arena allocation. A tile enters the
//! workspace as zero-copy slices of the bank (no row copies) and leaves before any other tile
//! or pass uses the workspace. The kernels see the same words at other addresses; a dense
//! pack whose plane offer was made for a tile that left forms its planes from the FP32 rows.
use super::*;
use crate::flash_next_workspace::HC_INJECT_WORDS;

pub(in crate::flash) struct Bank<'a> {
    residual: DeviceVector<'a>,
    mixed: DeviceVector<'a>,
    inject: DeviceVector<'a>,
    rows: usize,
}

/// The workspace's own vectors while a bank tile is in; `Bank::leave` puts them back.
#[must_use = "a bank tile must leave the workspace"]
pub(in crate::flash) struct Entered<'a> {
    residual: DeviceVector<'a>,
    mixed: DeviceVector<'a>,
    inject: DeviceVector<'a>,
}

impl<'a> Bank<'a> {
    /// A bank of `rows` rows (one allocation; an error leaves nothing allocated).
    pub(in crate::flash) fn new(b: &Backend<'a>, rows: usize) -> Result<Self> {
        if rows == 0 { return Err("empty prompt bank".into()); }
        let lengths = [rows.checked_mul(W), rows.checked_mul(H), rows.checked_mul(HC_INJECT_WORDS)];
        let lengths = lengths.map(|n| n.ok_or("prompt bank size overflow"));
        let mut v = b.vector_arena(&[lengths[0]?, lengths[1]?, lengths[2]?])?.into_iter();
        let (residual, mixed, inject) = (v.next().unwrap(), v.next().unwrap(), v.next().unwrap());
        Ok(Self { residual, mixed, inject, rows })
    }
    /// Bank rows `span`'s residual words (a zero-copy view).
    pub(in crate::flash) fn residual(&self, span: std::ops::Range<usize>) -> Result<DeviceVector<'a>> {
        if span.is_empty() || span.end > self.rows { return Err(format!("prompt bank rows {span:?} outside {} rows", self.rows)); }
        Ok(self.residual.slice(span.start * W..span.end * W)?)
    }
    /// Bank rows `span` become the workspace's residual, mixed and injection rows 0..span.len().
    pub(in crate::flash) fn enter(&self, work: &mut Workspace<'a>, span: std::ops::Range<usize>) -> Result<Entered<'a>> {
        if span.is_empty() || span.end > self.rows || span.len() > work.rows {
            return Err(format!("prompt bank tile {span:?} outside {} bank rows or {} workspace rows", self.rows, work.rows));
        }
        let residual = self.residual.slice(span.start * W..span.end * W)?;
        let mixed = self.mixed.slice(span.start * H..span.end * H)?;
        let inject = self.inject.slice(span.start * HC_INJECT_WORDS..span.end * HC_INJECT_WORDS)?;
        Ok(Entered {
            residual: std::mem::replace(&mut work.residual, residual),
            mixed: std::mem::replace(&mut work.mixed, mixed),
            inject: work.hc.swap_inject(inject),
        })
    }
    /// The tile leaves: the workspace gets its own vectors back. A plane offer made for the
    /// tile's mixed rows is dropped, so no later pack can take planes of rows that left.
    pub(in crate::flash) fn leave(b: &Backend<'a>, work: &mut Workspace<'a>, entered: Entered<'a>) {
        work.residual = entered.residual;
        work.mixed = entered.mixed;
        let _ = work.hc.swap_inject(entered.inject);
        b.drop_dense_offer();
    }
}
