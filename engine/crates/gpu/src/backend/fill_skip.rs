//! Zero-fill accounting and the place of a fill that `fill_free` lists as not needed.
use super::*;
impl Backend<'_> {
    /// The place of a zero fill that `fill_free` lists as not needed: each writer of this fresh private output stores every word
    /// with raw stores before any read of it, on every path that can reach a consumer. It runs nothing unless a proof run names `zero` or `poison`.
    pub fn skip_fill(&self, vector: &mut DeviceVector<'_>) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        match crate::fill_free::mode() {
            crate::fill_free::Mode::Skip=>Ok(()),
            crate::fill_free::Mode::Zero=>self.zero_vector(vector),
            crate::fill_free::Mode::Poison=>{let len=vector.len;let source=self.upload_vector(&vec![f32::from_bits(crate::fill_free::POISON);len])?;self.copy_prefix(&source,vector,len)},
        }
    }
    /// Range form of [`Self::skip_fill`] for newly mapped writable suffix pages on a segmented plane.
    pub fn skip_fill_range(&self, vector: &mut DeviceVector<'_>, range: std::ops::Range<usize>) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        if range.start > range.end || range.end > vector.len {
            return Err("skip fill range exceeds vector".into());
        }
        if range.is_empty() { return Ok(()); }
        match crate::fill_free::mode() {
            crate::fill_free::Mode::Skip => Ok(()),
            crate::fill_free::Mode::Zero => self.zero_vector_range(vector, range),
            crate::fill_free::Mode::Poison => {
                let len = range.len();
                let source = self.upload_vector(&vec![f32::from_bits(crate::fill_free::POISON); len])?;
                self.copy_range(&source, vector, 0, range.start, len)
            }
        }
    }
    pub(super) fn note_zero_fill(&self,launches:u128,rows:u128,words:usize) {
        let mut stats=self.stats.borrow_mut();stats.zero_fill_launches+=launches;stats.zero_fill_rows+=rows;stats.zero_fill_words+=words as u128;
    }
}
