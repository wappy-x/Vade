//! Two row parts in one mailbox message. All first-part rows precede all second-part rows.
//! Offsets and widths describe raw words. The copy does not change their bits.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair { pub rows: usize, pub widths: [usize; 2], pub columns: [usize; 2], pub offsets: [usize; 2] }
impl Pair {
    /// Check both vector extents and the combined slot extent before a device launch.
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn checked(rows: usize, parts: [(usize, usize, usize, usize); 2], slot_words: usize) -> Option<Self> {
        let mut p = Self { rows, widths: [0; 2], columns: [0; 2], offsets: [0; 2] };
        let mut words = 0usize;
        for (i, (len, width, columns, offset)) in parts.into_iter().enumerate() {
            if rows == 0 || columns == 0 || offset.checked_add(columns)? > width || rows.checked_mul(width)? > len { return None; }
            words = words.checked_add(rows.checked_mul(columns)?)?;
            p.widths[i] = width; p.columns[i] = columns; p.offsets[i] = offset;
        }
        if words > slot_words || words.checked_mul(4)? > u32::MAX as usize { return None; }
        Some(p)
    }
    pub fn words(self) -> usize { self.rows * (self.columns[0] + self.columns[1]) }
    /// The vector and word offset for a valid payload word. A quad uses this at its first word.
    #[inline(always)]
    pub fn locate(self, word: usize) -> (usize, usize) {
        let first = self.rows * self.columns[0];
        let (part, i) = if word < first { (0, word) } else { (1, word - first) };
        (part, i / self.columns[part] * self.widths[part] + self.offsets[part] + i % self.columns[part])
    }
}

#[cfg(test)]
#[path = "tp_payload_tests.rs"]
mod tests;
