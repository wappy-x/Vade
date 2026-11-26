//! The embedding rows of a step on the device (kernel body `kernels-inkling/src/embed_rows.rs` in the Inkling image,
//! kind `EmbedRows`): the stored BF16 rows of the tokens of a step, widened to 32-bit words, read from the embedding
//! table that the device holds. So the host keeps no 32-bit copy of the table and uploads no row a step.
//!
//! Safe code only: the host check of every extent, then the record of the launch with the token numbers staged
//! beside it. The body has one entry, the recorded one, so a call while launches are not recorded is refused; the
//! caller asks `embed_rows_ready` first.
use super::*;
use crate::resident_launch::{Command, Kind};

/// Threads of a block: one block a token row.
pub const EMBED_THREADS: u32 = 256;
/// Most token rows of one launch, and the widest row.
pub const EMBED_ROWS_MAX: usize = 16_384;
pub const EMBED_WIDTH_MAX: usize = 16_384;

/// The host check of one launch: `vocab` rows of `width` BF16 values in `table` (two a word), `ids` the token
/// numbers, `output` exactly `ids.len() x width` words that meet neither the table nor each other's operands.
pub fn check_embed(vocab: usize, width: usize, table: (u64, usize), ids: &[u32], output: (u64, usize)) -> Result<(), &'static str> {
    if vocab == 0 || width == 0 || !width.is_multiple_of(2) || width > EMBED_WIDTH_MAX { return Err("embedding rows: the width is 0, odd or above the envelope"); }
    if ids.is_empty() || ids.len() > EMBED_ROWS_MAX { return Err("embedding rows: the token rows of a launch are 1 to 16384"); }
    if vocab.checked_mul(width / 2) != Some(table.1) { return Err("embedding rows: the table differs from vocab x width BF16 values"); }
    if ids.len().checked_mul(width) != Some(output.1) { return Err("embedding rows: the output differs from the token rows x the width"); }
    if ids.iter().any(|&id| id as usize >= vocab) { return Err("embedding rows: a token is outside the stored rows"); }
    if table.0 == 0 || output.0 == 0 || !table.0.is_multiple_of(4) || !output.0.is_multiple_of(4) { return Err("embedding rows: an operand is null or not word aligned"); }
    let end = |(address, words): (u64, usize)| address.checked_add(words as u64 * 4);
    let (Some(table_end), Some(output_end)) = (end(table), end(output)) else { return Err("embedding rows: an operand ends above the address space") };
    if output.0 < table_end && table.0 < output_end { return Err("embedding rows: the output overlaps the table"); }
    Ok(())
}

impl<'a> Backend<'a> {
    /// True when `gather_embed_rows_into` runs now: the Inkling image holds the entry and launches are recorded.
    pub fn embed_rows_ready(&self, rows: usize) -> bool { (1..=EMBED_ROWS_MAX).contains(&rows) && rows * 4 <= crate::resident_launch::stage::BYTES / 4 && self.resident_launches_active() }
    /// Row `r` of `output` is row `ids[r]` of the table (`vocab x width` BF16 values, two a word), each value as its
    /// 32-bit word. One launch for all rows.
    pub fn gather_embed_rows_into(&self, vocab: usize, width: usize, table: &DeviceVector<'_>, ids: &[u32], output: &mut DeviceVector<'_>) -> Result<(), String> {
        if self.arm_capture() { return Err("embedding rows: refused inside a stream capture".into()); }
        self.require_resident_stream()?;
        check_embed(vocab, width, (table.device_address(), table.len()), ids, (output.device_address(), output.len()))?;
        self.check_vector(table, vocab * width / 2)?;
        self.check_vector(output, ids.len() * width)?;
        if table.is_segmented() { return Err("embedding rows: the table must be contiguous".into()); }
        if output.is_segmented() || !output.writable_from(0) { return Err("embedding rows: the output requires private contiguous storage".into()); }
        self.require_arm_kernels(&[Kind::EmbedRows.kernel()])?;
        self.resident_completion()?;
        let bytes = ids.iter().flat_map(|id| id.to_ne_bytes()).collect::<Vec<_>>();
        let blocks = u32::try_from(ids.len()).map_err(|_| "embedding rows: too many rows")?;
        let head = [table.device_address(), output.device_address(), vocab as u64, width as u64];
        let count = ids.len() as u64;
        let recorded = self.record_resident_tabled(Kind::EmbedRows, [blocks, 1, 1, EMBED_THREADS, 0], &bytes, |address| {
            let mut command = Command::new(Kind::EmbedRows);
            for (index, value) in head.into_iter().chain([address, count]).enumerate() { command.word(index, value); }
            command
        })?;
        if !recorded { return Err("embedding rows: the launch runs only as a recorded launch whose token table fits the stage, and that is not the case now".into()); }
        self.resident_completion()
    }
}
#[cfg(test)]
mod tests {
    use super::check_embed;
    /// Every refusal of the host check, and a launch it takes.
    #[test]
    fn the_embed_check_refuses_what_a_launch_cannot_take() {
        let (vocab, width) = (100usize, 64usize);
        let table = (0x1_0000_0000u64, vocab * width / 2);
        let output = |rows: usize| (0x2_0000_0000u64, rows * width);
        assert_eq!(check_embed(vocab, width, table, &[0, 99, 5], output(3)), Ok(()));
        let refused = |r: Result<(), &'static str>, text: &str| assert_eq!(r, Err(text));
        refused(check_embed(vocab, 63, (table.0, vocab * 63 / 2), &[0], output(1)), "embedding rows: the width is 0, odd or above the envelope");
        refused(check_embed(vocab, width, table, &[], output(0)), "embedding rows: the token rows of a launch are 1 to 16384");
        refused(check_embed(vocab, width, (table.0, table.1 - 1), &[0], output(1)), "embedding rows: the table differs from vocab x width BF16 values");
        refused(check_embed(vocab, width, table, &[0, 1], output(1)), "embedding rows: the output differs from the token rows x the width");
        refused(check_embed(vocab, width, table, &[0, 100], output(2)), "embedding rows: a token is outside the stored rows");
        refused(check_embed(vocab, width, table, &[0], (0x2_0000_0002, width)), "embedding rows: an operand is null or not word aligned");
        refused(check_embed(vocab, width, table, &[0], (table.0 + 64, width)), "embedding rows: the output overlaps the table");
    }
}
