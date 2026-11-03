//! Cohort MTP verify (VADE_GDN_COHORT): one descriptor table and one output gate per
//! multi-chat verify pass, and (VADE_GDN_COHORT_COPIES) its entry histories and replay
//! tapes in two launches per GDN layer, and (VADE_GDN_COHORT_KERNELS) its rows and history
//! kernels in one launch each. Split from gdn.rs (file-size ratchet).
use super::{Backend, DeviceVector, Gdn, GdnScratch, GdnState, Result, DIM};

/// VADE_GDN_COHORT (default on; 0 = one descriptor upload and one output gate per chat):
/// a multi-chat MTP verify pass whose every window runs the rows kernel uploads one
/// descriptor table for the pass and gates all its rows in one flash_gdn_gate launch.
/// The gate is elementwise per (row, head) and each chat's kernels read the same
/// descriptor words, so the bytes are the per-chat path's.
pub(crate) fn cohort_verify() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VADE_GDN_COHORT").is_ok_and(|v| v == "0"))
}

/// VADE_GDN_COHORT_COPIES (default on; 0 = four copies per chat): a cohort verify pass
/// (cohort_verify) snapshots every chat's entry conv history and captures every chat's replay
/// tape (qkv, a, b rows) with two flash_hybrid_copy launches per GDN layer (wide rows, then the
/// 48-wide a/b rows) instead of four copies per chat; the same words, all read before any of
/// the layer's scans runs (each chat's scan writes only its own state, history and rows).
pub(crate) fn cohort_copies() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VADE_GDN_COHORT_COPIES").is_ok_and(|v| v == "0"))
}

/// VADE_GDN_COHORT_KERNELS (default on; 0 = per-chat launches): a cohort verify layer
/// (cohort_verify, every window on the rows kernel, which convolves and prepares its rows
/// itself) collects each chat's rows-kernel and history parameters and runs them as two
/// (three when a window crosses a chunk end) launches with the chat in grid-Y
/// (flash_next_gdn_chunk_rows_cohort, flash_gdn_history_cohort), each CTA the per-chat
/// kernel's code on its chat's parameters. Per chat the order is unchanged (rows, rest rows,
/// history) and no chat touches another's words, so the bytes are the per-chat launches'.
/// Needs a PTX with those entries.
pub(crate) fn cohort_kernels() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VADE_GDN_COHORT_KERNELS").is_ok_and(|v| v == "0"))
}
/// Per-chat launch parameters a cohort verify layer collects (cohort_kernels), in chat order.
#[derive(Default)]
pub(crate) struct CohortLaunches {
    /// Rows-kernel calls: [0] each chat's first (up to the chunk end), [1] the rest past it.
    pub(super) rows: [Vec<spark_gpu_probe::gdn_params::GdnChunkNext>; 2],
    pub(super) history: Vec<spark_gpu_probe::kernel_params::GdnHistory>,
}
/// Table bytes of one chat's cohort launches: two rows calls, history.
pub(crate) const COHORT_LAUNCH_BYTES: usize = 2 * std::mem::size_of::<spark_gpu_probe::gdn_params::GdnChunkNext>()
    + std::mem::size_of::<spark_gpu_probe::kernel_params::GdnHistory>();
const _: () = assert!(COHORT_LAUNCH_BYTES % 8 == 0 && std::mem::size_of::<spark_gpu_probe::gdn_params::GdnChunkNext>() % 8 == 0);
use spark_gpu_probe::{gdn_params::GdnChunkNext, kernel_params::GdnHistory};
/// Writes `bytes` at byte `at` of a launch record.
fn put(record: &mut [u8], at: usize, bytes: &[u8]) { record[at..at + bytes.len()].copy_from_slice(bytes); }
/// One chat's GdnHistory as table bytes: each field's little-endian bytes at its repr(C)
/// offset and the trailing padding (fields end at byte 36 of 40) zero. Viewing the struct
/// itself as bytes would read its padding, which nothing initializes. The kernel reads the
/// table entry as the same repr(C) struct and uses named fields only.
fn history_record(h: &GdnHistory, out: &mut Vec<u8>) {
    use std::mem::offset_of as at;
    let mut r = [0u8; std::mem::size_of::<GdnHistory>()];
    for (o, a) in [(at!(GdnHistory, x), h.x as usize as u64), (at!(GdnHistory, history), h.history as usize as u64), (at!(GdnHistory, rows), h.rows)] {
        put(&mut r, o, &a.to_le_bytes());
    }
    for (o, w) in [(at!(GdnHistory, channels), h.channels), (at!(GdnHistory, tokens), h.tokens), (at!(GdnHistory, precision), h.precision)] {
        put(&mut r, o, &w.to_le_bytes());
    }
    out.extend_from_slice(&r);
}
// GdnChunkNext has no padding: padding can only follow a run of u32/f32 fields (before the
// next u64 or at the end), and each such run ends exactly where the next field or the struct
// starts. So its bytes are initialized words only.
const _: () = {
    use std::mem::{offset_of as at, size_of};
    assert!(at!(GdnChunkNext, z) == at!(GdnChunkNext, base) + 4 && at!(GdnChunkNext, x) == at!(GdnChunkNext, spare) + 4
        && at!(GdnChunkNext, ga) == at!(GdnChunkNext, x_start) + 4 && size_of::<GdnChunkNext>() == at!(GdnChunkNext, pk_row0) + 4);
};
fn chunk_bytes(v: &[GdnChunkNext]) -> &[u8] {
    // SAFETY: GdnChunkNext is repr(C) u64/u32/f32 fields without padding (asserted above).
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
/// Run a cohort verify layer's collected launches (cohort_kernels): upload the parameter
/// tables into `table` and launch every non-empty group with the chat in grid-Y, in the
/// per-chat order (rows, rest rows, history).
pub(crate) fn launch_cohort_scratch(b: &Backend<'_>, s: &GdnScratch<'_>, table: &mut DeviceVector<'_>) -> Result<()> {
    let Some(c) = s.cohort.borrow_mut().take() else { return Ok(()); };
    launch_cohort(b, table, c)
}
fn launch_cohort(b: &Backend<'_>, table: &mut DeviceVector<'_>, c: CohortLaunches) -> Result<()> {
    let total = (c.rows[0].len() + c.rows[1].len()) * std::mem::size_of::<GdnChunkNext>() + c.history.len() * std::mem::size_of::<GdnHistory>();
    if total == 0 { return Ok(()); }
    if total > table.len() * 4 { return Err("cohort GDN launches exceed their table".into()); }
    let mut bytes = Vec::with_capacity(total);
    c.rows.iter().for_each(|r| bytes.extend_from_slice(chunk_bytes(r)));
    c.history.iter().for_each(|h| history_record(h, &mut bytes));
    b.write_vector_bytes(&mut table.slice(0..total.div_ceil(4))?, &bytes)?;
    let mut at = table.device_address();
    let mut launch = |name: &str, blocks: u32, chats: usize, size: usize| -> Result<()> {
        if chats == 0 { return Ok(()); }
        let mut address = at;
        unsafe { b.launch_flash_rows(name, blocks, chats as u32, 128, 0, &mut [(&mut address as *mut u64).cast()])?; }
        at += (chats * size) as u64;
        Ok(())
    };
    for rows in &c.rows {
        let heads = rows.iter().map(|r| r.heads).max().unwrap_or(0);
        launch("flash_next_gdn_chunk_rows_cohort", heads, rows.len(), std::mem::size_of::<spark_gpu_probe::gdn_params::GdnChunkNext>())?;
    }
    let history_blocks = c.history.iter().map(|h| h.channels.div_ceil(128)).max().unwrap_or(0);
    launch("flash_gdn_history_cohort", history_blocks, c.history.len(), std::mem::size_of::<spark_gpu_probe::kernel_params::GdnHistory>())
}

/// flash_hybrid_copy rows per chat of a cohort verify layer (gdn::cohort_copies): entry
/// history, qkv tape, a tape, b tape.
pub(crate) const COPY_ROWS_PER_CHAT: usize = 4;
pub(crate) const COPY_ROW_WORDS: usize = std::mem::size_of::<spark_gpu_probe::kernel_params::HybridCopyRow>() / 4;
/// Upload `groups` of copy rows consecutively into `table` and launch flash_hybrid_copy once
/// per non-empty group (row descriptors in the grid's second dimension; word copies).
pub(crate) fn hybrid_copy_rows(b: &Backend<'_>, table: &mut DeviceVector<'_>,
    groups: &[&[spark_gpu_probe::kernel_params::HybridCopyRow]]) -> Result<()> {
    let all: Vec<_> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    if all.is_empty() { return Ok(()); }
    if all.len() * COPY_ROW_WORDS > table.len() { return Err("cohort copy rows exceed their table".into()); }
    let bytes = unsafe { std::slice::from_raw_parts(all.as_ptr().cast::<u8>(), std::mem::size_of_val(all.as_slice())) };
    b.write_vector_bytes(&mut table.slice(0..all.len() * COPY_ROW_WORDS)?, bytes)?;
    let mut first = 0;
    for group in groups.iter().filter(|g| !g.is_empty()) {
        let words = group.iter().map(|r| r.words).max().unwrap_or(0);
        let mut p = spark_gpu_probe::kernel_params::HybridCopy { source: std::ptr::null(), destination: std::ptr::null_mut(),
            words: 0, padding: 0, rows: table.device_address() + (first * COPY_ROW_WORDS * 4) as u64 };
        unsafe {
            b.launch_flash_rows("flash_hybrid_copy", words.div_ceil(128), group.len() as u32, 128, 0,
                &mut [(&mut p as *mut spark_gpu_probe::kernel_params::HybridCopy).cast()])?;
        }
        first += group.len();
    }
    Ok(())
}
/// Commit-time history rebuild (gdn_replay::table_commit): one GdnHistory per
/// GDN layer through the cohort history launch (launch_cohort), layer in grid-Y.
/// Each entry reads its entry history taps and tape rows and writes the
/// accepted history to its live history (history_token destination rule);
/// history_token overwrites the entry history first, so no reader may use
/// an entry history after this launch.
pub(crate) fn launch_history_table(b: &Backend<'_>, table: &mut DeviceVector<'_>,
    entries: Vec<spark_gpu_probe::kernel_params::GdnHistory>) -> Result<()> {
    launch_cohort(b, table, CohortLaunches { rows: [vec![], vec![]], history: entries })
}
/// Whether the process's PTX exports the cohort GDN entries (gdn::cohort_kernels).
pub(crate) fn cohort_kernel_entries(b: &Backend<'_>) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| ["flash_next_gdn_chunk_rows_cohort", "flash_gdn_history_cohort"]
        .iter().all(|name| b.has_flash(name)))
}

impl GdnScratch<'_> {
    /// Add to the launches a cohort verify layer is collecting (cohort_kernels).
    pub(super) fn collect(&self, add: impl FnOnce(&mut CohortLaunches)) -> Result<()> {
        add(self.cohort.borrow_mut().as_mut().ok_or("cohort GDN launches")?);
        Ok(())
    }
}

impl<'a> Gdn<'a, '_> {
    /// The entry conv-history snapshot scan_next copies, as one flash_hybrid_copy row (cohort_copies).
    pub(in crate::flash) fn history_copy_row(&self, entry: &GdnState<'a>) -> Result<spark_gpu_probe::kernel_params::HybridCopyRow> {
        if entry.layer != self.layer || entry.rank != self.rank || entry.history.len() != self.history.len() {
            return Err("GDN entry snapshot ownership differs".into());
        }
        Ok(spark_gpu_probe::kernel_params::HybridCopyRow { source: self.history.device_address(),
            destination: entry.history.device_address(), words: u32::try_from(self.history.len()).map_err(|_| "GDN history words")?, padding: 0 })
    }
    /// This layer's descriptor words (state, history, no snapshot) for a cohort table.
    pub(in crate::flash) fn descriptor_words(&self) -> [f32; 6] {
        let address = |a: u64| [f32::from_bits(a as u32), f32::from_bits((a >> 32) as u32)];
        let (s, h) = (address(self.state.device_address()), address(self.history.device_address()));
        [s[0], s[1], h[0], h[1], 0.0, 0.0]
    }
    /// The output gate of rows start..start + tokens in one launch (cohort verify;
    /// the same words as scan_next's per-call launches over those rows).
    pub(in crate::flash) fn gate_rows(&self, b: &Backend<'_>, s: &GdnScratch<'_>, start: usize, tokens: usize) -> Result<()> {
        use spark_gpu_probe::kernel_params::GdnGate;
        let hd = self.heads * DIM;
        let core = s.token_workspace(start, tokens)?[5];
        let at = |v: &DeviceVector<'_>| v.device_address() + (start * hd * 4) as u64;
        let mut gate = GdnGate { x: core as _, z: at(&s.z) as _, norm: s.norm_address() as _, output: at(&s.gated) as _,
            heads: (self.heads * tokens) as u32, tokens: 1, precision: 0, packed: 0, epsilon: self.epsilon, sessions: 0 };
        unsafe {
            b.launch_flash_shared("flash_gdn_gate", (self.heads * tokens * 32).div_ceil(128) as u32, 128, 0,
                &mut [(&mut gate as *mut GdnGate).cast()])
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    /// The cohort tables carry the history record's fields in declaration order at their repr(C)
    /// offsets and zero padding.
    #[test]
    fn cohort_history_record_is_the_field_layout_with_zero_padding() {
        use std::ptr::{without_provenance as c, without_provenance_mut as m};
        let h = GdnHistory { x: c(0x2001), history: m(0x2002), rows: 0x2003, channels: 31, tokens: 32, precision: 33 };
        let (mut got, mut want) = (Vec::new(), Vec::new());
        history_record(&h, &mut got);
        (0x2001u64..=0x2003).for_each(|a| want.extend_from_slice(&a.to_le_bytes()));
        (31u32..=33).for_each(|w| want.extend_from_slice(&w.to_le_bytes()));
        assert_eq!((want.len(), std::mem::size_of::<GdnHistory>()), (36, 40));
        want.resize(40, 0);
        assert_eq!(got, want);
    }
}
