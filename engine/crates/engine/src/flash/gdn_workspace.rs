//! Next-only GDN workspace sharing, separate from recurrence execution.
use super::*;

impl<'a, 'w> Gdn<'a, 'w> {
    /// Ordered mixed-checkpoint rows need full token-major Q/K/V, without
    /// chunk64's triangular-system workspaces or its 64-token constructor cap.
    pub(in crate::flash) fn next_scratch(b: &Backend<'a>, w: &Weights<'_>, rows: usize, hc_normalized: &DeviceVector<'a>) -> Result<GdnScratch<'a>> {
        let heads = (w.plan.value_heads.end-w.plan.value_heads.start) as usize;
        let keys = (w.plan.key_heads.end-w.plan.key_heads.start) as usize;
        let mut lengths = spark_gpu_probe::next_math::gdn_lengths(HIDDEN, heads, keys, rows);
        if chunk_form() {
            // Chunk-form workspace: gamma, A, P per (launch chunk, head) and the
            // first chunk's tail key rows (cumulative, system, attention, decayed_keys).
            let chunks = rows.min(CHUNK_PIECE) / CHUNK + 2;
            lengths[14] = chunks * heads * CHUNK;
            lengths[15] = chunks * heads * CHUNK * CHUNK;
            lengths[16] = chunks * heads * CHUNK * CHUNK;
            lengths[18] = keys * CHUNK * DIM;
        }
        // HC-up has consumed normalized before project_next writes qkv. All GDN
        // consumers finish before the following HC mix. Pooled steps have the same
        // order; rollback temporarily swaps in its retained tape and restores this
        // owning view. The final head and draft observe run after target attention.
        let qkv = crate::flash_next_workspace::qkv_view(hc_normalized, lengths[0])?;
        let lengths = crate::flash_next_workspace::gdn_owned_lengths(lengths);
        let mut scratch = GdnScratch::allocate(b, HIDDEN, heads, keys, rows, &[0.0; DIM], lengths)?;
        scratch.qkv = qkv;
        // Next's finish_next and pool_next write the caller's block, never output.
        Ok(scratch)
    }
}
