//! The tensor split's head split: the verify head of a one-chat pass of
//! 2..=RETAINED_ROWS rows, split by vocabulary halves across the tensor split's two ranks.
//! Both ranks hold the pass's final residual (every split layer swaps its halves), mix it
//! (no head weight is read for the mix), and project their half of the head rows with the
//! whole head's launches (Head::project_next_rows_at: each logit is the whole head's). Rank 1
//! sends its argmax rows (logit, id, valid; 12 bytes per row) to rank 0 through the mailbox;
//! rank 0 keeps the larger logit per row, ties to the lower id (crate::flash::head::merge_split), which is
//! the whole head's argmax. Rank 1 does not know the request's sampling, so it projects its
//! half on every such pass; rank 0 uses it on greedy verification only (the private greedy
//! API, Workspace::bound_head) and otherwise runs the whole head and drops the message. The
//! session's logits stay sparse; full-logit readers re-project the retained inputs
//! (canonical_bound_logits), as after a bounded head pass.
use super::*;

/// Rows of rank 0's half: the lower half of the vocabulary in whole 8-row groups.
pub(super) fn split_rows(vocabulary: usize) -> usize {
    crate::flash_head_options::split_rows(vocabulary)
}

/// Both ranks see the same answer for the same pass: the tensor split, a verify
/// pass of one chat and 1..=RETAINED_ROWS rows that crosses in the mailbox.
pub(super) fn eligible(model: &NextModel<'_>, chats: usize, count: usize, verify: bool) -> bool {
    verify && chats == 1 && (1..=crate::flash_head_options::RETAINED_ROWS).contains(&count) && model.tp_on()
        && !model.tp_host_pass(count)
}

impl NextModel<'_> {
    /// Drafting sessions retain their verify rows' head inputs (80 KiB each) for full-logit
    /// readers: under the bounded head or the split head.
    pub(in super::super) fn retains_head_inputs(&self) -> bool {
        self.head_bounds.is_some()
            || crate::flash_tp_switch::on()
    }
}

impl<'a, 'w> NextSession<'a, 'w> {
    /// Rank 1, after its layers: the upper half of an eligible pass's head, then its argmax rows
    /// to rank 0.
    pub(super) fn split_head_back(b: &Backend<'a>, sessions: &mut [&mut Self], work: &mut Workspace<'a>, count: usize,
        verify: bool) -> Result<()> {
        let model = sessions[0].model;
        if !eligible(model, sessions.len(), count, verify) {
            return Ok(());
        }
        let (vocabulary, start) = (model.vocabulary(), split_rows(model.vocabulary()));
        sessions[0].mixer.mix_tiled(b, &mut work.hc, &work.residual, &work.mixed, count)?;
        Self::split_half(b, model, work, start, vocabulary - start, count)?;
        model.tp_rows_to_front(b, &mut work.head_result, count, 3)
    }
    /// This rank's half [start, start + rows) of the head for the pass's mixed rows: the bounded
    /// screen of that half when the rank loaded its bounds (exact rescoring of the rows that can
    /// win), else the half's LBF12 launches. Either way the argmax rows are the half's exact ones.
    fn split_half(b: &Backend<'a>, model: &NextModel<'a>, work: &mut Workspace<'a>, start: usize, rows: usize, count: usize)
        -> Result<()> {
        let weight = model.weights.device_address("lm_head.weight")?;
        match model.head_bounds.as_ref() {
            Some(bounds) => Head::project_bound_rows(b, weight, start, rows, count, &work.mixed, &mut work.head_logits,
                &mut work.head_partials, &mut work.head_result, bounds),
            None => Head::project_next_rows_at(b, weight, start, rows, count, &work.mixed, &mut work.head_logits,
                &mut work.head_partials, &mut work.head_result),
        }
    }
    /// Rank 0, an eligible greedy pass with the mix done: the lower half of the head, then rank 1's
    /// argmax rows after its own in the result buffer (merged on readback, split_results). False:
    /// the pass runs the whole head (split_head_after takes rank 1's message).
    pub(super) fn split_head_front(b: &Backend<'a>, sessions: &mut [&mut Self], work: &mut Workspace<'a>,
        spans: &[std::ops::Range<usize>], count: usize) -> Result<bool> {
        let model = sessions[0].model;
        let s = &mut *sessions[0];
        let retained = s.draft.as_ref().is_some_and(|d| d.bound_input.is_some());
        if !eligible(model, 1, count, true) || spans.len() != 1 || !work.bound_head || !retained {
            return Ok(false);
        }
        if work.head_result.len() < 2 * count * 3 {
            return Err("split head result rows".into());
        }
        let start = split_rows(model.vocabulary());
        Self::split_half(b, model, work, 0, start, count)?;
        model.tp_rows_to_front(b, &mut work.head_result.slice(count * 3..2 * count * 3)?, count, 3)?;
        let d = s.draft.as_mut().ok_or("MTP disabled")?;
        b.copy_range(&work.mixed, d.bound_input.as_mut().ok_or("missing retained head inputs")?, spans[0].start * H, 0, count * H)?;
        d.bound_at.set(Some((s.pos + count, count)));
        s.head.mark_split(start as u32);
        Ok(true)
    }
    /// Rank 0, after the whole head of an eligible pass it did not split: rank 1's message is taken
    /// past this pass's rows and dropped.
    pub(super) fn split_head_after(b: &Backend<'a>, model: &NextModel<'a>, chats: usize, work: &mut Workspace<'a>,
        count: usize) -> Result<()> {
        if !eligible(model, chats, count, true) {
            return Ok(());
        }
        if work.head_result.len() < 2 * count * 3 {
            return Err("split head result rows".into());
        }
        model.tp_rows_to_front(b, &mut work.head_result.slice(count * 3..2 * count * 3)?, count, 3)
    }
    /// The argmax rows of a verify pass (`rows` rows): merged with rank 1's when the pass split
    /// the head.
    pub(in super::super) fn split_results(b: &Backend<'a>, sessions: &mut [&mut Self], result: &DeviceVector<'a>, rows: usize)
        -> Result<Vec<(u32, f32)>> {
        let split = match sessions { [s] => s.head.take_split(), _ => None };
        let Some(start) = split else { return b.read_argmax_results(result, rows) };
        let both = b.read_argmax_results(result, 2 * rows)?;
        crate::flash::head::merge_split(&both[..rows], &both[rows..], start)
    }
}
