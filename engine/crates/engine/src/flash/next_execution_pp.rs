//! Cohort rows that cross the layer pipeline: the front's kept rows for the
//! drafters and the back's layers [split, 48) for the front's rows.
use super::*;

impl<'a, 'w> NextSession<'a, 'w> {
    /// Pipeline front: the chat's kept rows (its queued prompt tiles) enter its draft
    /// history before any newer row, on the pooled path as on the tile path (backlog first).
    pub(in crate::flash) fn ingest_kept(&mut self, b: &Backend<'a>) -> Result<()> {
        if self.draft.is_none() || !self.model.pp_has_kept(self.slot) {
            return Ok(());
        }
        let model = self.model;
        let mut work = model.take_work(b, self.capacity, crate::flash_next_rows::prompt_tile())?;
        let d = self.draft.as_mut().unwrap();
        let result = Self::flush_into(b, model, d, &mut work)
            .and_then(|()| Self::observe_kept(b, model, d, &mut work, self.slot, self.pos, self.batch_drafts, None));
        model.return_work(work);
        result
    }
    /// Pipeline front, before a prefix capture drains the link: the kept rows of the chat's
    /// queued prompt tiles enter its draft history as each tile's ack comes back, while the
    /// back still runs the later tiles (as on the final tile, pp_return_and_observe), so the
    /// drain waits only for the last tiles. The same rows, order and chunks as ingest_kept.
    pub(in crate::flash) fn observe_queued_kept(&mut self, b: &Backend<'a>) -> Result<()> {
        if self.draft.is_none() || !self.model.pp_inline_kept() || !self.model.pp_has_kept(self.slot) {
            return Ok(());
        }
        let (model, slot, pos, batch) = (self.model, self.slot, self.pos, self.batch_drafts);
        let mut work = model.take_work(b, self.capacity, crate::flash_next_rows::prompt_tile())?;
        let d = self.draft.as_mut().unwrap();
        let result = Self::flush_into(b, model, d, &mut work).and_then(|()| loop {
            Self::observe_kept(b, model, d, &mut work, slot, pos, batch, None)?;
            if !model.pp_has_kept(slot) || !model.pp_pull_kept_ack(0)? { break Ok(()); }
        });
        model.return_work(work);
        result
    }
    /// Pipeline front: kept rows of the chat's queued prompt tiles enter its draft
    /// history first, in order, as world 1 observes each tile; uploaded through the MoE output
    /// buffer (idle once the layers ran) in chunks of the workspace's rows. `room`: only the
    /// oldest tiles until that many more rows fit in KEEP_ROWS (Link::take_kept_room).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_kept(b: &Backend<'a>, model: &NextModel<'a>, d: &mut mtp::Draft<'a, 'w>, work: &mut Workspace<'a>,
        slot: u32, pos: usize, batch: bool, room: Option<usize>) -> Result<()> {
        let kept = model.pp_take_kept(slot, room);
        if kept.is_empty() {
            return Ok(());
        }
        let (chunk, len, mut next) = (work.rows.max(1), work.expert_out.len(), 0);
        for (at, tokens, rows) in &kept {
            if *at < next || at + tokens.len() > pos {
                engine_model::events::note_line("pp_rows_not_fed", &format!("flashnext pp kept rows of slot {slot} out of order ({at} + {} after {next}, chat at {pos}); not fed", tokens.len()));
                break;
            }
            for c in (0..tokens.len()).step_by(chunk) {
                let n = (tokens.len() - c).min(chunk);
                work.expert_out.resize_within_capacity(n * W)?;
                b.write_vector_range(&mut work.expert_out, 0, &rows[c * W..(c + n) * W])?;
                if batch || (n > 1 && super::mtp::tile_all()) {
                    d.observe_rows(b, model, &work.mixed16, &mut work.hc, &mut work.qsa, at + c, &tokens[c..c + n],
                        &work.expert_out, 0)?;
                } else {
                    d.observe(b, model, at + c, &tokens[c..c + n], &work.expert_out, 0)?;
                }
            }
            next = at + tokens.len();
        }
        work.expert_out.resize_within_capacity(len)?;
        // The uploads read `kept` (host memory) until they complete.
        b.synchronize()
    }
    /// Pipeline front, a kept tile of `count` rows past KEEP_ROWS: read the acks already on the
    /// socket, and let each drafter observe only the oldest kept rows that came back until the
    /// tile fits, in order (the rows, chunks and order the synchronous path observes after its
    /// reply). The back keeps its queue, and the rest waits for the final tile, where this rank
    /// is idle while the back runs the last tiles.
    pub(super) fn observe_returned(b: &Backend<'a>, model: &NextModel<'a>, sessions: &mut [&mut Self], work: &mut Workspace<'a>, count: usize) -> Result<()> {
        model.pp_read_acks()?;
        for s in sessions.iter_mut() {
            if let Some(d) = s.draft.as_mut() {
                Self::flush_into(b, model, d, work)?;
                Self::observe_kept(b, model, d, work, s.slot, s.pos, s.batch_drafts, Some(count))?;
            }
        }
        Ok(())
    }
    /// Final prompt tile. On a direct last rank, kept rows arrive per tile during the wait
    /// and are observed while the back still runs the later tiles. Other pipelines keep one
    /// exchange and observe afterwards.
    pub(super) fn pp_return_and_observe(b: &Backend<'a>, model: &NextModel<'a>, sessions: &mut [&mut Self], work: &mut Workspace<'a>, output: bool, verify: bool, meta: &[[u32; 4]], count: usize) -> Result<()> {
        if !(output && !verify && model.pp_async() && model.pp_inline_kept()) {
            return model.pp_return_rows(b, output, verify, meta, &mut work.residual, count);
        }
        model.pp_post_final(b, meta, &work.residual, count)?;
        // The final reply is outstanding now: any error below must poison the link,
        // not leave the reply to be misread as a later tile's ack.
        let observed = (|| -> Result<()> {
            for s in sessions.iter_mut() {
                if let Some(d) = s.draft.as_mut() {
                    Self::flush_into(b, model, d, work)?;
                }
            }
            while model.pp_pull_kept_ack(1)? {
                for s in sessions.iter_mut() {
                    if s.draft.is_some() {
                        let d = s.draft.as_mut().unwrap();
                        Self::observe_kept(b, model, d, work, s.slot, s.pos, s.batch_drafts, None)?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(e) = observed {
            return Err(model.pp_fail_after_final(e));
        }
        model.pp_collect_final(b, &mut work.residual, count)
    }
    /// Pipeline back: layers [split, 48) for the front's prompt rows (the residual
    /// arrives at layer `split` and leaves after layer 47). Positions advance like the front's.
    pub(crate) fn pp_back_rows(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        lens: &[usize],
        residual: &mut [f32],
        verify: bool,
    ) -> Result<()> {
        let model = sessions[0].model;
        let split = model.tp_back_split().unwrap_or(model.pp_split.get());
        let owned: Vec<Vec<u32>> = lens.iter().map(|&n| vec![0u32; n]).collect();
        let tokens: Vec<&[u32]> = owned.iter().map(|t| t.as_slice()).collect();
        let spans = crate::flash_next_rows::validate(
            &sessions.iter().map(|s| s.pos).collect::<Vec<_>>(),
            &tokens,
            &sessions.iter().map(|s| s.capacity).collect::<Vec<_>>(),
            model.checkpoint.config.vocab,
        )?;
        let count = spans.last().unwrap().end;
        if residual.len() != count * W
            || count > crate::flash_next_rows::prompt_tile()
            || !(2..LAYERS).contains(&split)
        {
            return Err("pipeline back rows outside the prompt tile".into());
        }
        for s in sessions.iter() {
            crate::flash_next_rows::check_owners(&s.gdn_state, &s.qsa_state, s.model.state_range())?;
        }
        if verify {
            for (s, &n) in sessions.iter_mut().zip(lens) {
                s.prepare_back_verify(b, n)?;
            }
        }
        model.tp_gdn_frame(b, sessions, count)?;
        let owner = crate::flash_next_rows::workspace_owner(sessions.iter().map(|s| s.capacity))?;
        let capacity = sessions[owner].capacity;
        let mut work = model.take_work(b, capacity, count)?;
        if let Err(error) = work.status.clear(b, count, 0) {
            model.return_work(work);
            return Err(error);
        }
        for s in sessions.iter_mut() {
            s.valid = false;
            s.logits_ready = false;
            if let Some(d) = s.draft.as_ref() { d.bound_at.set(None); }
        }
        // Verify rows run the same captured segments as world one (VADE_MTP_GRAPH); so do the
        // tensor split's plain one-token rows (flash_tp_switch::plain_segments), as on rank 0.
        let mut result = if verify && super::mtp_graph() || crate::flash_tp_switch::plain_segments(model.tp_on(), verify, b.decode_readback_active(), lens.iter().copied()) {
            Self::execute::<true>(b, &mut work, sessions, &tokens, &spans, false, verify, Part::Back(split, &*residual))
                .inspect_err(|_| b.abort_segment())
        } else {
            Self::execute::<false>(b, &mut work, sessions, &tokens, &spans, false, verify, Part::Back(split, &*residual))
        };
        if result.is_ok() {
            result = b.synchronize().and_then(|()| match work.status.failed_after_readback() {
                Ok(true) => Err("Flash-Next device routing rejected a row; reset cohort".into()),
                Ok(false) if model.tp_failed() => Err("tensor split mailbox timed out or its proxy failed".into()),
                Ok(false) => Ok(()),
                Err(error) => Err(error),
            });
        }
        if result.is_ok() {
            result = b.read_vector_range_into(&work.residual, 0, residual);
        }
        model.return_work(work);
        if result.is_ok() {
            for (s, &n) in sessions.iter_mut().zip(lens) {
                s.pos += n;
                if verify {
                    s.sync_pool_state(b)?;
                } else {
                    s.sync_pool_position(b)?;
                }
                s.valid = true;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    /// Verify segments start every VERIFY_SEGMENT layers from layer 1 in their own slots; the
    /// tensor split adds a start at layer 3, after which the slots shift by one.
    #[test]
    fn verify_segment_starts_and_slots() {
        let starts = |split| (0..48).filter_map(|li| super::super::segment_start(li, split).map(|s| (li, s))).collect::<Vec<_>>();
        assert_eq!(starts(false), [(1, 0), (13, 1), (25, 2), (37, 3)]);
        assert_eq!(starts(true), [(1, 0), (3, 1), (13, 2), (25, 3), (37, 4)]);
    }
    /// A plain shape never reuses a verify shape's executables of the same rows and chats.
    #[test]
    fn plain_and_verify_shapes_differ() {
        use super::super::VerifyShape;
        let shape = |verify| VerifyShape { verify, rows: 1, chats: 1, shallow: 0 };
        assert!(shape(false) != shape(true));
        assert!(shape(false) == shape(false));
    }
}
