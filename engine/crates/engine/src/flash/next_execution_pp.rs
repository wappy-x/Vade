//! VADE_W2_PP=1 cohort rows that cross the layer pipeline: the front's kept rows for the
//! drafters (VADE_W2_PP_MTP=1) and the back's layers [split, 48) for the front's rows.
use super::*;

impl<'a, 'w> NextSession<'a, 'w> {
    /// VADE_W2_PP_MTP=1 front: the chat's kept rows (its queued prompt tiles) enter its draft
    /// history before any newer row, on the pooled path as on the tile path (backlog first).
    pub(super) fn ingest_kept(&mut self, b: &Backend<'a>) -> Result<()> {
        if self.draft.is_none() || !self.model.pp_has_kept(self.slot) {
            return Ok(());
        }
        let model = self.model;
        let mut work = model.take_work(b, self.capacity, crate::flash_next_rows::prompt_tile())?;
        let d = self.draft.as_mut().unwrap();
        let result = Self::flush_into(b, model, d, &mut work)
            .and_then(|()| Self::observe_kept(b, model, d, &mut work, self.slot, self.pos, self.batch_drafts));
        model.return_work(work);
        result
    }
    /// VADE_W2_PP_MTP=1 front: kept rows of the chat's queued prompt tiles enter its draft
    /// history first, in order, as world 1 observes each tile; uploaded through the MoE output
    /// buffer (idle once the layers ran) in chunks of the workspace's rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_kept(b: &Backend<'a>, model: &NextModel<'a>, d: &mut mtp::Draft<'a, 'w>, work: &mut Workspace<'a>,
        slot: u32, pos: usize, batch: bool) -> Result<()> {
        let kept = model.pp_take_kept(slot);
        if kept.is_empty() {
            return Ok(());
        }
        let (chunk, len, mut next) = (work.rows.max(1), work.expert_out.len(), 0);
        for (at, tokens, rows) in &kept {
            if *at < next || at + tokens.len() > pos {
                eprintln!("flashnext pp kept rows of slot {slot} out of order ({at} + {} after {next}, chat at {pos}); not fed", tokens.len());
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
            while model.pp_pull_kept_ack()? {
                for s in sessions.iter_mut() {
                    if s.draft.is_some() {
                        let d = s.draft.as_mut().unwrap();
                        Self::observe_kept(b, model, d, work, s.slot, s.pos, s.batch_drafts)?;
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
    /// VADE_W2_PP=1 back: layers [split, 48) for the front's prompt rows (the residual
    /// arrives at layer `split` and leaves after layer 47). Positions advance like the front's.
    pub(crate) fn pp_back_rows(
        b: &Backend<'a>,
        sessions: &mut [&mut Self],
        lens: &[usize],
        residual: &mut [f32],
        verify: bool,
    ) -> Result<()> {
        let model = sessions[0].model;
        let split = model.pp_split.get();
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
            crate::flash_next_rows::check_owners(&s.gdn_state, &s.qsa_state)?;
        }
        if verify {
            for (s, &n) in sessions.iter_mut().zip(lens) {
                s.prepare_back_verify(b, n)?;
            }
        }
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
        let mut result =
            Self::execute::<false>(b, &mut work, sessions, &tokens, &spans, false, verify, Part::Back(split, &*residual));
        if result.is_ok() {
            result = b.synchronize().and_then(|()| match work.status.failed_after_readback() {
                Ok(true) => Err("Flash-Next device routing rejected a row; reset cohort".into()),
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
