//! A cold prompt's sequence split over more than two ranks (flash/next_xsplit_plan.rs): each
//! rank walks its steps of one plan. Rank 0 keeps the head, the drafter and the first token;
//! afterwards it holds the whole state and owes the other ranks the push, as after the pair's
//! transaction (next_tp_seq.rs).
use super::*;
use crate::flash_next_xsplit_plan::{Msg, Op, Plan, Shape};

/// A rank's funding, all before its first stage: the KV pages of every row it writes, its bank
/// and its workspace. None: the rank refuses and nothing ran.
fn funded<'a, 'w>(b: &Backend<'a>, chat: &mut NextSession<'a, 'w>, kv_rows: usize, bank_rows: usize, tile: usize) -> Option<(Bank<'a>, Workspace<'a>)> {
    chat.fit_kv(b, kv_rows).ok()?;
    let bank = Bank::new(b, bank_rows).ok()?;
    let work = chat.model.take_work(b, chat.capacity, tile).ok()?;
    Some((bank, work))
}
/// Whether a stream that ran its steps finished with no failed row and no failed mailbox.
fn finished(b: &Backend<'_>, work: &Workspace<'_>, model: &NextModel<'_>) -> Result<()> {
    b.synchronize()?;
    if work.status.failed_after_readback()? || model.tp_failed() {
        return Err(if model.tp_failed() { "tensor split mailbox timed out or its proxy failed" } else { "Flash-Next device routing rejected a row; reset cohort" }.into());
    }
    Ok(())
}

impl<'a, 'w> NextSession<'a, 'w> {
    /// Rank 0, a cold prompt's first `tokens.len()` rows over every rank in `tile`-row tiles.
    /// Ok(false): the plan refused the shape or a rank could not fund it, nothing ran and the
    /// prompt takes the usual path.
    pub(super) fn xsplit_front(&mut self, b: &Backend<'a>, tokens: &[u32], tile: usize) -> Result<bool> {
        let model = self.model;
        let started = std::time::Instant::now();
        let (shape, world) = model.tp_xsplit_shape(self)?;
        let Some(plan) = Plan::new(tokens.len(), world, tile, &shape) else {
            eprintln!("flashnext tp xsplit refused rows={} by=plan tile={tile}", tokens.len());
            return Ok(false);
        };
        let Some((bank, mut work)) = funded(b, self, tokens.len(), plan.part(0).len(), tile) else {
            eprintln!("flashnext tp xsplit refused rows={} by=front", tokens.len());
            return Ok(false);
        };
        model.tp_split_start(self.slot);
        let asked = model.tp_split_post(self.slot, self.capacity, tokens.len(), tile)
            .and_then(|()| model.tp_split_begin(b))
            .and_then(|()| model.tp_xsplit_agree(b, true));
        match asked {
            Ok(true) => (),
            Ok(false) => {
                eprintln!("flashnext tp xsplit refused rows={} by=back", tokens.len());
                model.return_work(work);
                return Ok(false);
            }
            Err(error) => { model.return_work(work); return Err(model.tp_split_fail(error)); }
        }
        self.valid = false;
        self.logits_ready = false;
        if let Some(d) = self.draft.as_ref() { d.bound_at.set(None); }
        self.head.take_split();
        let result = work.status.clear(b, tile, 0)
            .and_then(|()| Self::xsplit_steps(b, &mut work, &bank, self, &plan, &shape, 0, tokens))
            .and_then(|()| finished(b, &work, model));
        model.return_work(work);
        result.map_err(|e| model.tp_split_fail(e))?;
        self.pos = tokens.len();
        self.sync_pool_position(b)?;
        self.valid = true;
        model.tp_split_owed(self.slot, tokens.len());
        eprintln!("flashnext tp xsplit rows={} part={} tile={tile} ms={:.1}", tokens.len(), plan.part(0).len(), started.elapsed().as_secs_f64() * 1e3);
        Ok(true)
    }
    /// Ranks 1.., KIND_HANDOFF with a SPLIT_PROMPT length: this rank's steps of the plan of
    /// `rows` rows in `tile`-row tiles for this chat at position 0. Afterwards the chat stands at
    /// the rows' end and holds only part of them, until rank 0's push (no other pass comes first).
    pub(in crate::flash) fn xsplit_back(b: &Backend<'a>, chat: &mut Self, rows: usize, tile: usize, rank: usize) -> Result<()> {
        let model = chat.model;
        if chat.pos != 0 || rows > chat.capacity { return Err(format!("prompt split of {rows} rows at position {}", chat.pos)); }
        model.tp_split_begin(b)?;
        let (shape, world) = model.tp_xsplit_shape(chat)?;
        let ready = Plan::new(rows, world, tile, &shape)
            .and_then(|p| funded(b, chat, p.part(rank).end, p.part(rank).len(), tile).map(|(bank, work)| (p, bank, work)));
        let agreed = model.tp_xsplit_agree(b, ready.is_some());
        let Some((plan, bank, mut work)) = ready else { return agreed.map(|_| ()) };
        if !agreed.as_ref().is_ok_and(|&all| all) {
            model.return_work(work);
            return agreed.map(|_| ());
        }
        chat.valid = false;
        let result = work.status.clear(b, tile, 0)
            .and_then(|()| Self::xsplit_steps(b, &mut work, &bank, chat, &plan, &shape, rank, &[]))
            .and_then(|()| finished(b, &work, model));
        model.return_work(work);
        result?;
        chat.pos = rows;
        chat.sync_pool_position(b)?;
        chat.valid = true;
        chat.logits_ready = false;
        Ok(())
    }
    /// One rank's steps. `tokens`: every row's token on rank 0 (the other ranks run no frontend).
    #[allow(clippy::too_many_arguments)]
    fn xsplit_steps(b: &Backend<'a>, work: &mut Workspace<'a>, bank: &Bank<'a>, chat: &mut Self, plan: &Plan, shape: &Shape, rank: usize, tokens: &[u32]) -> Result<()> {
        let model = chat.model;
        let ops = plan.ops(shape);
        let (sent, got) = model.tp_xsplit_counts()?;
        let start = plan.part(rank).start;
        let own = |owner: usize, tile: usize| { let span = plan.span(owner, tile); span.start - plan.part(owner).start..span.end - plan.part(owner).start };
        let mut put = sent;
        for &op in &ops[rank] {
            match op {
                Op::Lead { owner, tile } => {
                    let span = plan.span(owner, tile);
                    chat.lead(b, work, None, span.clone(), &tokens[span])?;
                }
                Op::Head { tile } => {
                    let span = plan.span(0, tile);
                    chat.lead(b, work, Some((bank, start)), span.clone(), &tokens[span])?;
                }
                Op::Enter { layer } => model.tp_xsplit_enter(b, chat, layer, start)?,
                Op::Stage { layer, tile } => {
                    let span = plan.span(rank, tile);
                    Self::tile_stage(b, work, bank, chat, own(rank, tile), span.start, layer, LAYERS, rank != 0 && layer == SPLIT)?;
                }
                Op::Put(msg) => {
                    put += 1;
                    match msg {
                        Msg::LayerPad { .. } | Msg::SeedPad { .. } | Msg::RowsPad { .. } => model.tp_xsplit_pad(b, put)?,
                        Msg::Seed { from, words, .. } => model.tp_xsplit_put(b, &work.residual, words, from, put)?,
                        Msg::State { layer, k, from, words } => model.tp_xsplit_state_put(b, chat, layer, k, words, from, put)?,
                        Msg::Rows { owner, tile, from, words } => model.tp_xsplit_put(b, &bank.residual(own(owner, tile))?, words, from, put)?,
                    }
                }
                Op::Take { origin, at, msg } => {
                    let seq = got[origin] + at;
                    match msg {
                        Msg::LayerPad { .. } | Msg::SeedPad { .. } | Msg::RowsPad { .. } => model.tp_xsplit_drop(b, origin, seq)?,
                        Msg::Seed { owner, tile, from, words } => model.tp_xsplit_take(b, &mut bank.residual(own(owner, tile))?, words, from, origin, seq)?,
                        Msg::State { layer, k, from, words } => model.tp_xsplit_state_take(b, chat, layer, k, words, from, origin, seq)?,
                        Msg::Rows { owner, tile, from, words } => {
                            work.residual.resize_within_capacity(plan.span(owner, tile).len() * W)?;
                            model.tp_xsplit_take(b, &mut work.residual, words, from, origin, seq)?;
                        }
                    }
                }
                Op::Drop { origin, at } => model.tp_xsplit_drop(b, origin, got[origin] + at)?,
                Op::Observe { owner, tile } => {
                    let span = plan.span(owner, tile);
                    let hidden = if owner == 0 { bank.residual(own(0, tile))? } else { work.residual.slice(0..span.len() * W)? };
                    chat.split_observe(b, work, &hidden, span.start, &tokens[span])?;
                }
            }
        }
        model.tp_xsplit_end(&Plan::puts(&ops));
        Ok(())
    }
}
