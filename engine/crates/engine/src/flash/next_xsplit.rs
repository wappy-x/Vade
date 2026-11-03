//! A cold prompt's sequence split over more than two ranks (flash/next_xsplit_plan.rs): each
//! rank walks its steps of one plan. Rank 0 keeps the head, all the draft rows and the first token;
//! afterwards it holds the whole state and owes the other ranks the push, as after the pair's
//! transaction (next_tp_seq.rs). A back holds the state of the layers it does not own only
//! during the split: rank 0's own part leaves it with the state after the last row.
use super::*;
use crate::flash_next_xsplit_plan::{Msg, Op, Plan, Shape};

/// A rank's funding: its bank, its workspace and the compact input of its tiles.
struct Funds<'a> {
    bank: Bank<'a>,
    work: Workspace<'a>,
    input: DeviceVector<'a>,
}
/// A rank's funding, all before its first stage: the KV pages of every row it writes, the state
/// of the layers it does not own, its bank, its input and its workspace. None: the rank refuses
/// and nothing ran.
fn funded<'a, 'w>(b: &Backend<'a>, chat: &mut NextSession<'a, 'w>, plan: &Plan, shape: &Shape, rank: usize) -> Option<Funds<'a>> {
    chat.fit_kv(b, if rank == 0 { plan.rows } else { plan.part(rank).end }).ok()?;
    chat.visit_layers(b).ok()?;
    // One row more than the part: the draft stage reads the row after its last draft row.
    let bank = Bank::new(b, plan.part(rank).len() + 1).ok()?;
    let input = b.vector_arena(&[plan.input_words(rank, shape)]).ok()?.into_iter().next()?;
    let work = chat.model.take_work(b, chat.capacity, plan.tile).ok()?;
    Some(Funds { bank, work, input })
}
/// Rank `rank`'s token ids from its part's first row through the row after it (the end of the
/// prompt on rank 0). A compact input starts with its rows' ids.
fn plan_ids(b: &Backend<'_>, input: &DeviceVector<'_>, plan: &Plan, shape: &Shape, rank: usize, next_word: usize, tokens: &[u32]) -> Result<Vec<u32>> {
    let part = plan.part(rank);
    if rank == 0 { return Ok(tokens[part].to_vec()); }
    let mut ids = Vec::with_capacity(part.len() + 1);
    for tile in 0..plan.tiles(rank) {
        let words = b.read_vector_range(&input.slice(plan.input(rank, tile, shape))?, 0, plan.span(rank, tile).len())?;
        ids.extend(words.iter().map(|word| word.to_bits()));
    }
    ids.push(b.read_vector_range(&input.slice(next_word..next_word + 1)?, 0, 1)?[0].to_bits());
    Ok(ids)
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
        let draft = self.draft.is_some();
        let (shape, world) = model.tp_xsplit_shape(self, draft)?;
        let Some(plan) = Plan::new(tokens.len(), world, tile, &shape) else {
            eprintln!("flashnext tp xsplit refused rows={} by=plan tile={tile}", tokens.len());
            return Ok(false);
        };
        let Some(Funds { bank, mut work, input }) = funded(b, self, &plan, &shape, 0) else {
            eprintln!("flashnext tp xsplit refused rows={} by=front", tokens.len());
            return Ok(false);
        };
        model.tp_split_start(self.slot);
        let asked = model.tp_split_post(self.slot, self.capacity, tokens.len(), tile, draft)
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
            .and_then(|()| Self::xsplit_steps(b, &mut work, &bank, &input, self, &plan, &shape, 0, tokens))
            .and_then(|()| finished(b, &work, model));
        model.return_work(work);
        result.map_err(|e| model.tp_split_fail(e))?;
        self.pos = tokens.len();
        self.sync_pool_position(b)?;
        self.valid = true;
        model.tp_xsplit_owed(self.slot, tokens.len());
        eprintln!("flashnext tp xsplit rows={} part={} tile={tile} ms={:.1}", tokens.len(), plan.part(0).len(), started.elapsed().as_secs_f64() * 1e3);
        Ok(true)
    }
    /// Ranks 1.., KIND_HANDOFF with a SPLIT_PROMPT length: this rank's steps of the plan of
    /// `rows` rows in `tile`-row tiles for this chat at position 0. `draft`: rank 0's chat has a
    /// drafter. Afterwards the chat stands at the rows' end and holds only part of them, until
    /// rank 0's push (no other pass comes first).
    pub(in crate::flash) fn xsplit_back(b: &Backend<'a>, chat: &mut Self, rows: usize, tile: usize, rank: usize, draft: bool) -> Result<()> {
        let model = chat.model;
        if chat.pos != 0 || rows > chat.capacity { return Err(format!("prompt split of {rows} rows at position {}", chat.pos)); }
        model.tp_split_begin(b)?;
        let (shape, world) = model.tp_xsplit_shape(chat, draft)?;
        let ready = Plan::new(rows, world, tile, &shape)
            .and_then(|p| funded(b, chat, &p, &shape, rank).map(|funds| (p, funds)));
        let agreed = model.tp_xsplit_agree(b, ready.is_some());
        let Some((plan, Funds { bank, mut work, input })) = ready else {
            chat.leave_layers();
            return agreed.map(|_| ());
        };
        if !agreed.as_ref().is_ok_and(|&all| all) {
            model.return_work(work);
            chat.leave_layers();
            return agreed.map(|_| ());
        }
        chat.valid = false;
        // The first part starts from a new sequence's PLE history; the later parts take theirs.
        let result = chat.ple.reset(b)
            .and_then(|()| work.status.clear(b, tile, 0))
            .and_then(|()| Self::xsplit_steps(b, &mut work, &bank, &input, chat, &plan, &shape, rank, &[]))
            .and_then(|()| finished(b, &work, model));
        model.return_work(work);
        result?;
        chat.leave_layers();
        chat.pos = rows;
        chat.sync_pool_position(b)?;
        chat.valid = true;
        chat.logits_ready = false;
        Ok(())
    }
    /// The state of the layers this rank does not own, for the split (fresh, as at position 0).
    fn visit_layers(&mut self, b: &Backend<'a>) -> Result<()> {
        let model = self.model;
        let layers: Vec<usize> = (0..LAYERS).filter(|&li| li % 4 != 3 && !model.owns_state(li)).collect();
        if layers.is_empty() { return Ok(()); }
        let mut fresh = if gdn::session_arena() { GdnState::fresh_arena(b, &model.weights, &layers)? } else { Vec::new() }.into_iter();
        for li in layers { self.gdn_state[li] = Some(Gdn::load_next(b, &model.weights, li, model.gdn_proj[li], fresh.next())?.into_state()); }
        Ok(())
    }
    /// The state of the layers this rank does not own goes: its rows are rank 0's now.
    fn leave_layers(&mut self) {
        let model = self.model;
        for (li, state) in self.gdn_state.iter_mut().enumerate() { if !model.owns_state(li) { *state = None; } }
    }
    /// One rank's steps. `tokens`: every row's token on rank 0 (the other ranks hash no row).
    #[allow(clippy::too_many_arguments)]
    fn xsplit_steps(b: &Backend<'a>, work: &mut Workspace<'a>, bank: &Bank<'a>, input: &DeviceVector<'a>, chat: &mut Self, plan: &Plan, shape: &Shape, rank: usize, tokens: &[u32]) -> Result<()> {
        let model = chat.model;
        let ops = plan.ops(shape);
        let (sent, got) = model.tp_xsplit_counts()?;
        let start = plan.part(rank).start;
        // The last word of the input vector, and the token ids at the part's rows and one past them.
        let next_word = plan.input_words(rank, shape) - 1;
        let mut ids = Vec::new();
        let own = |owner: usize, tile: usize| { let span = plan.span(owner, tile); span.start - plan.part(owner).start..span.end - plan.part(owner).start };
        let mut put = sent;
        for &op in &ops[rank] {
            match op {
                Op::Pack { owner, tile } => {
                    let packed = chat.pack_rows(&tokens[plan.span(owner, tile)])?;
                    if owner == 0 {
                        b.write_vector_bytes(&mut input.slice(plan.input(0, tile, shape))?, &packed)?;
                    } else {
                        b.write_vector_bytes(work.prompt_packed.as_mut().ok_or("missing packed prompt workspace")?, &packed)?;
                    }
                }
                Op::Enter { layer } => model.tp_xsplit_enter(b, chat, layer, start)?,
                Op::Stage { layer, tile } => {
                    let span = plan.span(rank, tile);
                    if layer == 0 {
                        let entered = bank.enter(work, own(rank, tile))?;
                        let ran = Self::frontend_rows(b, work, model, input.slice(plan.input(rank, tile, shape))?.device_address(), span.len());
                        Bank::leave(b, work, entered);
                        ran?;
                    }
                    Self::tile_stage(b, work, bank, chat, own(rank, tile), span.start, layer, LAYERS, layer == 0)?;
                }
                Op::Put(msg) => {
                    put += 1;
                    match msg {
                        Msg::LayerPad { .. } | Msg::SeedPad { .. } => model.tp_xsplit_pad(b, put)?,
                        Msg::Seed { words, .. } => model.tp_xsplit_put(b, work.prompt_packed.as_ref().ok_or("missing packed prompt workspace")?, words, 0, put)?,
                        Msg::State { layer, k, from, words } => model.tp_xsplit_state_put(b, chat, layer, k, words, from, put)?,
                        Msg::Next { owner } => {
                            let mut word = input.slice(next_word..next_word + 1)?;
                            b.write_vector(&mut word, &[f32::from_bits(tokens[plan.part(owner).end])])?;
                            model.tp_xsplit_put(b, &word, 1, 0, put)?;
                        }
                        Msg::Draft { k, from, words, .. } => model.tp_xsplit_draft_put(b, chat, k, words, from, put)?,
                    }
                }
                Op::Take { origin, at, msg } => {
                    let seq = got[origin] + at;
                    match msg {
                        Msg::LayerPad { .. } | Msg::SeedPad { .. } => model.tp_xsplit_drop(b, origin, seq)?,
                        Msg::Seed { owner, tile, words } => model.tp_xsplit_take(b, &mut input.slice(plan.input(owner, tile, shape))?, words, 0, origin, seq)?,
                        Msg::State { layer, k, from, words } => model.tp_xsplit_state_take(b, chat, layer, k, words, from, origin, seq)?,
                        Msg::Next { .. } => model.tp_xsplit_take(b, &mut input.slice(next_word..next_word + 1)?, 1, 0, origin, seq)?,
                        Msg::Draft { k, from, words, .. } => model.tp_xsplit_draft_take(b, chat, k, words, from, origin, seq)?,
                    }
                }
                Op::Drop { origin, at } => model.tp_xsplit_drop(b, origin, got[origin] + at)?,
                Op::Observe { tile } => {
                    if tile == 0 {
                        let draft = chat.draft.as_mut().ok_or("the split's plan has a drafter and the chat has none")?;
                        draft.start_at(b, &bank.residual(0..1)?)?;
                    }
                    if ids.is_empty() { ids = plan_ids(b, input, plan, shape, rank, next_word, tokens)?; }
                    let seen = plan.observed(rank, tile);
                    let rows = seen.start - start..seen.end - start;
                    chat.split_observe(b, work, &bank.residual(rows.clone())?, seen.start, &ids[rows])?;
                }
            }
        }
        model.tp_xsplit_end(&Plan::puts(&ops));
        Ok(())
    }
}
