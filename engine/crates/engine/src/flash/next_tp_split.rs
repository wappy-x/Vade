//! Mailbox steps of the layer-major prompt transaction (flash/next_split_plan.rs wire), with
//! the token split's credit rule: each data message counts in `Seq::msgs`; before message m with
//! m >= SLOTS the sender takes the pad of message m - SLOTS; the receiver puts a pad for each
//! data message it takes. Layer state crosses with tp_seq_state, the last credits with
//! tp_seq_credits. Rows cross as slot-sized pieces of one vector's words, in order.
use super::*;

impl<'a> NextModel<'a> {
    /// A transaction starts: no data message of this rank is unconfirmed (the earlier passes
    /// took their credits), so the count starts again; the pads exist.
    pub(in crate::flash) fn tp_split_begin(&self, b: &Backend<'a>) -> Result<()> {
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        if s.seq.pads.is_none() { s.seq.pads = Some([b.vector(1)?, b.vector(1)?]); }
        s.seq.msgs = 0;
        Ok(())
    }
    /// Put every word of `v` as slot-sized data messages.
    pub(in crate::flash) fn tp_split_put(&self, b: &Backend<'a>, v: &DeviceVector<'a>) -> Result<()> {
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let slot = mailbox.slot_bytes() / 4;
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        let (mut sent, mut taken, mut msgs, limit) = (s.sent, s.got[s.peer], s.seq.msgs, s.limit_ns);
        let [_, pad_in] = s.seq.pads.as_mut().ok_or("token split pads missing")?;
        for from in (0..v.len()).step_by(slot) {
            if msgs >= TP_MAILBOX_SLOTS {
                taken += 1;
                b.tp_mailbox_take(mailbox, pad_in, 1, 1, 1, 0, taken, limit)?;
            }
            sent += 1;
            b.tp_mailbox_put(mailbox, v, 1, v.len(), slot.min(v.len() - from), from, sent)?;
            msgs += 1;
        }
        (s.sent, s.got[s.peer], s.seq.msgs) = (sent, taken, msgs);
        Ok(())
    }
    /// Take every word of `v` as slot-sized data messages, each followed by its credit pad.
    pub(in crate::flash) fn tp_split_take(&self, b: &Backend<'a>, v: &mut DeviceVector<'a>) -> Result<()> {
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let slot = mailbox.slot_bytes() / 4;
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        let (mut sent, mut taken, limit) = (s.sent, s.got[s.peer], s.limit_ns);
        let [pad, _] = s.seq.pads.as_mut().ok_or("token split pads missing")?;
        let len = v.len();
        for from in (0..len).step_by(slot) {
            taken += 1;
            b.tp_mailbox_take(mailbox, v, 1, len, slot.min(len - from), from, taken, limit)?;
            sent += 1;
            b.tp_mailbox_put(mailbox, pad, 1, 1, 1, 0, sent)?;
        }
        (s.sent, s.got[s.peer]) = (sent, taken);
        Ok(())
    }
    /// The front's notice pad outside the credit count (`put`), or the worker's take of it; the
    /// pads outlive the queued operation.
    pub(in crate::flash) fn tp_split_notice(&self, b: &Backend<'a>, put: bool) -> Result<()> {
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        let [pad, pad_in] = s.seq.pads.as_mut().ok_or("token split pads missing")?;
        if put {
            s.sent += 1;
            b.tp_mailbox_put(mailbox, pad, 1, 1, 1, 0, s.sent)
        } else {
            s.got[s.peer] += 1;
            b.tp_mailbox_take(mailbox, pad_in, 1, 1, 1, 0, s.got[s.peer], s.limit_ns)
        }
    }
    /// The worker's funding answer, one word outside the credit count (`put`), or the front's
    /// take of it into `word`; `word` must live until the stream passed the operation.
    pub(in crate::flash) fn tp_split_word(&self, b: &Backend<'a>, word: &mut DeviceVector<'a>, put: bool) -> Result<()> {
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        if put {
            s.sent += 1;
            b.tp_mailbox_put(mailbox, word, 1, 1, 1, 0, s.sent)?;
        } else {
            s.got[s.peer] += 1;
            b.tp_mailbox_take(mailbox, word, 1, 1, 1, 0, s.got[s.peer], s.limit_ns)?;
        }
        Ok(())
    }
}

impl<'a> NextModel<'a> {
    /// Rank 0: the rows of a whole cold one-chat prompt call that take the transaction (the
    /// call's first `rows` rows), or None. A valid session that owns its state (a failed one keeps
    /// the usual path, which refuses it until reset), no pending token-split export, no dump, and
    /// one attention selection for every row (prefill_inner's one-row rule never cuts its tiles).
    pub(in crate::flash) fn tp_split_cut(&self, chat: &NextSession<'a, '_>, output: bool, rows: usize, chunk: usize) -> Option<usize> {
        let owns = crate::flash_next_rows::check_owners(&chat.gdn_state, &chat.qsa_state, self.state_range()).is_ok();
        // The transaction runs on the pair only; one Spark (no pipeline queue) stops at pp_async first.
        if !crate::flash_tp_switch::split_session(output, chat.valid, chat.pos, chat.dump.is_some() || chat.layer_bytes.is_some(), owns)
            || !self.pp_async() || crate::flash_next_rows::prompt_tile() < crate::flash_tp_switch::PIPELINE_TILE || !self.tp_pair() { return None; }
        let n = crate::flash_tp_switch::split_rows(0, rows, chunk)?;
        self.tp.borrow().as_ref().filter(|s| s.rank == 0 && s.seq.chats.pending().is_none())?;
        (3..LAYERS).step_by(4).all(|li| (1..n).all(|i| chat.indexer.select(li, i, i + 1) == chat.indexer.select(li, 0, 1))).then_some(n)
    }
    /// Rank 0: ask the next rank to run its side of a `rows`-row transaction for this chat (a
    /// handoff request with the SPLIT_PROMPT length; its ack comes with the next drain). The word
    /// `cut` rides in the row's first word: the pair's cut, or over more ranks the plan's tile.
    pub(in crate::flash) fn tp_split_post(&self, slot: u32, capacity: usize, rows: usize, cut: usize) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let pp = guard.as_mut().filter(|l| !l.poisoned).ok_or("tensor split link failed earlier; the front is stopping")?;
        let meta = [[slot, capacity as u32, 0, crate::flash_tp_switch::SPLIT_PROMPT + rows as u32]];
        // No TILE or SEQ bit of an earlier pass rides on the request (tp_part sets them per pass).
        (pp.tp_tile, pp.tp_seq) = (false, false);
        let mut row = vec![0.0; W];
        row[0] = cut as f32;
        let posted = pp.drain().and_then(|()| { pp.tp_handoff = true; pp.post(&meta, row, None) });
        posted.map_err(|e| pp.fail(e))
    }
    /// Rank 0, before the transaction of a new chat on `slot`: it owes and keeps nothing (a
    /// previous chat's tile handoff, token-split record or kept rows on the slot go).
    pub(in crate::flash) fn tp_split_start(&self, slot: u32) {
        if let Some(s) = self.tp.borrow_mut().as_mut() {
            s.handoff.retain(|&k| k != slot);
            s.seq.chats.settled(&[slot]);
        }
        if self.pp_mtp() { self.pp_drop_kept(slot); }
    }
    /// Rank 0, a transaction that failed after its request: the ranks' mailbox counts may differ,
    /// so the link stops (as a failed handoff's); the error is returned.
    pub(in crate::flash) fn tp_split_fail(&self, error: String) -> String {
        self.pp.borrow_mut().as_mut().map_or(error.clone(), |pp| pp.fail(error))
    }
    /// Rank 0, after a transaction: the other ranks hold part of its rows (the pair's rank 1 layers
    /// SPLIT..L of rows 0..cut) and rank 0 every layer, as after the token split's import, so the
    /// chat owes them the push.
    pub(in crate::flash) fn tp_split_owed(&self, slot: u32, cut: usize) {
        if let Some(s) = self.tp.borrow_mut().as_mut() {
            s.seq.chats.posted(slot, cut);
            s.seq.chats.imported(slot);
        }
    }
    /// Ranks 1.., KIND_HANDOFF with one SPLIT_PROMPT length: run this rank's side of the
    /// transaction with the word in the request row's first word (the pair's cut, or the sequence
    /// split's tile). False: not a transaction request.
    pub(in crate::flash) fn tp_split_back<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], lens: &[usize], row: &[f32]) -> Result<bool> {
        let mark = crate::flash_tp_switch::SPLIT_PROMPT as usize;
        let [n] = *lens else { return Ok(false) };
        if !(mark..2 * mark).contains(&n) { return Ok(false); }
        let [chat] = sessions else { return Err("prompt transaction with more than one chat".into()) };
        let cut = row.first().copied().ok_or("prompt transaction request without its row")?;
        // Over more ranks the word is the plan's tile (next_xsplit.rs).
        if !self.tp_pair() {
            let rank = self.tp_rank().ok_or("tensor split missing")?;
            return NextSession::xsplit_back(b, chat, n - mark, cut as usize, rank).map(|()| true);
        }
        NextSession::split_worker(b, chat, n - mark, cut as usize).map(|()| true)
    }
}

/// The sequence split over more than two ranks (flash/next_xsplit_plan.rs): its mailbox steps.
impl<'a> NextModel<'a> {
    /// Rank 0: the rows of a whole cold one-chat prompt call that take the split, as tp_split_cut.
    pub(in crate::flash) fn tp_xsplit_cut(&self, chat: &NextSession<'a, '_>, output: bool, rows: usize, chunk: usize) -> Option<usize> {
        let owns = crate::flash_next_rows::check_owners(&chat.gdn_state, &chat.qsa_state, self.state_range()).is_ok();
        if !crate::flash_tp_switch::split_session(output, chat.valid, chat.pos, chat.dump.is_some() || chat.layer_bytes.is_some(), owns)
            || !self.pp_async() || crate::flash_next_rows::prompt_tile() < crate::flash_tp_switch::PIPELINE_TILE || self.tp_pair() { return None; }
        let n = crate::flash_tp_switch::xsplit_rows(0, rows, chunk)?;
        self.tp.borrow().as_ref().filter(|s| s.rank == 0 && s.seq.chats.pending().is_none())?;
        (3..LAYERS).step_by(4).all(|li| (1..n).all(|i| chat.indexer.select(li, i, i + 1) == chat.indexer.select(li, 0, 1))).then_some(n)
    }
    /// The shape every rank builds the plan from, and the ranks.
    pub(in crate::flash) fn tp_xsplit_shape(&self, chat: &mut NextSession<'a, '_>) -> Result<(crate::flash_next_xsplit_plan::Shape, usize)> {
        use crate::flash_next_xsplit_plan::{LayerState, Shape};
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let world = self.tp.borrow().as_ref().map(|s| s.layout.world).ok_or("tensor split missing")?;
        let mut state = Vec::with_capacity(LAYERS - SPLIT);
        for li in SPLIT..LAYERS {
            state.push(if li % 4 == 3 {
                let (_, widths, _) = chat.qsa_state[li].as_mut().ok_or("QSA state missing for the split")?.handoff()?;
                LayerState::Rows(widths)
            } else {
                let (state, history, _) = chat.gdn_state[li].as_mut().ok_or("GDN state missing for the split")?.handoff();
                LayerState::Whole([state.len(), history.len(), 0])
            });
        }
        Ok((Shape { slot: mailbox.slot_bytes() / 4, slots: mailbox.layout().slots, residual: W, split: SPLIT, state }, world))
    }
    /// Every rank: all ranks funded the split? Each puts its answer and takes every other's.
    pub(in crate::flash) fn tp_xsplit_agree(&self, b: &Backend<'a>, funded: bool) -> Result<bool> {
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let (world, origins) = self.tp.borrow().as_ref().map(|s| (s.layout.world, s.origins())).ok_or("tensor split missing")?;
        let mut words: Vec<DeviceVector<'a>> = (0..world).map(|_| b.vector(1)).collect::<Result<_>>()?;
        {
            let mut guard = self.tp.borrow_mut();
            let s = guard.as_mut().ok_or("tensor split missing")?;
            let rank = s.rank;
            b.write_vector(&mut words[rank], &[if funded { 1.0 } else { 0.0 }])?;
            s.sent += 1;
            b.tp_mailbox_put(mailbox, &words[rank], 1, 1, 1, 0, s.sent)?;
            for &o in &origins {
                let (seq, limit) = (s.take(o), s.limit_ns);
                b.tp_mailbox_take_from(mailbox, &mut words[o], 1, 1, 1, 0, mailbox.layout().channel(o, rank), seq, limit)?;
            }
        }
        b.synchronize()?;
        if self.tp_failed() { return Err("tensor split mailbox timed out or its proxy failed before the prompt split".into()); }
        let mut all = funded;
        for &o in &origins { all &= b.read_vector_range(&words[o], 0, 1)?[0] == 1.0; }
        Ok(all)
    }
    /// This rank's message counts when a plan starts: its sends and each origin's takes.
    pub(in crate::flash) fn tp_xsplit_counts(&self) -> Result<(u64, Vec<u64>)> {
        self.tp.borrow().as_ref().map(|s| (s.sent, s.got.clone())).ok_or_else(|| "tensor split missing".into())
    }
    /// A plan ended: count every rank's messages, also the ones this rank did not take.
    pub(in crate::flash) fn tp_xsplit_end(&self, puts: &[u64]) {
        if let Some(s) = self.tp.borrow_mut().as_mut() {
            s.sent += puts[s.rank];
            for (o, n) in puts.iter().enumerate() { if o != s.rank { s.got[o] += n; } }
        }
    }
    pub(in crate::flash) fn tp_xsplit_put(&self, b: &Backend<'a>, v: &DeviceVector<'a>, words: usize, from: usize, seq: u64) -> Result<()> {
        b.tp_mailbox_put(self.tp_box.get().ok_or("tensor split mailbox missing")?, v, 1, v.len(), words, from, seq)
    }
    pub(in crate::flash) fn tp_xsplit_take(&self, b: &Backend<'a>, v: &mut DeviceVector<'a>, words: usize, from: usize, origin: usize, seq: u64) -> Result<()> {
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let (rank, limit) = self.tp.borrow().as_ref().map(|s| (s.rank, s.limit_ns)).ok_or("tensor split missing")?;
        let width = v.len();
        b.tp_mailbox_take_from(mailbox, v, 1, width, words, from, mailbox.layout().channel(origin, rank), seq, limit)
    }
    /// One pad word; the pads outlive the queued operation.
    pub(in crate::flash) fn tp_xsplit_pad(&self, b: &Backend<'a>, seq: u64) -> Result<()> {
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let guard = self.tp.borrow();
        let [pad, _] = guard.as_ref().ok_or("tensor split missing")?.seq.pads.as_ref().ok_or("token split pads missing")?;
        b.tp_mailbox_put(mailbox, pad, 1, 1, 1, 0, seq)
    }
    /// Wait for message `seq` of `origin` and read one word of it.
    pub(in crate::flash) fn tp_xsplit_drop(&self, b: &Backend<'a>, origin: usize, seq: u64) -> Result<()> {
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        let (rank, limit) = (s.rank, s.limit_ns);
        let [_, pad_in] = s.seq.pads.as_mut().ok_or("token split pads missing")?;
        b.tp_mailbox_take_from(mailbox, pad_in, 1, 1, 1, 0, mailbox.layout().channel(origin, rank), seq, limit)
    }
    /// Layer `li`'s position at row `rows` of this rank's part, as a prefix-cache resume sets it.
    pub(in crate::flash) fn tp_xsplit_enter(&self, b: &Backend<'a>, chat: &mut NextSession<'a, '_>, li: usize, rows: usize) -> Result<()> {
        if li % 4 == 3 { chat.fit_kv(b, rows)?; }
        layer_words(chat, li, Some(if li % 4 == 3 { rows } else { 1 })).map(|_| ())
    }
    #[allow(clippy::too_many_arguments)]
    pub(in crate::flash) fn tp_xsplit_state_put(&self, b: &Backend<'a>, chat: &mut NextSession<'a, '_>, li: usize, k: usize, words: usize, from: usize, seq: u64) -> Result<()> {
        self.tp_xsplit_put(b, state_vector(chat, li, k)?, words, from, seq)
    }
    #[allow(clippy::too_many_arguments)]
    pub(in crate::flash) fn tp_xsplit_state_take(&self, b: &Backend<'a>, chat: &mut NextSession<'a, '_>, li: usize, k: usize, words: usize, from: usize, origin: usize, seq: u64) -> Result<()> {
        self.tp_xsplit_take(b, state_vector(chat, li, k)?, words, from, origin, seq)
    }
}
