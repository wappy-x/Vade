//! Token split of a two-rank prompt. A cold one-chat prompt of 2K to 8K rows comes whole
//! (tp_switch::seq_slice), and the engine runs it as two parts back to back (tp_seq_cut). The first
//! part (rows 0..P) runs as pass A: rank 0 runs layers 0..SPLIT
//! and posts the rows as a pipeline tile with the SEQ bit; rank 1 runs layers SPLIT..L and, after
//! each layer, puts that layer's sequence state at row P in the mailbox (attention keys, values
//! and raw index rows 0..P; GDN state and conv history). Its KEEP ack returns the rows for the
//! drafter, as for any pipeline tile. The second part (rows P..N) runs on rank 0 alone over all
//! layers; before each layer from SPLIT it takes rank 1's state of that layer, the words a
//! prefix-cache resume at P installs, so rows P..N keep world one's arithmetic, and a prompt tail
//! after it runs on rank 0 alone too. The chat's first other pass first sends rank 1 the rest of
//! the state (layers SPLIT..L whole, one way) and moves rank 1 to rank 0's position; decode then
//! runs on the split. If another pass comes before the second part (a failed second part), rank 0
//! first takes and drops the rest of the export, so rank 1 finishes pass A and the link goes on;
//! the chat's later passes then fail (tp_switch::SeqChats).
use super::*;
use crate::flash_tp_switch::{seq_push_tail, SeqChats, SeqMode, SeqPlan, SEQ_LIVE};
use spark_gpu_probe::driver::TP_MAILBOX_SLOTS;
#[path = "next_tp_split.rs"]
mod split;

/// KIND_HANDOFF length field of SEQ_PUSH or more: ranks 1.. receive layers SPLIT..L, as
/// tp_seq_handoffs sends them, and advance by the field's tail rows (tp_switch::seq_push_tail).
/// Tile handoffs send 1 and next_tp_heads' requests 2 and 3.
pub(super) const SEQ_PUSH: u32 = 1 << 24;
/// Mailbox slot size on two ranks: one attention layer's state at 4096 rows (keys, values, index)
/// crosses in about three messages, and the four slots buffer about two layers. With the
/// split's 1.31 MB slots an attention layer took 15 two-way rendezvous, and the ranks waited for
/// each other about 0.4 s per cold 8K prompt (own-prefill-split r1, PROFILE ps-prof3-X).
pub(super) fn slot_bytes(base: usize) -> usize { base.max(10 << 20) }
/// Mailbox slot size over more ranks: one attention vector of a sequence split's part crosses in
/// one message, so the messages of three layers fit the eight slots (next_xsplit_plan.rs).
pub(super) fn chain_slot_bytes(base: usize) -> usize { base.max(32 << 20) }
/// Rank 1 puts a layer's state LAG layers after it ran (the credits let it run ahead, up to the
/// mailbox's slots; rank 0 runs its rows of a layer only after rank 1 ran it).
const LAG: usize = 1;

/// This rank's token-split chats (rank 0), pass mode, credit pads and message count.
pub(super) struct Seq<'a> {
    pub(super) chats: SeqChats,
    /// This rank's pad word and the peer's (the credits of the other direction).
    pads: Option<[DeviceVector<'a>; 2]>,
    /// State messages of the current pass so far.
    msgs: usize,
    /// Rank 0: the first part this pass posts (slot, rows, its export's messages).
    posting: Option<(u32, usize, usize)>,
    /// Rank 0: messages of the pending export not yet taken.
    owed: usize,
}
impl Default for Seq<'_> {
    fn default() -> Self { Self { chats: SeqChats::default(), pads: None, msgs: 0, posting: None, owed: 0 } }
}

impl<'a> NextModel<'a> {
    /// Rank 1: the pass is a SEQ tile (rank 0's request bit).
    pub(super) fn tp_seq_mail(&self, seq: bool) {
        if let Some(s) = self.tp.borrow_mut().as_mut() { (s.seq.chats.mode, s.seq.msgs) = (if seq { SeqMode::Export } else { SeqMode::Off }, 0); }
    }
    pub(super) fn tp_seq_mode(&self) -> SeqMode {
        self.tp.borrow().as_ref().map_or(SeqMode::Off, |s| s.seq.chats.mode)
    }
    /// Rank 0, before a pass: Some(first part) for a token-split prompt pass (the first part, or
    /// the second part or a tail here); None for any other pass, after it took the rest of an
    /// export this pass does not import and sent owed chats' state to the other ranks.
    pub(super) fn tp_seq_plan<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], spans: &[std::ops::Range<usize>],
        verify: bool, output: bool) -> Result<Option<bool>> {
        // Over more ranks the only chats here are a sequence split's: Tile needs a pair's tp_seq_cut.
        let chats: Vec<_> = sessions.iter().zip(spans).map(|(c, span)| (c.slot, c.pos, span.len(), !c.logits_ready)).collect();
        let plan = {
            let mut guard = self.tp.borrow_mut();
            let s = guard.as_mut().ok_or("tensor split missing")?;
            s.seq.posting = None;
            s.seq.chats.plan(&chats, verify, output)?
        };
        match plan {
            // The first part's tile goes out asynchronously (pp_post), as rank 1 waits for rank 0's
            // import before it acks; without room for that the prompt runs as usual.
            SeqPlan::Tile if self.pp_async() && self.pp_keep_room(spans[0].len()) => {
                let slot = self.tp_box.get().ok_or("tensor split mailbox missing")?.slot_bytes() / 4;
                let messages = export_messages(sessions[0], spans[0].len(), slot)?;
                if let Some(s) = self.tp.borrow_mut().as_mut() { s.seq.posting = Some((sessions[0].slot, spans[0].len(), messages)); }
                Ok(Some(true))
            }
            SeqPlan::Tile => Ok(None),
            SeqPlan::Here => Ok(Some(false)),
            SeqPlan::Normal { discard, push } => {
                if let Some(slot) = discard { self.tp_seq_discard(b, slot)?; }
                self.tp_seq_handoffs(b, sessions, &push)?;
                Ok(None)
            }
        }
    }
    /// Rank 0, after the first part's tile is posted: rank 1 now owes its export.
    pub(in crate::flash) fn tp_seq_posted(&self) -> Result<()> {
        if let Some(s) = self.tp.borrow_mut().as_mut() && let Some((slot, rows, messages)) = s.seq.posting.take() {
            s.seq.chats.posted(slot, rows);
            s.seq.owed = messages;
        }
        Ok(())
    }
    /// Rank 0: this chat owes a sequence split's push, so a tail runs alone before it.
    pub(in crate::flash) fn tp_seq_owes(&self, slot: u32) -> bool {
        !self.tp_pair() && self.tp.borrow().as_ref().is_some_and(|s| s.rank == 0 && s.seq.chats.owes(slot))
    }
    /// Rank 0, after a failed pass: take and drop the rest of a pending export now, as rank 1
    /// waits for its credits only up to the link's timeout and the front may stay idle; that
    /// chat's later passes fail. The pass's own error is the one returned: a failed discard
    /// fails the link (pp.fail) or leaves the mailbox's error word set for the next take.
    pub(in crate::flash) fn tp_seq_abandon(&self, b: &Backend<'a>) {
        let slot = self.tp.borrow_mut().as_mut().and_then(|s| s.seq.chats.lose());
        if let Some(slot) = slot { let _ = self.tp_seq_discard(b, slot); }
    }
    /// Rank 0: take the rest of a pending export (one word of each message) and return each
    /// message's credit, so rank 1 ends pass A; then read its ack and drop the tile's kept rows.
    fn tp_seq_discard(&self, b: &Backend<'a>, slot: u32) -> Result<()> {
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        {
            let mut guard = self.tp.borrow_mut();
            let s = guard.as_mut().ok_or("tensor split missing")?;
            if s.seq.pads.is_none() { s.seq.pads = Some([b.vector(1)?, b.vector(1)?]); }
            let (mut sent, mut taken, limit) = (s.sent, s.got[s.peer], s.limit_ns);
            let [pad, scratch] = s.seq.pads.as_mut().ok_or("token split pads missing")?;
            for _ in 0..s.seq.owed {
                taken += 1;
                b.tp_mailbox_take(mailbox, scratch, 1, 1, 1, 0, taken, limit)?;
                sent += 1;
                b.tp_mailbox_put(mailbox, pad, 1, 1, 1, 0, sent)?;
            }
            (s.sent, s.got[s.peer], s.seq.owed) = (sent, taken, 0);
        }
        {
            let mut guard = self.pp.borrow_mut();
            let pp = guard.as_mut().filter(|l| !l.poisoned).ok_or("tensor split link failed earlier; the front is stopping")?;
            let drained = pp.drain();
            drained.map_err(|e| pp.fail(e))?;
        }
        self.pp_drop_kept(slot);
        Ok(())
    }
    /// Both ranks, at the top of layer `li` (and with li = the pass's end after the last layer):
    /// rank 1 in pass A puts layer li - 1's state; rank 0 in pass B takes layer li's.
    pub(in crate::flash) fn tp_seq_layer<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], li: usize, count: usize) -> Result<()> {
        // Rank 1 puts layer li - LAG at the top of layer li, and every layer still owed after the
        // last one (li = the pass's end); rank 0 takes layer li at its top.
        let (layers, rows, put) = match self.tp_seq_mode() {
            SeqMode::Off | SeqMode::Alone => return Ok(()),
            SeqMode::Export if li == LAYERS => ((LAYERS - LAG).max(SPLIT)..LAYERS, sessions[0].pos + count, true),
            SeqMode::Export if li >= SPLIT + LAG => (li - LAG..li - LAG + 1, sessions[0].pos + count, true),
            SeqMode::Import(rows) if (SPLIT..LAYERS).contains(&li) => (li..li + 1, rows, false),
            _ => return Ok(()),
        };
        for layer in layers { self.tp_seq_state(b, sessions, layer, rows, put)?; }
        if put && li == LAYERS { self.tp_seq_credits(b)?; }
        // Rank 0 queued the takes of the whole export: the chat now owes rank 1 the push.
        if !put && li + 1 == LAYERS && let Some(s) = self.tp.borrow_mut().as_mut() { s.seq.chats.imported(sessions[0].slot); }
        Ok(())
    }
    /// Rank 1, after its last layer (and rank 0 after a transaction's seeds, next_split.rs):
    /// take the credits of the messages still unconfirmed.
    pub(in crate::flash) fn tp_seq_credits(&self, b: &Backend<'a>) -> Result<()> {
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        let [_, pad_in] = s.seq.pads.as_mut().ok_or("token split pads missing")?;
        for _ in 0..s.seq.msgs.min(TP_MAILBOX_SLOTS) {
            s.got[s.peer] += 1;
            b.tp_mailbox_take(mailbox, pad_in, 1, 1, 1, 0, s.got[s.peer], s.limit_ns)?;
        }
        s.seq.msgs = 0;
        Ok(())
    }
    /// One layer's state at row `rows`: rank 1 puts it (`put`), rank 0 takes it.
    pub(in crate::flash) fn tp_seq_state<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], layer: usize, rows: usize, put: bool) -> Result<()> {
        let [chat] = sessions else { return Err("token split pass with more than one chat".into()) };
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let slot = mailbox.slot_bytes() / 4;
        if layer % 4 == 3 { chat.fit_kv(b, rows)?; }
        // Rank 0 sets the attention rows and the GDN cached flag as rank 1 holds them after the layer.
        let (_, words) = layer_words(chat, layer, Some(if layer % 4 == 3 { rows } else { 1 }))?;
        let mut parts = Vec::new();
        for (k, n) in words.into_iter().enumerate() { pieces(&mut parts, (0, layer, k), n, slot); }
        // The mailbox has no flow control, so the receiver returns a credit (one pad word) for each
        // message it took, and the sender takes credit m - SLOTS before it puts message m: message
        // m's slot is free, and up to SLOTS messages travel while the receiver catches up.
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        if s.seq.pads.is_none() { s.seq.pads = Some([b.vector(1)?, b.vector(1)?]); }
        let (mut sent, mut taken, mut msgs, limit) = (s.sent, s.got[s.peer], s.seq.msgs, s.limit_ns);
        let [pad, pad_in] = s.seq.pads.as_mut().ok_or("token split pads missing")?;
        for &Piece { at: (_, li, k), from, words } in &parts {
            let v = state_vector(chat, li, k)?;
            if put {
                if msgs >= TP_MAILBOX_SLOTS {
                    taken += 1;
                    b.tp_mailbox_take(mailbox, pad_in, 1, 1, 1, 0, taken, limit)?;
                }
                sent += 1;
                b.tp_mailbox_put(mailbox, v, 1, v.len(), words, from, sent)?;
            } else {
                taken += 1;
                let width = v.len();
                b.tp_mailbox_take(mailbox, v, 1, width, words, from, taken, limit)?;
                sent += 1;
                b.tp_mailbox_put(mailbox, pad, 1, 1, 1, 0, sent)?;
            }
            msgs += 1;
        }
        if !put { s.seq.owed = s.seq.owed.saturating_sub(parts.len()); }
        (s.sent, s.got[s.peer], s.seq.msgs) = (sent, taken, msgs);
        Ok(())
    }
    /// Rank 0, pass B, before the drafter observes: read pass A's ack, which carries its rows.
    pub(in crate::flash) fn tp_seq_drain(&self) -> Result<()> {
        if !matches!(self.tp_seq_mode(), SeqMode::Import(_)) { return Ok(()); }
        let mut guard = self.pp.borrow_mut();
        let pp = guard.as_mut().filter(|l| !l.poisoned).ok_or("tensor split link failed earlier; the front is stopping")?;
        let drained = pp.drain();
        drained.map_err(|e| pp.fail(e))
    }
    /// Rank 0, before a pass: push the rest of the state to the other ranks for the owed chats at
    /// `push` (index, their position): layers SPLIT..L one way, whole on the pair; over more ranks
    /// without the attention rows every rank already holds. Afterwards every rank holds the
    /// chats' whole state. A checkpoint that only rank 0 holds crosses first (tp_seq_checkpoint).
    fn tp_seq_handoffs<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], push: &[(usize, usize)]) -> Result<()> {
        if push.is_empty() { return Ok(()); }
        // The row of a checkpoint only rank 0 holds, at or after the other ranks' position.
        let saved = match push {
            [(i, rows)] => sessions[*i].checkpoint_unsent().filter(|at| (*rows..=sessions[*i].pos).contains(at)),
            _ => None,
        };
        // A stale record (position behind its first part) is a request error, not an overflow.
        let meta: Vec<[u32; 4]> = push.iter().map(|&(i, rows)| {
            let s = &sessions[i];
            let tail = s.pos.checked_sub(rows).ok_or_else(|| format!("token split: slot {} at {} is behind its first part ({rows} rows)", s.slot, s.pos))?;
            Ok([s.slot, s.capacity as u32, rows as u32, SEQ_PUSH + saved.map_or(tail, |at| at - rows) as u32])
        }).collect::<Result<_>>()?;
        let at: Vec<usize> = push.iter().map(|&(i, _)| i).collect();
        let link = self.tp.borrow().as_ref().map(|s| s.link.try_clone()).ok_or("tensor split missing")?.map_err(err)?;
        self.tp_seq_post(&meta)?;
        let mut chats: Vec<&mut NextSession<'a, 'w>> = Vec::new();
        for (i, chat) in sessions.iter_mut().enumerate() { if at.contains(&i) { chats.push(chat); } }
        if saved.is_some() { chats[0].swap_checkpoint_gdn(); } else { chats.iter_mut().for_each(|chat| chat.drop_unsent_checkpoint()); }
        // As tp_part's handoff: a failure leaves the chats' state partly installed.
        let mut pushed = if self.tp_pair() { self.tp_swap_state(b, &mut chats, SPLIT..LAYERS, SPLIT..SPLIT, link) } else {
            let shared: Vec<usize> = push.iter().map(|&(_, rows)| self.tp_xsplit_shared(rows)).collect();
            self.tp_gather_state(b, &mut chats, Some(&shared))
        };
        if let Some(at) = saved {
            chats[0].swap_checkpoint_gdn();
            pushed = pushed.and_then(|()| self.tp_seq_checkpoint(b, chats[0], meta[0], at));
        }
        if let Err(error) = pushed {
            chats.iter_mut().for_each(|chat| chat.valid = false);
            return Err(self.pp.borrow_mut().as_mut().map_or(error.clone(), |pp| pp.fail(error)));
        }
        let settled: Vec<u32> = chats.iter().map(|chat| chat.slot).collect();
        if let Some(s) = self.tp.borrow_mut().as_mut() { s.seq.chats.settled(&settled); }
        Ok(())
    }
    /// Rank 0: post a KIND_HANDOFF request with these chats' `meta`.
    fn tp_seq_post(&self, meta: &[[u32; 4]]) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let pp = guard.as_mut().filter(|l| !l.poisoned).ok_or("tensor split link failed earlier; the front is stopping")?;
        let posted = pp.drain().and_then(|()| { pp.tp_handoff = true; pp.post(meta, vec![0.0; meta.len() * W], None) });
        posted.map_err(|e| pp.fail(e))
    }
    /// Rank 0, after the push of a chat's checkpoint: the push held the recurrent state at row
    /// `at`, and the other ranks stand at that row, so they capture as at any checkpoint. A tail
    /// rank 0 ran alone after that row then crosses: the live recurrent state in a second push,
    /// as the first push held every attention row. So the first token of a sequence split comes
    /// before the push, on every prompt (next_prefix.rs, capture_checkpoint).
    fn tp_seq_checkpoint<'w>(&self, b: &Backend<'a>, chat: &mut NextSession<'a, 'w>, [slot, capacity, ..]: [u32; 4], at: usize) -> Result<()> {
        let remote = self.pp_cache_capture(slot, capacity as usize, at, 0)?;
        chat.checkpoint_sent(remote);
        if chat.pos == at { return Ok(()); }
        self.tp_seq_post(&[[slot, capacity, at as u32, SEQ_PUSH + SEQ_LIVE + (chat.pos - at) as u32]])?;
        let shared = [chat.pos];
        self.tp_gather_state(b, &mut [chat], Some(&shared))
    }
    /// Ranks 1.., KIND_HANDOFF with every length at least SEQ_PUSH: receive layers SPLIT..L, as
    /// tp_seq_handoffs sends them, and advance by the field's tail rows. False: not a token-split
    /// handoff.
    pub(super) fn tp_seq_handoff_back<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], lens: &[usize]) -> Result<bool> {
        if lens.is_empty() || lens.iter().any(|&n| n < SEQ_PUSH as usize) { return Ok(false); }
        let tail = |n: usize| seq_push_tail(n - SEQ_PUSH as usize);
        let swapped = if self.tp_pair() {
            let link = self.tp.borrow().as_ref().map(|s| s.link.try_clone()).ok_or("tensor split missing")?.map_err(err)?;
            self.tp_swap_state(b, sessions, SPLIT..SPLIT, SPLIT..LAYERS, link)
        } else {
            let live = |n: usize| n & SEQ_LIVE as usize != 0;
            let shared: Vec<usize> = sessions.iter().zip(lens).map(|(chat, &n)| if live(n) { chat.pos + tail(n) } else { self.tp_xsplit_shared(chat.pos) }).collect();
            self.tp_gather_state(b, sessions, Some(&shared))
        };
        if swapped.is_err() { sessions.iter_mut().for_each(|chat| chat.valid = false); }
        swapped?;
        for (chat, &n) in sessions.iter_mut().zip(lens) { chat.pos += tail(n); }
        Ok(true)
    }
}

impl<'a> NextModel<'a> {
    /// Rows of a chat's next prompt slice on a two-rank split front (tp_switch::seq_slice), for a
    /// live chat or a cached one (None: pipeline tiles); None off the split front. Over more ranks
    /// a prompt the sequence split takes comes whole; every other slice is a chain tile of
    /// prompt_options::chain_tile rows.
    pub fn prompt_slice(&self, chat: Option<&NextSession<'a, '_>>, end: usize, chunk: usize) -> Option<usize> {
        self.tp.borrow().as_ref().filter(|s| s.rank == 0)?;
        if !self.tp_pair() {
            // A prompt the sequence split takes comes whole.
            if let Some(c) = chat && crate::flash_tp_switch::xsplit_rows(c.pos, end, chunk).is_some() { return Some(end - c.pos); }
            return Some(crate::flash_prompt_options::chain_tile(end).min(chunk));
        }
        let tile = crate::flash_tp_switch::PIPELINE_TILE.min(chunk);
        Some(chat.map_or(tile, |c| crate::flash_tp_switch::seq_slice(c.pos, end, chunk)))
    }
    /// Rank 0, a prompt call's next rows: the first part when the call holds a whole cold prompt
    /// the token split takes (its final call from position 0); None: the usual rows. Only a first
    /// part cut here splits, so the second part follows it in the same call.
    pub(in crate::flash) fn tp_seq_cut(&self, chat: &NextSession<'a, '_>, output: bool, rows: usize, chunk: usize) -> Option<usize> {
        if !output || chat.pos != 0 { return None; }
        let mut guard = self.tp.borrow_mut();
        // The token split runs on the pair only (one Spark has no split and stops here).
        let s = guard.as_mut().filter(|s| s.rank == 0 && s.layout.world == crate::flash_tp_switch::PAIR)?;
        let p = crate::flash_tp_switch::seq_first(0, rows, chunk)?;
        s.seq.chats.expect(chat.slot, p);
        Some(p)
    }
}

/// Rank 0, at the first part: the messages of rank 1's export of `rows` rows (tp_seq_state's
/// pieces of layers SPLIT..L), read without changing the chat's state.
fn export_messages(chat: &mut NextSession<'_, '_>, rows: usize, slot: usize) -> Result<usize> {
    let mut parts = Vec::new();
    for li in SPLIT..LAYERS {
        let words = if li % 4 == 3 {
            let (vectors, widths, _) = chat.qsa_state[li].as_mut().ok_or("QSA state missing for the handoff")?.handoff()?;
            let mut words = [0; 3];
            for (k, (v, w)) in vectors.into_iter().zip(widths).enumerate() {
                if w != 0 { words[k] = handoff_words(rows, w, v.len())?; }
            }
            words
        } else {
            let (state, history, _) = chat.gdn_state[li].as_mut().ok_or("GDN state missing for the handoff")?.handoff();
            [state.len(), history.len(), 0]
        };
        for (k, n) in words.into_iter().enumerate() { pieces(&mut parts, (0, li, k), n, slot); }
    }
    Ok(parts.len())
}
