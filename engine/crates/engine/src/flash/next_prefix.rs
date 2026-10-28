//! Exact prompt-prefix checkpoints for the serving cache (VADE_FLASHNEXT_PREFIX_CACHE).
//! Multi-token prompt tiles start only on absolute 64-token boundaries (prompt_rows), so
//! the recurrent state after a prompt's last whole 64-row tile is the state a cold prefill
//! of any prompt with that prefix holds at the same position. A checkpoint copies GDN, PLE
//! history, the PLE hash and the draft's last hidden row there; QSA and draft KV rows below
//! it are never rewritten by the request's decode, so `save_prefix` copies them from the
//! live session later and moves the checkpoint in. `session_from_prefix` installs the
//! copies in a fresh session and prefill resumes at that 64-aligned position (an unaligned
//! resume would run single rows up to the next boundary). No kernel, tile or order change.
//! Adoption (VADE_FLASHNEXT_ADOPT, serve): `save_prefix_in_place` publishes the checkpoint
//! alone; the finished session's QSA and draft arenas move out with `into_storage` (no pool
//! slot), and `session_from_storage` builds the next turn's session on them, rewound to the
//! checkpoint: the rows below it are the ones `save_prefix` would have copied.
//! Funding: a checkpoint lives from the final prompt tile until it is published or the
//! first MTP verify, which releases it before allocating the rollback storage the session
//! reservation and opt-in retention extra fund; the 16 MiB session slack covers PLE
//! history and the draft row. Models without MTP fund `checkpoint_bytes`.
use super::*;

pub(super) struct Checkpoint<'a> {
    position: usize,
    gdn: Vec<Option<GdnState<'a>>>,
    ple: DeviceVector<'a>,
    hash: PleHash,
    last: Option<DeviceVector<'a>>,
    remote: Option<pp::cache::Lease>,
}

/// Immutable prompt prefix at a 64-aligned position, independent of any live session.
pub struct NextPrefix<'a> {
    position: usize,
    qsa: Vec<Option<QsaState<'a>>>,
    gdn: Vec<Option<GdnState<'a>>>,
    ple: DeviceVector<'a>,
    hash: PleHash,
    draft: Option<(QsaState<'a>, DeviceVector<'a>)>,
    /// In place: no QSA/draft row copies; `last` is the draft's saved row.
    in_place: bool,
    last: Option<DeviceVector<'a>>,
    bytes: usize,
    remote: Option<pp::cache::Lease>,
}
impl NextPrefix<'_> {
    pub fn position(&self) -> usize {
        self.position
    }
    /// Rows stay in the publishing session's arenas (restore with `session_from_storage`).
    pub fn in_place(&self) -> bool {
        self.in_place
    }
    /// Upper bound of this prefix's device bytes on either rank.
    pub fn device_bytes(&self) -> usize {
        self.bytes
    }
}
/// Rounding allowance per retained device vector.
const VECTOR_SLACK: usize = 256;

/// A finished session's KV arenas (QSA layers and draft attention): rows below its
/// published checkpoint are intact. Holds no state-pool slot.
pub struct NextStorage<'a> {
    capacity: usize,
    position: usize,
    qsa: Vec<Option<QsaState<'a>>>,
    draft: Option<QsaState<'a>>,
}

impl<'a, 'w> NextSession<'a, 'w> {
    /// Capture a checkpoint at the last 64-aligned row of each output prefill (per request).
    pub fn set_prefix_checkpoint(&mut self, on: bool) {
        self.checkpointing = on;
        if !on {
            *self.checkpoint.get_mut() = None;
        }
    }
    /// Device bytes one checkpoint holds (GDN, PLE history, draft row); fund it per session.
    pub fn checkpoint_bytes() -> usize {
        // Ple::reset: the convolution history holds 9 rows of W.
        crate::flash_next_rows::recurrent_bytes() + (2 * LAYERS + 2) * VECTOR_SLACK + (9 * W + W) * 4
    }
    /// Current GDN storage can include prompt-chunk history beyond the decode state.
    pub(super) fn checkpoint_allocation_bytes(&self) -> usize {
        self.gdn_state.iter().flatten().flat_map(|g| g.storage_lengths()).map(|n| n * 4 + VECTOR_SLACK).sum::<usize>()
            + self.ple.history_len() * 4 + VECTOR_SLACK
            + if self.draft.is_some() && self.model.prompt_frontend { W * 4 + VECTOR_SLACK } else { 0 }
    }
    /// Called before a final partial prompt tile and after an output prefill; acts at
    /// 64-aligned positions only. Preparation drains live draft history: failure invalidates
    /// the session and reaches the caller. Only local snapshot copies are best effort.
    pub(super) fn capture_checkpoint(&mut self, b: &Backend<'a>) -> Result<()> {
        if !self.checkpointing || self.pos == 0 || !self.pos.is_multiple_of(64) { return Ok(()); }
        let remote = match self.prepare_checkpoint(b) {
            Ok(remote) => remote,
            Err(error) => {
                self.valid = false;
                self.logits_ready = false;
                *self.checkpoint.get_mut() = None;
                return Err(error);
            }
        };
        if self.model.pp_front() && remote.is_none() || self.try_capture(b, remote).is_err() {
            *self.checkpoint.get_mut() = None;
        }
        Ok(())
    }
    /// One Spark: nothing to prepare. Split front: settle the state the ranks owe each other,
    /// then the worker captures its own rows at the same position (pp_cache_capture).
    fn prepare_checkpoint(&mut self, b: &Backend<'a>) -> Result<Option<pp::cache::Lease>> {
        if !self.model.pp_front() { return Ok(None); }
        // Prompt tiles can leave TP state split by layers, and short passes by GDN heads.
        // Both settle by copies only (no sum), before the same whole-state checkpoint on each rank.
        self.model.tp_part(b, &mut [self], &[0..0], true, false)?;
        self.model.tp_gdn_frame(b, &mut [self], usize::MAX)?;
        // The ordered capture drains the link, so the kept rows of queued tiles are back.
        let remote = self.model.pp_cache_capture(self.slot, self.capacity, self.pos)?;
        self.ingest_kept(b)?;
        self.flush_draft_feed(b)?;
        Ok(remote)
    }
    fn try_capture(&mut self, b: &Backend<'a>, remote: Option<pp::cache::Lease>) -> Result<()> {
        // A tensor split head half is no prefix of another chat's whole state.
        if self.gdn_state.iter().flatten().any(|g| g.is_half()) { return Err("GDN state holds one split rank's heads".into()); }
        let mut c = match self.checkpoint.get_mut().take() {
            Some(c) => c,
            None => {
                // One arena (one allocation, one free) holds the whole checkpoint: a cold
                // request must not pay a driver allocation per layer.
                let mut lengths: Vec<usize> = self.gdn_state.iter().flatten().flat_map(|g| g.storage_lengths()).collect();
                lengths.push(self.ple.history_len());
                if self.draft.is_some() && self.model.prompt_frontend { lengths.push(W); }
                let mut v = b.vector_arena(&lengths)?.into_iter();
                let gdn = self.gdn_state.iter()
                    .map(|g| g.as_ref().map(|g| g.with_storage(v.next().unwrap(), v.next().unwrap())))
                    .collect();
                Checkpoint { position: 0, gdn, ple: v.next().unwrap(), hash: self.hash.clone(), last: v.next(), remote: None }
            }
        };
        for (live, saved) in self.gdn_state.iter().zip(c.gdn.iter_mut()) {
            if let (Some(live), Some(saved)) = (live, saved) {
                live.copy_into(b, saved)?;
            }
        }
        self.ple.copy_history(b, &mut c.ple)?;
        c.hash = self.hash.clone();
        c.last = match &self.draft {
            Some(d) if self.model.prompt_frontend => {
                let mut last = match c.last.take() {
                    Some(v) => v,
                    None => b.vector(W)?,
                };
                b.copy_range(&d.last, &mut last, 0, 0, W)?;
                Some(last)
            }
            _ => None,
        };
        c.position = self.pos;
        c.remote = remote;
        *self.checkpoint.get_mut() = Some(c);
        Ok(())
    }
    /// The restorable prefix length, when a checkpoint with its draft history exists.
    pub fn checkpoint_position(&self) -> Option<usize> {
        self.checkpoint
            .borrow()
            .as_ref()
            .filter(|c| self.valid && (c.last.is_some() || self.model.mtp_depth == 0 || !self.model.prompt_frontend))
            .map(|c| c.position)
    }
    /// Device bytes `save_prefix` allocates for the current checkpoint.
    pub fn prefix_bytes(&self) -> usize {
        let held = self.checkpoint.borrow();
        let Some(c) = held.as_ref() else { return 0 };
        let rows = c.position;
        let qsa: usize = self.qsa_state.iter().flatten().map(|q| q.rows_bytes(rows)).sum();
        let draft = match (&self.draft, &c.last) {
            (Some(d), Some(_)) => d.history_bytes(rows - 1) + W * 4 + VECTOR_SLACK,
            _ => 0,
        };
        let local = qsa + draft + crate::flash_next_rows::recurrent_bytes() + 2 * LAYERS * VECTOR_SLACK
            + c.ple.len() * 4 + VECTOR_SLACK;
        // A tensor split charges the larger of the two ranks' copies.
        local.max(c.remote.as_ref().map_or(0, |r| r.bytes))
    }
    /// Move the checkpoint out with copies of the QSA and draft rows below it (immutable).
    pub fn save_prefix(&self, b: &Backend<'a>) -> Result<NextPrefix<'a>> {
        let rows = self.checkpoint_position().ok_or("Flash-Next session has no prompt checkpoint")?;
        let bytes = self.prefix_bytes();
        let draft = self.draft.as_ref().filter(|_| self.checkpoint.borrow().as_ref().is_some_and(|c| c.last.is_some()));
        // One arena for every layer's rows and the draft history (one allocation, one free).
        let mut lengths: Vec<usize> = self.qsa_state.iter().flatten().flat_map(|q| q.rows_lengths(rows)).collect();
        if let Some(d) = draft { lengths.extend(d.history_lengths(rows - 1)); }
        let mut v = b.vector_arena(&lengths)?.into_iter();
        let mut next3 = || [v.next().unwrap(), v.next().unwrap(), v.next().unwrap()];
        let qsa = self
            .qsa_state
            .iter()
            .map(|q| q.as_ref().map(|q| q.copy_rows_into(b, rows, next3())).transpose())
            .collect::<Result<Vec<_>>>()?;
        let draft = draft.map(|d| d.copy_history_into(b, rows - 1, next3())).transpose()?;
        let c = self.checkpoint.borrow_mut().take().ok_or("missing prompt checkpoint")?;
        if let Some(remote) = &c.remote { self.model.pp_cache_publish(remote)?; }
        Ok(NextPrefix {
            position: rows,
            qsa,
            gdn: c.gdn,
            ple: c.ple,
            hash: c.hash,
            draft: draft.zip(c.last),
            in_place: false,
            last: None,
            bytes,
            remote: c.remote,
        })
    }
    /// Device bytes `save_prefix_in_place` keeps (the checkpoint alone).
    pub fn checkpoint_prefix_bytes(&self) -> usize {
        let held = self.checkpoint.borrow();
        let Some(c) = held.as_ref() else { return 0 };
        let local = crate::flash_next_rows::recurrent_bytes() + 2 * LAYERS * VECTOR_SLACK + c.ple.len() * 4 + VECTOR_SLACK
            + c.last.as_ref().map_or(0, |_| W * 4 + VECTOR_SLACK);
        local.max(c.remote.as_ref().map_or(0, |r| r.bytes))
    }
    /// Move the checkpoint out without copying rows: they stay in this session's arenas
    /// until `into_storage` hands them to the next turn.
    pub fn save_prefix_in_place(&self) -> Result<NextPrefix<'a>> {
        let rows = self.checkpoint_position().ok_or("Flash-Next session has no prompt checkpoint")?;
        let bytes = self.checkpoint_prefix_bytes();
        let c = self.checkpoint.borrow_mut().take().ok_or("missing prompt checkpoint")?;
        if let Some(remote) = &c.remote { self.model.pp_cache_publish(remote)?; }
        Ok(NextPrefix { position: rows, qsa: Vec::new(), gdn: c.gdn, ple: c.ple, hash: c.hash, draft: None,
            in_place: true, last: c.last, bytes, remote: c.remote })
    }
    /// The KV arenas of a finished, valid session with no attention in flight.
    pub fn into_storage(mut self) -> Option<NextStorage<'a>> {
        if !self.valid || self.qsa_state.iter().flatten().any(|q| !q.idle()) { return None; }
        let draft = self.draft.take().map(|d| d.into_attention());
        if draft.as_ref().is_some_and(|q| !q.idle()) { return None; }
        Some(NextStorage { capacity: self.capacity, position: self.pos, qsa: std::mem::take(&mut self.qsa_state), draft })
    }
}
impl<'a> NextModel<'a> {
    /// A fresh session holding `prefix`; prefill continues at its 64-aligned position.
    pub fn session_from_prefix<'w>(
        &'w self,
        b: &Backend<'a>,
        capacity: usize,
        chunk: usize,
        prefix: &NextPrefix<'a>,
    ) -> Result<NextSession<'a, 'w>> {
        if prefix.in_place || prefix.position == 0 || !prefix.position.is_multiple_of(64) || prefix.position >= capacity {
            return Err("invalid Flash-Next prefix position".into());
        }
        let mut s = self.session_with_chunk(b, capacity, chunk)?;
        s.valid = false;
        for (live, saved) in s.qsa_state.iter_mut().zip(&prefix.qsa) {
            match (live, saved) {
                (Some(live), Some(saved)) => live.restore_rows(b, saved)?,
                (None, None) => {}
                _ => return Err("Flash-Next prefix layer layout differs".into()),
            }
        }
        for (live, saved) in s.gdn_state.iter_mut().zip(&prefix.gdn) {
            match (live, saved) {
                (Some(live), Some(saved)) => saved.copy_into(b, live)?,
                (None, None) => {}
                _ => return Err("Flash-Next prefix layer layout differs".into()),
            }
        }
        s.ple.restore_history(b, &prefix.ple)?;
        s.hash = prefix.hash.clone();
        match (&mut s.draft, &prefix.draft) {
            (Some(d), Some((rows, last))) => d.restore_history(b, rows, last)?,
            // Workers own verification state, but no frontend draft history.
            (Some(_), None) if self.prompt_frontend => s.disable_mtp(),
            _ => {}
        }
        if let Some(remote) = &prefix.remote { self.pp_cache_restore(remote, s.slot, capacity, prefix.position)?; }
        s.pos = prefix.position;
        s.sync_pool_state(b)?;
        s.logits_ready = false;
        if let Some(d) = s.draft.as_ref() { d.bound_at.set(None); }
        s.valid = true;
        Ok(s)
    }
    /// The next turn's session on a finished session's arenas, rewound to `prefix` (published
    /// in place by that session): GDN, PLE and the draft row come from the checkpoint as in
    /// `session_from_prefix`; QSA and draft rows below it are already in place.
    pub fn session_from_storage<'w>(
        &'w self,
        b: &Backend<'a>,
        chunk: usize,
        prefix: &NextPrefix<'a>,
        storage: NextStorage<'a>,
    ) -> Result<NextSession<'a, 'w>> {
        let NextStorage { capacity, position, qsa, draft } = storage;
        if !prefix.in_place || prefix.position == 0 || !prefix.position.is_multiple_of(64)
            || prefix.position >= capacity || position < prefix.position {
            return Err("invalid Flash-Next adopted prefix".into());
        }
        crate::flash_next_rows::reservation_rows(capacity, chunk)?;
        self.pp_cache_releases()?;
        let had_draft = draft.is_some();
        let mut s = NextSession::new_from(b, self, capacity, Some(qsa), draft)?;
        s.chunk_limit = chunk;
        s.chunk = chunk;
        s.valid = false;
        for q in s.qsa_state.iter_mut().flatten() {
            q.rewind(prefix.position);
        }
        for (live, saved) in s.gdn_state.iter_mut().zip(&prefix.gdn) {
            match (live, saved) {
                (Some(live), Some(saved)) => saved.copy_into(b, live)?,
                (None, None) => {}
                _ => return Err("Flash-Next prefix layer layout differs".into()),
            }
        }
        s.ple.restore_history(b, &prefix.ple)?;
        s.hash = prefix.hash.clone();
        match (&mut s.draft, &prefix.last) {
            (Some(d), Some(last)) if had_draft => d.adopt_history(b, prefix.position - 1, last)?,
            (Some(_), _) if self.prompt_frontend => s.disable_mtp(),
            _ => {}
        }
        if let Some(remote) = &prefix.remote { self.pp_cache_restore(remote, s.slot, capacity, prefix.position)?; }
        s.pos = prefix.position;
        s.sync_pool_state(b)?;
        s.logits_ready = false;
        if let Some(d) = s.draft.as_ref() { d.bound_at.set(None); }
        s.valid = true;
        Ok(s)
    }
}
