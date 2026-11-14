//! Compact immutable Flash checkpoints; weights and execution scratch stay out.
use super::*;
use std::rc::Rc;

pub(super) struct Recurrent<'a> {
    pub position: usize,
    pub rows: Vec<[i64; 2]>,
    pub ple: Vec<DeviceVector<'a>>,
    pub gdn: Vec<Vec<Option<GdnState<'a>>>>,
    pub mtp: Vec<mtp::SavedDraft<'a>>,
}
impl Recurrent<'_> {
    pub(super) fn bytes(&self) -> usize {
        self.mtp.iter().map(mtp::SavedDraft::bytes).sum::<usize>() + self.ple.iter().map(|v| v.len() * 4).sum::<usize>()
            + self.gdn.iter().flatten().flatten().map(GdnState::device_bytes).sum::<usize>()
            + self.rows.len() * 16 + self.gdn.len() * 48 * std::mem::size_of::<Option<GdnState<'_>>>()
    }
}

pub struct Snapshot<'a> {
    recurrent: Rc<Recurrent<'a>>,
    canonical: Rc<Recurrent<'a>>,
    qsa: Vec<Vec<Option<QsaState<'a>>>>,
    tail: Vec<u32>,
}
impl Snapshot<'_> {
    pub fn prefix_len(&self) -> usize { self.canonical.position }
    pub fn bytes(&self) -> usize {
        self.recurrent.bytes() + if Rc::ptr_eq(&self.recurrent, &self.canonical) { 0 } else { self.canonical.bytes() }
            + self.qsa.iter().flatten().flatten().map(QsaState::device_bytes).sum::<usize>()
            + self.tail.capacity() * 4 + std::mem::size_of::<Self>()
            + self.qsa.len() * 48 * std::mem::size_of::<Option<QsaState<'_>>>()
    }
}
impl<'a, 'w, E: Exchange> Session<'a, 'w, E> {
    /// Only the Flash HTTP service pays for turn checkpoints.
    pub fn enable_cache(&mut self) { self.caching = true; }
    pub(super) fn recurrent(&self, b: &Backend<'a>) -> Result<Recurrent<'a>> {
        Ok(Recurrent { position: self.position, rows: self.rows.iter().map(Rows::history).collect(),
            ple: self.ple.iter().map(|p| p.save_history(b)).collect::<Result<_>>()?,
            mtp: self.save_mtp(b, self.position)?,
            gdn: self.gdn_cache.iter().map(|layers| layers.iter().map(|v|
                v.as_ref().map(|v| v.snapshot(b)).transpose()).collect()).collect::<Result<_>>()? })
    }
    pub(super) fn save_mtp(&self, b: &Backend<'a>, position: usize) -> Result<Vec<mtp::SavedDraft<'a>>> {
        self.mtp.iter().map(|d| d.snapshot(b, position)).collect()
    }
    pub(super) fn restore_mtp(&mut self, b: &Backend<'a>, saved: &[mtp::SavedDraft<'a>]) -> Result<()> {
        // Plain requests restore only canonical target state. Once history was
        // omitted, a cache hit must not silently resume a stale draft prefix.
        if !self.mtp_enabled { return Ok(()); }
        if saved.is_empty() {
            if !self.disable_mtp_history(b)? { return Err("Flash checkpoint restore overlaps verification".into()); }
            return Ok(());
        }
        if self.mtp.len() != saved.len() { return Err("Flash checkpoint is missing MTP history".into()); }
        for (draft, saved) in self.mtp.iter_mut().zip(saved) { draft.restore(b, saved)?; }
        Ok(())
    }
    pub(super) fn restore_recurrent(&mut self, b: &Backend<'a>, saved: &Recurrent<'a>) -> Result<()> {
        // In-place verified rollback keeps captured allocation addresses.
        if self.gdn_cache.iter().zip(&saved.gdn).any(|(live, copies)|
            live.iter().zip(copies).any(|(live, copy)| live.is_none() && copy.is_some())) { self.drop_graphs(); }
        for r in 0..self.ranks.len() {
            self.rows[r].restore_history(saved.rows[r]);
            self.ple[r].restore_history(b, &saved.ple[r])?;
            for (live, copy) in self.gdn_cache[r].iter_mut().zip(&saved.gdn[r]) {
                if let Some(copy) = copy {
                    if let Some(live) = live { copy.copy_into(b, live)?; }
                    else { *live = Some(copy.snapshot(b)?); }
                } else if let Some(live) = live { live.reset(b)?; }
            }
            for state in self.qsa_cache[r].iter_mut().flatten() { state.rewind(saved.position); }
        }
        self.position = saved.position;
        Ok(())
    }
    /// The prefix a snapshot taken now could restore before new prompt work.
    pub fn prefix_len(&self) -> usize { self.canonical.as_ref().map_or(self.position, |c| c.position) }
    /// A lower bound: exclude shared canonical snapshots and execution scratch.
    /// Uncredited live bytes conservatively remain in the growth allowance.
    pub fn private_cache_bytes(&self) -> usize {
        self.ple.len() * Ple::HISTORY_BYTES
            + self.gdn_cache.iter().flatten().flatten().map(GdnState::device_bytes).sum::<usize>()
            + self.qsa_cache.iter().flatten().flatten().map(QsaState::device_bytes).sum::<usize>()
    }
    pub fn snapshot_bytes(&self) -> usize {
        let recurrent = self.gdn_cache.iter().flatten().flatten().map(GdnState::device_bytes).sum::<usize>()
            + self.ranks.len() * Ple::HISTORY_BYTES;
        let current_mtp = self.mtp.iter().map(Draft::snapshot_bytes).sum::<usize>();
        let canonical_mtp = self.canonical.as_ref().map_or(current_mtp, |saved|
            saved.mtp.iter().map(mtp::SavedDraft::bytes).sum());
        // Metadata slack also covers zero-KV ranks and vector alignment.
        current_mtp + canonical_mtp + recurrent * 2 + self.position * self.model.persistent.iter().map(|w|
            12 * (2 * (w.plan.kv_heads.end - w.plan.kv_heads.start) as usize * 256 + 128) * 4).sum::<usize>()
            + self.prefill_tail.len() * 4 + 65536
    }
    pub fn snapshot(&self, b: &Backend<'a>) -> Result<Snapshot<'a>> {
        if !self.valid || !self.caching || self.verification.is_some() { return Err("Flash snapshot needs a committed cached session".into()); }
        let recurrent = match &self.canonical {
            Some(saved) if saved.position == self.position => saved.clone(),
            _ => Rc::new(self.recurrent(b)?),
        };
        let canonical = self.canonical.clone().unwrap_or_else(|| recurrent.clone());
        let qsa = self.qsa_cache.iter().map(|layers| layers.iter().map(|v|
            v.as_ref().map(|v| v.snapshot(b, self.position.max(1))).transpose()).collect()).collect::<Result<_>>()?;
        b.synchronize()?;
        Ok(Snapshot { recurrent, canonical, qsa, tail: self.prefill_tail.clone() })
    }
    pub fn restore(&mut self, b: &Backend<'a>, saved: &Snapshot<'a>, prefix: bool) -> Result<()> {
        self.valid = false;
        self.drop_graphs();
        self.qsa_cache = saved.qsa.iter().map(|layers| layers.iter().map(|v|
            v.as_ref().map(|v| v.snapshot(b, self.capacity)).transpose()).collect()).collect::<Result<_>>()?;
        let recurrent = if prefix { &saved.canonical } else { &saved.recurrent };
        self.restore_recurrent(b, recurrent)?;
        self.restore_mtp(b, &recurrent.mtp)?;
        self.prepare_attention_scratch(b)?;
        self.canonical = Some(saved.canonical.clone());
        self.prefill_tail = if prefix { Vec::new() } else { saved.tail.clone() };
        self.caching = true;
        for r in 0..self.ranks.len() {
            self.cache_bytes[r] = Ple::HISTORY_BYTES
                + self.gdn_cache[r].iter().flatten().map(GdnState::device_bytes).sum::<usize>()
                + self.qsa_cache[r].iter().flatten().map(QsaState::device_bytes).sum::<usize>();
        }
        b.synchronize()?;
        self.valid = true;
        Ok(())
    }
    /// Decode recurrence cannot stand in for prompt chunk arithmetic. Commit
    /// the response as canonical prefill once, after its output has been sent.
    pub fn canonicalize(&mut self, b: &Backend<'a>) -> Result<()> {
        self.commit_verification(b, self.position)?;
        if self.prefill_tail.is_empty() { return Ok(()); }
        let tokens = std::mem::take(&mut self.prefill_tail);
        let saved = self.canonical.clone().ok_or("missing Flash canonical checkpoint")?;
        self.restore_recurrent(b, &saved)?;
        self.restore_mtp(b, &saved.mtp)?;
        self.forward_prefill(b, &tokens, false)?;
        Ok(())
    }
}
