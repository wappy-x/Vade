//! KV pages: chat memory that grows with use. Flash-Next serving reads the `Geometry`:
//! admission charges each chat `Geometry::charge` for its own tokens (flashnext::kv_charge),
//! never the window, and the session maps physical pages only for the rows it writes
//! (NextSession::fit_kv), so the charge bounds its memory. The `Pool` ledger below is the
//! host model of a per-slot page pool (look-ahead claims, idle-slot trims, park and resume);
//! serving keeps its own reservations in the scheduler and does not use `Pool` yet.
//! Every `Pool` decision is a pure function of the page count, the geometry and the
//! operation list: two ranks that agree on the first two and apply the same operations
//! in the same order hold the same ledger, and `digest` proves it.

/// Storage of one K or V head row. Fp32 is what the kernels write and read today.
/// Bf16 and Fp8 round at KV insertion and need new attention reads: a value change,
/// so a separate exactness and quality decision before either can be used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format { Fp32, Bf16, Fp8 }
impl Format {
    /// Bytes of one head row of `values` elements; an 8-bit row keeps one FP32 scale.
    pub fn row_bytes(self, values: u64) -> u64 {
        match self { Format::Fp32 => values.saturating_mul(4), Format::Bf16 => values.saturating_mul(2), Format::Fp8 => values.saturating_add(4) }
    }
}

/// Page geometry of one rank: page bytes, bytes per token of each growing vector, the
/// fixed bytes of one chat (recurrent and draft state, charged in pages with its first
/// page) and the window. Pages for `t` tokens (the window at most): the fixed pages plus,
/// per vector, ceil(t * width / page). Extents grow in `step` tokens up to the window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry { page: u64, vectors: Vec<u64>, fixed: u64, window: u64, step: u64 }
impl Geometry {
    /// Step: the tokens that fill one page of the widest vector (1..=window). Narrower
    /// vectors end a step inside a page; `pages` counts each partial page once.
    pub fn new(page: u64, vectors: Vec<u64>, fixed: u64, window: u64) -> Result<Self, String> {
        let widest = vectors.iter().copied().max().unwrap_or(0).max(1);
        Self::with_step(page, vectors, fixed, window, (page / widest).clamp(1, window.max(1)))
    }
    pub fn with_step(page: u64, vectors: Vec<u64>, fixed: u64, window: u64, step: u64) -> Result<Self, String> {
        if page == 0 || window == 0 || vectors.is_empty() || vectors.contains(&0) || !(1..=window).contains(&step) {
            return Err("KV page geometry needs a page, a window, a step in 1..=window and nonzero widths".into());
        }
        let g = Self { page, vectors, fixed, window, step };
        g.window.checked_next_multiple_of(step).and_then(|t| g.vectors.iter().try_fold(g.fixed.div_ceil(page), |n, &w| n.checked_add(t.checked_mul(w)?)))
            .ok_or("KV page geometry overflows")?;
        Ok(g)
    }
    /// One attention layer's growing vectors on a rank: K and V rows of the KV heads it
    /// keeps, and the FP32 index row (the selection input; it keeps its format). Widths
    /// saturate; the geometry then refuses them as an overflow.
    pub fn attention(kv_heads: u64, head_dim: u64, index_width: u64, format: Format) -> Vec<u64> {
        let kv = kv_heads.saturating_mul(format.row_bytes(head_dim));
        [kv, kv, index_width.saturating_mul(4)].into_iter().filter(|&w| w != 0).collect()
    }
    pub fn page(&self) -> u64 { self.page }
    pub fn step(&self) -> u64 { self.step }
    pub fn window(&self) -> u64 { self.window }
    pub fn token_bytes(&self) -> u64 { self.vectors.iter().sum() }
    pub fn pages(&self, tokens: u64) -> u64 {
        if tokens == 0 { return 0; }
        let t = tokens.min(self.window);
        self.fixed.div_ceil(self.page) + self.vectors.iter().map(|w| (t * w).div_ceil(self.page)).sum::<u64>()
    }
    /// `tokens` rounded up to the step; the window at most. A window that is not a step
    /// multiple ends in a short step, so each slot's virtual range is the window.
    pub fn extent(&self, tokens: u64) -> u64 { tokens.min(self.window).next_multiple_of(self.step).min(self.window) }
    /// Bytes one chat of `tokens` charges at admission, `ahead` rows further (the window at
    /// most): its fixed pages and, per vector, the pages of its extent.
    pub fn charge(&self, tokens: u64, ahead: u64) -> Option<u64> {
        self.pages(self.extent(tokens.checked_add(ahead)?)).checked_mul(self.page)
    }
    /// Identity of the geometry; both ranks must show the same value before serving.
    pub fn digest(&self) -> u64 {
        let mut h = Fnv::new();
        for v in [self.page, self.fixed, self.window, self.step, self.vectors.len() as u64] { h.word(v); }
        for &v in &self.vectors { h.word(v); }
        h.0
    }
}

/// FNV-1a over 64-bit words, little endian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fnv(u64);
impl Fnv {
    fn new() -> Self { Self(0xcbf2_9ce4_8422_2325) }
    fn word(&mut self, v: u64) { for b in v.to_le_bytes() { self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01b3); } }
}

/// Device work for the backend, in order: unmap an idle slot's vectors down to `to`
/// tokens, or map a slot's vectors from `from` up to `to` tokens (both extents, so the
/// window at most).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Map { Trim { slot: usize, from: u64, to: u64 }, Grow { slot: usize, from: u64, to: u64 } }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Every slot holds a live chat.
    NoSlot,
    /// The claims would exceed the pool now: `need` pages, `left` unclaimed. `need` is at
    /// most the pool, so the chat fits when other chats end.
    Full { need: u64, left: u64 },
    /// The chat must wait for pages before its length passes its claim. Only a chat that
    /// shares the pool waits (`victim` breaks a full stall); a lone chat never waits.
    Wait,
    /// The length alone needs `need` pages, more than the `total` of the pool: the chat
    /// never fits. End it with a length stop; do not queue it. When the length counts
    /// drafted rows, first retry with fewer draft rows (also on a Wait).
    TooLong { need: u64, total: u64 },
    Invalid(&'static str),
}

/// A parked chat: no slot, no claim, no pages. Its state lives outside the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parked { pub len: u64, pub limit: u64 }

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Slot { live: bool, mapped: u64, len: u64, limit: u64, covered: u64, claim: u64, seq: u64 }

/// One rank's page pool. `total` pages; `free` pages are not mapped to any slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pool { geometry: Geometry, total: u64, free: u64, claimed: u64, ahead: u64, slots: Vec<Slot>, seq: u64, digest: Fnv }
impl Pool {
    /// `ahead`: tokens each claim covers past the chat's length (1..=window).
    pub fn new(geometry: Geometry, total: u64, slots: usize, ahead: u64) -> Result<Self, String> {
        if slots == 0 || !(1..=geometry.window).contains(&ahead) { return Err("a KV page pool needs a slot and a look-ahead in 1..=window".into()); }
        let mut digest = Fnv::new();
        for v in [geometry.digest(), total, slots as u64, ahead] { digest.word(v); }
        Ok(Self { geometry, total, free: total, claimed: 0, ahead, slots: vec![Slot::default(); slots], seq: 0, digest })
    }
    /// Pages in the pool from free bytes: the same rule on every rank.
    pub fn pages_for_bytes(geometry: &Geometry, bytes: u64) -> u64 { bytes / geometry.page }
    /// The ranks' common page count: the smaller pool; geometries must be identical.
    pub fn agree(local: (u64, u64), peer: (u64, u64)) -> Result<u64, String> {
        if local.1 != peer.1 { return Err(format!("KV page geometry differs between ranks: {:#x} vs {:#x}", local.1, peer.1)); }
        Ok(local.0.min(peer.0))
    }
    pub fn geometry(&self) -> &Geometry { &self.geometry }
    pub fn total(&self) -> u64 { self.total }
    pub fn free(&self) -> u64 { self.free }
    /// Pages no live claim covers: the room for new claims.
    pub fn unclaimed(&self) -> u64 { self.total - self.claimed }
    pub fn live(&self) -> usize { self.slots.iter().filter(|s| s.live).count() }
    pub fn digest(&self) -> u64 { self.digest.0 }
    /// (length, claim pages, mapped pages) of a live slot.
    pub fn chat(&self, slot: usize) -> Option<(u64, u64, u64)> {
        self.slots.get(slot).filter(|s| s.live).map(|s| (s.len, s.claim, self.geometry.pages(s.mapped)))
    }
    fn record(&mut self, op: u64, args: [u64; 3], outcome: u64) {
        for v in [op, args[0], args[1], args[2], outcome, self.free, self.claimed] { self.digest.word(v); }
    }
    fn claim_for(&self, covered: u64) -> u64 { self.geometry.pages(self.geometry.extent(covered)) }
    /// The longest coverage from `len` up to `want` whose claim fits in `room` pages:
    /// `want`, or else the longest extent below it (the claim grows with the extent).
    /// None when `len` alone does not fit.
    fn fit(&self, len: u64, want: u64, room: u64) -> Option<u64> {
        let g = &self.geometry;
        if self.claim_for(want) <= room { return Some(want); }
        let (mut lo, mut hi) = (g.extent(len), g.extent(want));
        if g.pages(lo) > room { return None; }
        // pages(lo) fits and pages(hi) does not; lo is a step multiple below hi.
        while hi - lo > g.step {
            let mid = lo + (hi - lo).div_ceil(g.step) / 2 * g.step;
            if g.pages(mid) <= room { lo = mid } else { hi = mid }
        }
        Some(lo)
    }
    /// Admit a chat of `prompt` tokens that may reach `limit` tokens (prompt plus its
    /// output limit; the window at most). Returns the slot and the device work.
    pub fn admit(&mut self, prompt: u64, limit: u64) -> Result<(usize, Vec<Map>), Refusal> {
        let result = self.admit_inner(prompt, limit);
        let outcome = result.as_ref().map_or(u64::MAX, |r| r.0 as u64);
        self.record(1, [prompt, limit, 0], outcome);
        result
    }
    fn admit_inner(&mut self, prompt: u64, limit: u64) -> Result<(usize, Vec<Map>), Refusal> {
        let limit = limit.min(self.geometry.window);
        if prompt == 0 || prompt > limit { return Err(Refusal::Invalid("a chat needs 1..=limit prompt tokens")); }
        let need = self.claim_for(prompt);
        if need > self.total { return Err(Refusal::TooLong { need, total: self.total }); }
        // The full look-ahead, the pool at most: the chat moves `ahead` tokens (or to the
        // pool edge) before it can wait, and an empty pool always takes it.
        let covered = self.fit(prompt, prompt.saturating_add(self.ahead).min(limit), self.total).unwrap_or(prompt);
        let claim = self.claim_for(covered);
        if claim > self.unclaimed() { return Err(Refusal::Full { need: claim, left: self.unclaimed() }); }
        let (top, target, g) = (self.geometry.extent(covered), self.geometry.extent(prompt), &self.geometry);
        // The idle slot with the least device work: pages to map for the prompt plus pages
        // to unmap down to the claim. Ties take the lower index.
        let work = |m: u64| g.pages(target) - g.pages(m.min(target)) + g.pages(m) - g.pages(m.min(top));
        let slot = self.slots.iter().enumerate().filter(|(_, s)| !s.live).min_by_key(|&(i, s)| (work(s.mapped), i))
            .map(|(i, _)| i).ok_or(Refusal::NoSlot)?;
        let mut work = Vec::new();
        let mapped = self.slots[slot].mapped;
        if mapped > top { self.unmap(slot, top, &mut work); }
        self.seq += 1;
        self.slots[slot] = Slot { live: true, mapped: self.slots[slot].mapped, len: prompt, limit, covered, claim, seq: self.seq };
        self.claimed += claim;
        self.map_to(slot, self.geometry.extent(prompt), &mut work);
        Ok((slot, work))
    }
    /// Set a live chat's length (drafted rows included). Past its claim the claim
    /// extends first: the look-ahead shrinks to the unclaimed pages, and the chat waits
    /// (nothing changes) only when its length alone does not fit. A length that can never
    /// fit is TooLong. A shorter length (a rewind) keeps the pages.
    pub fn grow(&mut self, slot: usize, len: u64) -> Result<Vec<Map>, Refusal> {
        let result = self.grow_inner(slot, len);
        let outcome = result.as_ref().map_or(u64::MAX, |w| w.len() as u64);
        self.record(2, [slot as u64, len, 0], outcome);
        result
    }
    fn grow_inner(&mut self, slot: usize, len: u64) -> Result<Vec<Map>, Refusal> {
        let s = *self.slots.get(slot).filter(|s| s.live).ok_or(Refusal::Invalid("no live chat in this slot"))?;
        if len == 0 || len > s.limit { return Err(Refusal::Invalid("a chat length must be 1..=its limit")); }
        if len > s.covered {
            let need = self.claim_for(len);
            if need > self.total { return Err(Refusal::TooLong { need, total: self.total }); }
            let covered = self.fit(len, len.saturating_add(self.ahead).min(s.limit), s.claim + self.unclaimed()).ok_or(Refusal::Wait)?;
            let claim = self.claim_for(covered);
            self.claimed += claim - s.claim;
            (self.slots[slot].covered, self.slots[slot].claim) = (covered, claim);
        }
        self.slots[slot].len = len;
        let mut work = Vec::new();
        self.map_to(slot, self.geometry.extent(len), &mut work);
        Ok(work)
    }
    /// End a chat. Its slot keeps the mapped pages for the next chat until the pool needs them.
    pub fn release(&mut self, slot: usize) -> Result<(), Refusal> {
        let result = self.end(slot).map(|_| ());
        self.record(3, [slot as u64, 0, 0], result.is_ok() as u64);
        result
    }
    /// Park a chat (its state goes to the second tier): as release, and keep its record.
    pub fn park(&mut self, slot: usize) -> Result<Parked, Refusal> {
        let result = self.end(slot);
        self.record(4, [slot as u64, 0, 0], result.is_ok() as u64);
        result
    }
    /// Bring a parked chat back: an admission at its stored length, with the full
    /// look-ahead (the pool at most). So a chat parked because every chat waited is Full
    /// until another chat ends, and an empty pool always takes it back. Resume parked
    /// chats only after a round in which no live chat waits.
    pub fn resume(&mut self, parked: Parked) -> Result<(usize, Vec<Map>), Refusal> {
        let result = self.admit_inner(parked.len, parked.limit);
        let outcome = result.as_ref().map_or(u64::MAX, |r| r.0 as u64);
        self.record(6, [parked.len, parked.limit, 0], outcome);
        result
    }
    fn end(&mut self, slot: usize) -> Result<Parked, Refusal> {
        let s = *self.slots.get(slot).filter(|s| s.live).ok_or(Refusal::Invalid("no live chat in this slot"))?;
        self.claimed -= s.claim;
        self.seq += 1;
        self.slots[slot] = Slot { mapped: s.mapped, seq: self.seq, ..Slot::default() };
        Ok(Parked { len: s.len, limit: s.limit })
    }
    /// The chat to evict or park when every live chat waits: the newest admission. Waits
    /// need two or more live chats, so the other chats then grow. Resume the parked chat
    /// only after a round in which no live chat waits (see `resume`).
    pub fn victim(&self) -> Option<usize> {
        self.slots.iter().enumerate().filter(|(_, s)| s.live).max_by_key(|(_, s)| s.seq).map(|(i, _)| i)
    }
    /// Return idle mappings to the device until `free` pages are unmapped (memory pressure).
    pub fn trim(&mut self, free: u64) -> Vec<Map> {
        let mut work = Vec::new();
        self.take(free, &mut work);
        self.record(5, [free, 0, 0], work.len() as u64);
        work
    }
    fn unmap(&mut self, slot: usize, to: u64, work: &mut Vec<Map>) {
        let from = self.slots[slot].mapped;
        self.free += self.geometry.pages(from) - self.geometry.pages(to);
        self.slots[slot].mapped = to;
        work.push(Map::Trim { slot, from, to });
    }
    /// Unmap idle slots, least recently ended first, until `need` pages are free. Each
    /// slot keeps the longest step multiple that still frees enough.
    fn take(&mut self, need: u64, work: &mut Vec<Map>) {
        while self.free < need {
            let Some(i) = self.slots.iter().enumerate().filter(|(_, s)| !s.live && s.mapped != 0).min_by_key(|(_, s)| s.seq).map(|(i, _)| i) else { return; };
            let (g, mapped) = (&self.geometry, self.slots[i].mapped);
            let keep = g.pages(mapped).saturating_sub(need - self.free);
            let to = (0..mapped.div_ceil(g.step)).rev().map(|n| n * g.step).find(|&e| g.pages(e) <= keep).unwrap_or(0);
            self.unmap(i, to, work);
        }
    }
    fn map_to(&mut self, slot: usize, to: u64, work: &mut Vec<Map>) {
        let from = self.slots[slot].mapped;
        if to <= from { return; }
        let need = self.geometry.pages(to) - self.geometry.pages(from);
        self.take(need, work);
        // Claims never exceed the pool and a live mapping stays inside its claim.
        assert!(self.free >= need, "KV page claim invariant broken");
        self.free -= need;
        self.slots[slot].mapped = to;
        work.push(Map::Grow { slot, from, to });
    }
    #[cfg(test)]
    fn check(&self) {
        let mapped: u64 = self.slots.iter().map(|s| self.geometry.pages(s.mapped)).sum();
        assert_eq!(self.free + mapped, self.total);
        assert_eq!(self.claimed, self.slots.iter().filter(|s| s.live).map(|s| s.claim).sum::<u64>());
        assert!(self.claimed <= self.total);
        for s in self.slots.iter().filter(|s| s.live) {
            assert!(self.geometry.pages(s.mapped) <= s.claim && s.len <= s.covered && s.covered <= s.limit);
        }
    }
}

/// Slots to fund at start: the most (1..=`max`) whose shared reserve still leaves
/// one smallest claim per slot. Returns (slots, pages). Every rank runs the same rule;
/// the ranks then agree on the smaller of each.
pub fn plan_slots(geometry: &Geometry, available: u64, ahead: u64, max: usize, mut shared: impl FnMut(usize) -> u64) -> (usize, u64) {
    let smallest = geometry.pages(geometry.extent(ahead.saturating_add(1)));
    for slots in (1..=max.max(1)).rev() {
        let pages = Pool::pages_for_bytes(geometry, available.saturating_sub(shared(slots)));
        if slots == 1 || smallest.checked_mul(slots as u64).is_some_and(|n| n <= pages) { return (slots, pages); }
    }
    unreachable!()
}

#[cfg(test)]
#[path = "kv_pages_tests.rs"]
mod tests;
