//! Elastic token-prefix snapshots. Forks preserve source KV.
//! Session keys are bearer capabilities and must not be logged or shared.
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    io::Read,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// An opaque bearer capability. Debug deliberately never prints its value.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SessionKey([u8; 32]);
impl fmt::Debug for SessionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SessionKey(<redacted>)")
    }
}
impl SessionKey {
    /// Draw from the operating system random device, never time or a counter.
    #[cfg(unix)]
    pub fn generate() -> Result<Self, String> {
        let mut bytes = [0; 32];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut bytes))
            .map_err(|_| "operating system randomness unavailable".to_string())?;
        Ok(Self(bytes))
    }
    #[cfg(not(unix))]
    pub fn generate() -> Result<Self, String> {
        Err("session capability generation requires Unix".into())
    }
    pub fn parse(text: &str) -> Result<Self, String> {
        if text.len() != 64 {
            return Err("invalid session capability".into());
        }
        let mut out = [0; 32];
        fn digit(b: u8) -> Option<u8> {
            match b {
                b'0'..=b'9' => Some(b - b'0'),
                b'a'..=b'f' => Some(b - b'a' + 10),
                _ => None,
            }
        }
        for (dst, pair) in out.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
            *dst = digit(pair[0]).ok_or("invalid session capability")? * 16
                + digit(pair[1]).ok_or("invalid session capability")?;
        }
        Ok(Self(out))
    }
    /// For the client response only. Avoid recording this string in diagnostics.
    pub fn expose(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for b in self.0 {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 15) as usize] as char);
        }
        out
    }
    pub(crate) fn matches(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0).fold(0u8, |different, (a, b)| different | (a ^ b)) == 0
    }
}

/// `tokens` is exactly the prefix already committed into session KV; logits
/// predict the next token after that prefix. A yielded but unstepped token is
/// intentionally absent. The engine adapter supplies a conservative bound for
/// all session-owned allocations, including spare vector capacity, in bytes.
pub struct CachedSession<S> {
    /// Pins the source checkpoint while this fork remains a live request.
    pub checkpoint_lease: Option<Arc<()>>,
    pub session: S,
    pub tokens: Vec<u32>,
    pub next_logits: Vec<f32>,
    pub reserved_session_bytes: usize,
}
pub(crate) fn shared_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}
impl<S: crate::Session> CachedSession<S> {
    pub(crate) fn fork_prefix(&self, prompt: &[u32]) -> Result<Option<Self>, String> {
        self.fork_prefix_limit(prompt, self.tokens.len())
    }
    /// Logical retained rows and rows restored without recurrent replay.
    fn restorable_prefix(&self, prompt: &[u32], limit: usize) -> Option<(usize, usize)> {
        let shared = shared_prefix(&self.tokens, prompt).min(limit);
        let exact = shared == self.tokens.len() && !self.next_logits.is_empty();
        let retain = if shared == prompt.len() && !exact { shared.saturating_sub(1) } else { shared };
        // An exact replay restores the whole prompt only where the backend can fork there;
        // otherwise (a recurrent checkpoint below the prompt end) it replays from that checkpoint.
        let restored = if exact && shared == prompt.len() && self.session.can_fork_prefix(retain) { retain }
            else { self.session.checkpoint_reusable_tokens(retain).min(retain) };
        // Recurrent state cannot rewind between snapshots. Replay the verified
        // suffix from the selected canonical boundary instead of claiming it.
        let retain = if self.session.can_fork_prefix(retain) { retain } else { restored };
        if retain == 0 || !self.session.can_fork_prefix(retain) { return None; }
        Some((retain, restored))
    }
    pub(crate) fn fork_prefix_limit(&self, prompt: &[u32], limit: usize) -> Result<Option<Self>, String> {
        let Some((retain, _)) = self.restorable_prefix(prompt, limit) else { return Ok(None); };
        let session = if retain == self.tokens.len() { self.session.fork()? }
            else { self.session.fork_prefix(retain)? };
        Ok(Some(Self {
            checkpoint_lease: self.checkpoint_lease.clone(),
            session, tokens: self.tokens[..retain].to_vec(),
            next_logits: if retain == self.tokens.len() && retain == prompt.len() { self.next_logits.clone() } else { Vec::new() },
            reserved_session_bytes: self.reserved_session_bytes,
        }))
    }
}

// Hashes select candidates only. Scope and every reused token are compared
// exactly, so a collision can never share another prompt or API key's KV.
fn prefix_blocks(tokens: &[u32]) -> impl Iterator<Item = (usize, u64)> + '_ {
    let mut hash = 0xcbf29ce484222325u64;
    tokens.iter().enumerate().filter_map(move |(i, &token)| {
        hash = (hash ^ u64::from(token)).wrapping_mul(0x100000001b3);
        (i == 0 || (i + 1) % 64 == 0).then_some((i + 1, hash))
    })
}
fn scope_hash(scope: Option<&str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    scope.hash(&mut hash);
    hash.finish()
}
/// Length framing keeps authorization and salt exact, including absent values.
/// Hashes still select candidates only; stored namespaces are compared in full.
pub(crate) fn namespace(authorization: Option<&str>, salt: Option<&str>) -> Option<String> {
    if authorization.is_none() && salt.is_none() { return None; }
    Some(format!("{}:{}{}:{}", authorization.map_or(0, |s| s.len() + 1),
        authorization.unwrap_or(""), salt.map_or(0, |s| s.len() + 1), salt.unwrap_or("")))
}
fn index_charge(tokens: usize, scope: Option<&str>) -> usize {
    // Conservative node/key/vector bound per indexed block, charged even when
    // another snapshot already owns the bucket. Includes allocation identities.
    index_metadata_bytes(tokens, scope.map_or(0, str::len)).expect("bounded cache index metadata")
}
fn index_metadata_bytes(tokens: usize, scope: usize) -> Option<usize> {
    (1 + tokens / 64).checked_mul(160).and_then(|n| n.checked_add(scope))
}
/// Bound one automatic entry's private charge, separate from its shared
/// snapshot allocations. The adapter supplies its own private storage bound,
/// snapshot identity count and maximum namespace length.
pub fn publication_metadata_bytes(tokens: usize, logits: usize, private: usize,
    allocations: usize, scope: usize) -> Option<usize> {
    tokens.checked_mul(4)
        .and_then(|n| logits.checked_mul(4).and_then(|m| n.checked_add(m)))
        .and_then(|n| n.checked_add(private))
        .and_then(|n| index_metadata_bytes(tokens, scope).and_then(|m| n.checked_add(m)))
        .and_then(|n| allocations.checked_mul(96).and_then(|m| n.checked_add(m)))
}
#[derive(Clone, Debug)]
pub struct CacheLimits {
    pub entries: usize,
    pub retained_bytes: usize,
    pub tokens_per_entry: usize,
    pub logits_per_entry: usize,
    pub idle_ttl: Duration,
}
impl CacheLimits {
    pub(crate) fn checkpoint_entries(&self) -> usize { self.entries.min(16) }
    pub(crate) fn checkpoint_bytes(&self) -> usize { self.retained_bytes / 2 }
    fn validate(&self) -> Result<(), String> {
        // Zero retained bytes disables snapshots; insertions still enforce it.
        if self.entries == 0
            || self.entries > 128
            || self.tokens_per_entry == 0
            || self.tokens_per_entry > 262144
            || self.logits_per_entry == 0
            || self.logits_per_entry > 1_048_576
            || self.idle_ttl.is_zero()
            || self.idle_ttl > Duration::from_secs(86400)
        {
            return Err("invalid continuation cache limits".into());
        }
        Ok(())
    }
}
/// Issuance metadata, not a promise against subsequent pressure eviction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub handle: SessionKey,
    pub prompt_tokens: usize,
    pub reusable_prefix_tokens: usize,
    pub expires_at_ms: u64,
    pub retained_bytes: usize,
    pub scope_retained_bytes: usize,
}
struct Entry<S> {
    id: u64,
    scope: Option<String>,
    owner: Option<String>,
    key: SessionKey,
    state: CachedSession<S>,
    saved: Instant,
    charge: usize,
    allocations: Vec<(usize, usize)>,
    checkpoint: Option<Instant>,
    children: Arc<()>,
}
impl<S> Entry<S> {
    fn pinned(&self) -> bool { self.checkpoint.is_some() && Arc::strong_count(&self.children) > 1 }
}
fn checkpoint_metadata<S>(scope: Option<&str>, owner: Option<&str>) -> usize {
    std::mem::size_of::<Entry<S>>() + 32 + scope.map_or(0, str::len) + owner.map_or(0, str::len)
}
/// Retained payload borrows memory not reserved by runnable/retired requests.
/// A fixed maximum entry count also bounds metadata; this type requires no
/// Send/Sync and preserves any thread affinity of the contained backend state.
pub struct SessionCache<S> {
    limits: CacheLimits,
    entries: VecDeque<Entry<S>>,
    retained_bytes: usize,
    available_bytes: usize,
    funded_pages: BTreeSet<usize>,
    allocations: BTreeMap<usize, (usize, usize)>,
    prefixes: BTreeMap<(u64, usize, u64), Vec<u64>>,
    next_id: u64,
    next_checkpoint_expiry: Option<Instant>,
}
impl<S> SessionCache<S> {
    fn valid_allocations(&self, allocations: &[(usize, usize)]) -> bool {
        let mut identities = BTreeSet::new();
        allocations.iter().all(|&(id, bytes)| id != 0 && bytes != 0 && identities.insert(id)
            && self.allocations.get(&id).is_none_or(|&(prior, _)| prior == bytes))
    }
    pub fn new(limits: CacheLimits) -> Result<Self, String> {
        limits.validate()?;
        Ok(Self {
            available_bytes: limits.retained_bytes,
            funded_pages: BTreeSet::new(),
            limits,
            entries: VecDeque::new(),
            retained_bytes: 0,
            allocations: BTreeMap::new(),
            prefixes: BTreeMap::new(),
            next_id: 0,
            next_checkpoint_expiry: None,
        })
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
    /// Only immutable whole pages already funded by requests can be credited.
    pub(crate) fn unfunded_bytes(&self) -> usize {
        let funded = self.allocations.iter().filter(|(id, _)| self.funded_pages.contains(id))
            .map(|(_, (bytes, _))| bytes).sum::<usize>();
        self.retained_bytes - funded
    }
    /// Newly committed pages still lie within the source's active reservation.
    pub(crate) fn fund_pages(&mut self, pages: Vec<(usize, usize)>) {
        self.funded_pages.extend(pages.into_iter().map(|(id, _)| id));
    }
    #[cfg(test)]
    pub(crate) fn rebalance(&mut self, bytes: usize, pages: BTreeSet<usize>, now: Instant) where S: crate::Session {
        self.rebalance_within(bytes, None, pages, now)
    }
    /// `free`: physical bytes the engine can spare beyond its own needs (negative: short) and its
    /// growth slack (Engine::cache_free_physical_bytes). Unfunded retention yields down to what it
    /// holds plus the first; inserts must also leave the slack free.
    pub(crate) fn rebalance_within(&mut self, bytes: usize, free: Option<(i128, usize)>, pages: BTreeSet<usize>, now: Instant)
        where S: crate::Session {
        let entries = self.entries.len();
        self.expire_checkpoints(now);
        self.funded_pages = pages;
        let physical = free.map(|(free, _)| (self.unfunded_bytes() as i128 + free).clamp(0, usize::MAX as i128) as usize);
        self.available_bytes = physical.map_or(bytes, |p| p.min(bytes)).min(self.limits.retained_bytes);
        self.yield_to(bytes, physical, now, entries);
        // Retention grows only while the engine's slack also stays free; it yields only past
        // the physical room itself, so pages mapped inside the slack evict nothing.
        if let (Some(p), Some((_, slack))) = (physical, free) {
            self.available_bytes = self.available_bytes.min(p.saturating_sub(slack));
        }
    }
    fn yield_to(&mut self, bytes: usize, physical: Option<usize>, now: Instant, entries: usize) where S: crate::Session {
        if self.unfunded_bytes() <= self.available_bytes { return; }
        self.expire(now);
        if self.unfunded_bytes() > self.available_bytes {
            // Proposal pages yield before canonical cached target prefixes.
            for index in 0..self.entries.len() { self.release_draft(index); }
        }
        // Exact-replay snapshots yield next, oldest first, before whole entries.
        for index in 0..self.entries.len() {
            if self.unfunded_bytes() <= self.available_bytes { break; }
            self.release_replay(index);
        }
        while self.unfunded_bytes() > self.available_bytes {
            let Some(index) = self.entries.iter().position(|e| !e.pinned()) else { break; };
            self.remove(index);
        }
        if self.entries.len() != entries {
            eprintln!("prefix_cache_pressure evicted_entries={} retained_bytes={} unfunded_bytes={} available_bytes={}",
                entries - self.entries.len(), self.retained_bytes, self.unfunded_bytes(), self.available_bytes);
            if let Some(p) = physical.filter(|&p| p < bytes.min(self.limits.retained_bytes)) {
                eprintln!("prefix_cache_yield physical_bytes={p} pool_bytes={bytes}");
            }
        }
    }
    /// Charge shared pages once, evicting until the incoming snapshot fits.
    /// Exact-replay snapshots yield, oldest entry first, before any whole entry.
    fn make_room(&mut self, charge: usize, allocations: &[(usize, usize)]) -> bool where S: crate::Session {
        if allocations.iter().filter(|(id, _)| !self.funded_pages.contains(id))
            .try_fold(charge, |sum, (_, bytes)| sum.checked_add(*bytes))
            .is_none_or(|bytes| bytes > self.available_bytes) { return false; }
        let mut replay = 0;
        loop {
            let added = allocations.iter().filter(|(id, _)|
                !self.allocations.contains_key(id) && !self.funded_pages.contains(id))
                .try_fold(charge, |sum, (_, bytes)| sum.checked_add(*bytes));
            if self.entries.len() < self.limits.entries
                && added.is_some_and(|bytes| bytes <= self.available_bytes.saturating_sub(self.unfunded_bytes())) { return true; }
            if self.entries.is_empty() { return false; }
            if self.entries.len() < self.limits.entries && replay < self.entries.len() {
                let retained = self.retained_bytes;
                self.release_replay(replay);
                if self.retained_bytes != retained {
                    eprintln!("prefix_cache_make_room released_replay_id={} freed_bytes={} retained_bytes={} incoming_bytes={charge}",
                        self.entries[replay].id, retained - self.retained_bytes, self.retained_bytes);
                }
                replay += 1;
                continue;
            }
            let retained = self.retained_bytes;
            let Some(index) = self.entries.iter().position(|e| !e.pinned()) else { return false; };
            let victim = self.remove(index);
            eprintln!("prefix_cache_make_room evicted_id={} evicted_tokens={} freed_bytes={} retained_bytes={} incoming_bytes={charge}",
                victim.id, victim.state.tokens.len(), retained - self.retained_bytes, self.retained_bytes);
        }
    }
    fn remove(&mut self, index: usize) -> Entry<S> {
        let entry = self
            .entries
            .remove(index)
            .expect("cache index came from its own entries");
        self.retained_bytes -= entry.charge;
        if entry.checkpoint.is_none() {
            let scope = scope_hash(entry.scope.as_deref());
            for (position, hash) in prefix_blocks(&entry.state.tokens) {
                let key = (scope, position, hash);
                let ids = self.prefixes.get_mut(&key).unwrap();
                ids.retain(|&id| id != entry.id);
                if ids.is_empty() { self.prefixes.remove(&key); }
            }
        }
        for &(id, _) in &entry.allocations {
            let allocation = self.allocations.get_mut(&id).unwrap();
            allocation.1 -= 1;
            if allocation.1 == 0 { self.retained_bytes -= allocation.0; self.allocations.remove(&id); }
        }
        entry
    }
    pub fn expire(&mut self, now: Instant) {
        let ttl = self.limits.idle_ttl;
        // Test every entry, so even callers supplying nonmonotonic instants
        // cannot accidentally protect an older entry behind a newer one.
        let mut i = 0;
        while i < self.entries.len() {
            if !self.entries[i].pinned() && self.entries[i].checkpoint.map_or_else(|| now.saturating_duration_since(self.entries[i].saved) >= ttl,
                |expiry| now >= expiry) {
                self.remove(i);
            } else {
                i += 1;
            }
        }
    }
    fn expire_checkpoints(&mut self, now: Instant) {
        if self.next_checkpoint_expiry.is_none_or(|deadline| now < deadline) { return; }
        self.next_checkpoint_expiry = None;
        let mut i = 0;
        while i < self.entries.len() {
            if !self.entries[i].pinned() && self.entries[i].checkpoint.is_some_and(|expiry| now >= expiry) { self.remove(i); }
            else {
                if let Some(expiry) = self.entries[i].checkpoint { self.track_checkpoint_expiry(expiry); }
                i += 1;
            }
        }
    }
    fn track_checkpoint_expiry(&mut self, deadline: Instant) {
        self.next_checkpoint_expiry = Some(self.next_checkpoint_expiry.map_or(deadline, |old| old.min(deadline)));
    }
    /// Prefer superseded conversation prefixes to another conversation's latest
    /// checkpoint. Keep LRU order among equally useful entries. Authorization
    /// still owns the quota; namespaces and exact tokens establish ancestry.
    fn checkpoint_victim(&self, owner: Option<&str>, scope: Option<&str>, tokens: &[u32]) -> Option<usize> {
        self.entries.iter().enumerate()
            .filter(|(_, e)| e.checkpoint.is_some() && !e.pinned() && e.owner.as_deref() == owner)
            .min_by_key(|(index, e)| {
                let superseded = (e.scope.as_deref() == scope && tokens.starts_with(&e.state.tokens))
                    || self.entries.iter().any(|newer| newer.checkpoint.is_some()
                        && newer.id > e.id && newer.owner == e.owner && newer.scope == e.scope
                        && newer.state.tokens.starts_with(&e.state.tokens));
                (!superseded, *index)
            })
            .map(|(index, _)| index)
    }
    /// Quotas apply to the caller's checkpoints even when active requests fund
    /// their pages. Shared physical storage is charged once within each owner.
    fn checkpoint_room(&mut self, owner: Option<&str>, scope: Option<&str>, tokens: &[u32],
        charge: usize, allocations: &[(usize, usize)]) -> bool {
        let limit = self.limits.checkpoint_bytes();
        if allocations.iter().try_fold(charge, |sum, (_, bytes)| sum.checked_add(*bytes))
            .is_none_or(|bytes| bytes > limit) { return false; }
        loop {
            let (count, bytes) = self.checkpoint_charge(owner, charge, allocations);
            if count < self.limits.checkpoint_entries() && bytes <= limit { return true; }
            let Some(index) = self.checkpoint_victim(owner, scope, tokens)
                else { return false; };
            self.remove(index);
        }
    }
    fn checkpoint_charge(&self, scope: Option<&str>, charge: usize, allocations: &[(usize, usize)]) -> (usize, usize) {
        let mut pages = BTreeMap::new();
        let (mut bytes, mut count) = (charge, 0);
        for entry in self.entries.iter().filter(|e| e.checkpoint.is_some() && e.owner.as_deref() == scope) {
            bytes = bytes.saturating_add(entry.charge);
            count += 1;
            pages.extend(entry.allocations.iter().copied());
        }
        pages.extend(allocations.iter().copied());
        (count, pages.values().fold(bytes, |n, b| n.saturating_add(*b)))
    }
    /// Idempotent and non-disclosing, including foreign/expired/unknown handles.
    pub(crate) fn release_checkpoint(&mut self, key: &SessionKey, scope: Option<&str>) {
        if let Some(index) = self.entries.iter().position(|e| e.checkpoint.is_some()
            && e.owner.as_deref() == scope && e.key.matches(key)) { self.remove(index); }
    }
    /// Multiple message boundaries may belong to the same conversation.
    #[cfg(test)]
    fn insert_snapshot(
        &mut self,
        key: SessionKey,
        state: CachedSession<S>,
        now: Instant,
    ) -> Result<(), String> where S: crate::Session {
        self.insert_scoped(key, None, state, now)
    }
    fn insert_scoped(&mut self, key: SessionKey, scope: Option<&str>, state: CachedSession<S>, now: Instant)
        -> Result<(), String> where S: crate::Session {
        self.insert_entry(key, scope, None, state, now, None)
    }
    fn insert_entry(&mut self, key: SessionKey, scope: Option<&str>, owner: Option<&str>, state: CachedSession<S>, now: Instant,
        checkpoint: Option<Instant>) -> Result<(), String> where S: crate::Session {
        if !state.session.can_fork()
            || state.session.checkpoint_reusable_tokens(state.tokens.len()) > state.tokens.len()
            || state.tokens.is_empty()
            || state.tokens.len() > self.limits.tokens_per_entry
            || state.next_logits.len() > self.limits.logits_per_entry
            || state.next_logits.iter().any(|v| !v.is_finite())
            || state.reserved_session_bytes == 0
        {
            return Err("invalid continuation cache state".into());
        }
        let allocations = state.session.retained_allocations();
        let reserved = if allocations.is_some() { state.session.retained_private_bytes() } else { state.reserved_session_bytes };
        let allocations = allocations.unwrap_or_default();
        if !self.valid_allocations(&allocations) {
            return Err("invalid continuation allocation identity".into());
        }
        let charge = state
            .tokens
            .capacity()
            .checked_mul(std::mem::size_of::<u32>())
            .and_then(|n| {
                state
                    .next_logits
                    .capacity()
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| n.checked_add(reserved))
            .and_then(|n| n.checked_add(if checkpoint.is_some() { checkpoint_metadata::<S>(scope, owner) }
                else { index_charge(state.tokens.len(), scope) }))
            .and_then(|n| allocations.len().checked_mul(96).and_then(|m| n.checked_add(m)))
            .ok_or("continuation cache charge overflow")?;
        if charge > self.limits.retained_bytes {
            return Err(crate::budget::session_refusal(charge as u128, self.limits.retained_bytes, &format!("private retained state {charge} bytes x 1 session; CONTEXT={} tokens", state.tokens.len()), "the retained-cache ceiling"));
        }
        if checkpoint.is_some() {
            self.expire_checkpoints(now);
            if !self.checkpoint_room(owner, scope, &state.tokens, charge, &allocations) { return Err(crate::budget::session_refusal(allocations.iter().fold(charge as u128, |n, (_, b)| n + *b as u128), self.limits.checkpoint_bytes(), &format!("checkpoint state {charge} private bytes + shared pages x 1 session; CONTEXT={} tokens; checkpoint entry ceiling={}", state.tokens.len(), self.limits.checkpoint_entries()), "the checkpoint scope's retained-cache quota")); }
        } else {
            if self.covered(&key, scope, &state.tokens, !state.next_logits.is_empty()) { return Ok(()); }
            self.drop_prefixes(&key, scope, &state.tokens, false);
        }
        if !self.make_room(charge, &allocations) { return Err(crate::budget::session_refusal(allocations.iter().filter(|(id, _)| !self.funded_pages.contains(id)).fold(charge as u128, |n, (_, b)| n + *b as u128), self.available_bytes, &format!("retained state {charge} private bytes + unfunded shared pages x 1 session; CONTEXT={} tokens; cache entry ceiling={}", state.tokens.len(), self.limits.entries), "the retained-cache pool after live reservations and eviction")); }
        self.retained_bytes += charge;
        for &(id, bytes) in &allocations {
            let allocation = self.allocations.entry(id).or_insert_with(|| { self.retained_bytes += bytes; (bytes, 0) });
            allocation.1 += 1;
        }
        self.next_id += 1;
        let id = self.next_id;
        if checkpoint.is_none() {
            let tag = scope_hash(scope);
            for (position, hash) in prefix_blocks(&state.tokens) {
                self.prefixes.entry((tag, position, hash)).or_default().push(id);
            }
        }
        self.entries.push_back(Entry {
            id, scope: scope.map(str::to_owned), owner: owner.map(str::to_owned),
            key,
            state,
            saved: now,
            charge,
            allocations,
            checkpoint,
            children: Arc::new(()),
        });
        if let Some(expiry) = checkpoint { self.track_checkpoint_expiry(expiry); }
        Ok(())
    }
    fn covered(&self, key: &SessionKey, scope: Option<&str>, tokens: &[u32], has_logits: bool) -> bool {
        self.entries
            .iter()
            .any(|e| e.checkpoint.is_none() && &e.key == key && e.scope.as_deref() == scope && e.state.tokens.starts_with(tokens)
                && (!has_logits || (e.state.tokens.len() == tokens.len() && !e.state.next_logits.is_empty())))
    }
    fn drop_prefixes(&mut self, key: &SessionKey, scope: Option<&str>, tokens: &[u32], has_logits: bool) {
        while let Some(i) = self
            .entries
            .iter()
            .position(|e| e.checkpoint.is_none() && &e.key == key && e.scope.as_deref() == scope && tokens.starts_with(&e.state.tokens)
                && (has_logits || e.state.next_logits.is_empty() || tokens.len() == e.state.tokens.len()))
        {
            self.remove(i);
        }
    }
}

impl<S: crate::Session> SessionCache<S> {
    /// Only called at a complete canonical prompt boundary, before decode KV.
    pub(crate) fn checkpoint(&mut self, scope: Option<&str>, owner: Option<&str>, state: &CachedSession<S>, now: Instant)
        -> Result<Checkpoint, &'static str> {
        if state.tokens.is_empty() || state.next_logits.is_empty() || !state.session.can_fork()
            || state.session.checkpoint_reusable_tokens(state.tokens.len()) > state.tokens.len()
            || state.reserved_session_bytes == 0
            || state.tokens.len() > self.limits.tokens_per_entry
            || state.next_logits.len() > self.limits.logits_per_entry
            || state.next_logits.iter().any(|v| !v.is_finite()) { return Err("unavailable"); }
        let key = SessionKey::generate().map_err(|_| "randomness_unavailable")?;
        let ttl = self.limits.idle_ttl;
        let expiry = now + ttl;
        let expires_at_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default()
            .saturating_add(ttl).as_millis().min(u64::MAX as u128) as u64;
        let allocations = state.session.retained_allocations();
        let reserved = state.session.retained_bytes().unwrap_or(state.reserved_session_bytes);
        let charge = if allocations.is_some() { state.session.retained_private_bytes() } else { reserved }
            .saturating_add(state.tokens.len().saturating_mul(4))
            .saturating_add(state.next_logits.len().saturating_mul(4))
            .saturating_add(checkpoint_metadata::<S>(scope, owner))
            .saturating_add(allocations.as_ref().map_or(0, |a| a.len() * 96));
        let allocations = allocations.unwrap_or_default();
        if !self.valid_allocations(&allocations) { return Err("unavailable"); }
        self.expire_checkpoints(now);
        if !self.checkpoint_room(owner, scope, &state.tokens, charge, &allocations) || !self.make_room(charge, &allocations) {
            return Err("capacity");
        }
        let session = state.session.fork().map_err(|_| "unavailable")?;
        self.fund_pages(session.funded_snapshot_allocations(None));
        let reusable_prefix_tokens = session.checkpoint_reusable_tokens(state.tokens.len());
        let reserved_session_bytes = session.retained_bytes().unwrap_or(reserved);
        self.insert_entry(key, scope, owner, CachedSession { checkpoint_lease: None, session, tokens: state.tokens.clone(),
            next_logits: state.next_logits.clone(), reserved_session_bytes }, now, Some(expiry))
            .map_err(|_| "capacity")?;
        let entry = self.entries.back().unwrap();
        Ok(Checkpoint { handle: key, prompt_tokens: entry.state.tokens.len(), reusable_prefix_tokens,
            expires_at_ms,
            retained_bytes: entry.charge + entry.allocations.iter().map(|(_, b)| b).sum::<usize>(),
            scope_retained_bytes: self.checkpoint_charge(owner, 0, &[]).1 })
    }
    /// The handle is a hint, never authority to skip prompt verification. Do not
    /// distinguish a foreign handle from an evicted/expired/old-load handle.
    fn checkpoint_index(&self, key: &SessionKey, scope: Option<&str>, now: Instant) -> Option<usize> {
        self.entries.iter().position(|e| e.checkpoint.is_some_and(|expiry| now < expiry)
            && e.scope.as_deref() == scope && e.key.matches(key))
    }
    /// Admission may credit only the pages the named, verified checkpoint will
    /// actually fork. This lookup confers no reservation; eviction can intervene.
    pub(crate) fn checkpoint_state(&self, key: &SessionKey, scope: Option<&str>, prompt: &[u32], now: Instant)
        -> Option<(&CachedSession<S>, usize)> {
        let entry = &self.entries[self.checkpoint_index(key, scope, now)?];
        let (retain, _) = entry.state.restorable_prefix(prompt, entry.state.tokens.len())?;
        Some((&entry.state, retain))
    }
    pub(crate) fn fork_checkpoint(&mut self, key: &SessionKey, scope: Option<&str>, prompt: &[u32], now: Instant)
        -> (Option<CachedSession<S>>, &'static str) {
        self.expire_checkpoints(now);
        let Some(index) = self.checkpoint_index(key, scope, now) else { return (None, "unavailable"); };
        if self.entries[index].state.restorable_prefix(prompt, self.entries[index].state.tokens.len()).is_none() {
            return (None, "prompt_mismatch");
        }
        match self.fork_entry(index, prompt, now) {
            Ok(Some(state)) => (Some(state), "hit"),
            _ => (None, "unavailable"),
        }
    }
    /// Proposal KV is expendable; target KV and its canonical logits are not.
    /// Use the same physical accounting for pressure and continuation handoff.
    fn release_draft(&mut self, index: usize) { self.release(index, S::disable_model_draft); }
    fn release_replay(&mut self, index: usize) { self.release(index, S::release_replay_snapshot); }
    fn release(&mut self, index: usize, release: fn(&mut S)) {
        let entry = &mut self.entries[index];
        let before = entry.state.session.retained_bytes();
        let private_before = entry.state.session.retained_private_bytes();
        release(&mut entry.state.session);
        // Backends without allocation lists use a conservative private charge.
        let Some(allocations) = entry.state.session.retained_allocations() else {
            let freed = before.zip(entry.state.session.retained_bytes()).map_or(0, |(a, b)| a.saturating_sub(b)).min(entry.charge);
            entry.charge -= freed;
            self.retained_bytes -= freed;
            return;
        };
        for &(id, _) in &entry.allocations {
            let allocation = self.allocations.get_mut(&id).unwrap();
            allocation.1 -= 1;
            if allocation.1 == 0 { self.retained_bytes -= allocation.0; self.allocations.remove(&id); }
        }
        let metadata = entry.allocations.len().saturating_sub(allocations.len()) * 96
            + private_before.saturating_sub(entry.state.session.retained_private_bytes());
        entry.charge -= metadata;
        self.retained_bytes -= metadata;
        entry.allocations = allocations;
        for &(id, bytes) in &entry.allocations {
            let allocation = self.allocations.entry(id).or_insert_with(|| { self.retained_bytes += bytes; (bytes, 0) });
            allocation.1 += 1;
        }
    }
    fn candidates(&self, scope: Option<&str>, prompt: &[u32], key: Option<&SessionKey>) -> Vec<(usize, usize)> {
        // The first-token bucket includes shorter valid checkpoints even when
        // a newer entry has a longer lexical match but cannot restore it.
        let Some((position, hash)) = prefix_blocks(prompt).next() else { return Vec::new(); };
        let Some(ids) = self.prefixes.get(&(scope_hash(scope), position, hash)) else { return Vec::new(); };
        let mut candidates: Vec<_> = self.entries.iter().enumerate()
            .filter(|(_, entry)| entry.checkpoint.is_none() && entry.scope.as_deref() == scope
                && key.is_none_or(|key| &entry.key == key) && ids.contains(&entry.id))
            .filter_map(|(index, entry)| entry.state.restorable_prefix(prompt, entry.state.tokens.len())
                .map(|(retain, restored)| (index, restored, retain)))
            .collect();
        candidates.sort_unstable_by_key(|&(index, restored, retain)| std::cmp::Reverse((restored, retain, index)));
        candidates.into_iter().map(|(index, restored, _)| (index, restored)).collect()
    }
    pub(crate) fn reusable_tokens(&self, scope: Option<&str>, prompt: &[u32]) -> usize {
        self.candidates(scope, prompt, None).first().map_or(0, |&(_, restored)| restored)
    }
    pub(crate) fn profile_match(&self, id: u64, scope: Option<&str>, prompt: &[u32]) {
        let scoped = self.entries.iter().filter(|e| e.scope.as_deref() == scope).count();
        if let Some(&(index, restored)) = self.candidates(scope, prompt, None).first() {
            let tokens = &self.entries[index].state.tokens;
            let shared = shared_prefix(tokens, prompt);
            let reason = if shared == prompt.len() { "prompt_end" }
                else if shared == tokens.len() { "cached_end" } else { "token_difference" };
            eprintln!("request_prefix_match id={id} prompt_tokens={} cached_tokens={} matched_tokens={shared} restorable_tokens={restored} reason={reason} scoped_entries={scoped} retained_bytes={}",
                prompt.len(), tokens.len(), self.retained_bytes);
        } else {
            let reason = if self.entries.is_empty() { "empty" }
                else if scoped == 0 { "scope_miss" } else { "no_restorable_prefix" };
            eprintln!("request_prefix_match id={id} prompt_tokens={} matched_tokens=0 reason={reason} scoped_entries={scoped} retained_bytes={}",
                prompt.len(), self.retained_bytes);
        }
    }
    pub(crate) fn reusable_allocations(&self, scope: Option<&str>, prompt: &[u32]) -> Vec<(usize, usize)> {
        self.candidates(scope, prompt, None).first().and_then(|&(i, restored)|
            self.entries[i].state.session.prefix_allocations(restored)).unwrap_or_default()
    }
    pub(crate) fn funded_candidate_allocations(&self, scope: Option<&str>, prompt: &[u32],
        key: Option<&SessionKey>, automatic: bool, checkpoint: Option<&SessionKey>,
        checkpoint_scope: Option<&str>, now: Instant) -> Vec<(usize, usize)> {
        if let Some((state, _)) = checkpoint.and_then(|key| self.checkpoint_state(key, checkpoint_scope, prompt, now)) {
            return state.session.funded_snapshot_allocations(Some(prompt.len()));
        }
        let candidate = if automatic { self.candidates(scope, prompt, None) }
            else if key.is_some() { self.candidates(scope, prompt, key) } else { Vec::new() };
        candidate.first().map_or_else(Vec::new, |&(i, _)| self.entries[i].state.session.funded_snapshot_allocations(Some(prompt.len())))
    }
    pub(crate) fn fork_automatic(&mut self, scope: Option<&str>, prompt: &[u32], now: Instant)
        -> Result<Option<CachedSession<S>>, String> {
        self.fork_candidates(scope, prompt, None, now)
    }
    fn fork_candidates(&mut self, scope: Option<&str>, prompt: &[u32], key: Option<&SessionKey>, now: Instant)
        -> Result<Option<CachedSession<S>>, String> {
        for (index, _) in self.candidates(scope, prompt, key) {
            if let Some(state) = self.fork_entry(index, prompt, now)? { return Ok(Some(state)); }
        }
        Ok(None)
    }
    /// Search only the supplied capability's history. Fork before truncating;
    /// another branch can still use the original snapshot unchanged.
    pub fn fork_prefix(
        &mut self,
        key: &SessionKey,
        prompt: &[u32],
        now: Instant,
    ) -> Result<Option<CachedSession<S>>, String> {
        self.fork_scoped(key, None, prompt, now)
    }
    pub(crate) fn fork_scoped(&mut self, key: &SessionKey, scope: Option<&str>, prompt: &[u32], now: Instant)
        -> Result<Option<CachedSession<S>>, String> {
        self.fork_candidates(scope, prompt, Some(key), now)
    }

    fn fork_entry(&mut self, index: usize, prompt: &[u32], now: Instant)
        -> Result<Option<CachedSession<S>>, String> {
        let mut state = match self.entries[index].state.fork_prefix(prompt) {
            Ok(Some(state)) => state,
            _ => return Ok(None),
        };
        if self.entries[index].checkpoint.is_some() {
            state.checkpoint_lease = Some(self.entries[index].children.clone());
        }
        // A continuation publishes a new prompt snapshot after prefill. Hand
        // it the proposal prefix instead of pinning every historical draft tail.
        // Exact replays skip prefill, so keep their original proposal snapshot.
        // All target tokens/logits survive either path for exact branch reuse.
        if self.entries[index].checkpoint.is_none() && state.tokens.len() < prompt.len() { self.release_draft(index); }
        // An automatic entry consumed to its end by a longer prompt is superseded
        // by the prompt snapshot this request publishes: keep its LRU age so it
        // yields first. Exact replays publish nothing new and stay refreshed.
        let entry = &self.entries[index];
        if entry.checkpoint.is_none() && state.tokens.len() == entry.state.tokens.len()
            && state.tokens.len() < prompt.len() { return Ok(Some(state)); }
        let mut entry = self.entries.remove(index).unwrap();
        entry.saved = now;
        self.entries.push_back(entry);
        Ok(Some(state))
    }
    pub fn snapshot(&mut self, key: SessionKey, state: &CachedSession<S>, now: Instant) {
        self.snapshot_scoped(key, None, state, now, false);
    }
    pub(crate) fn retain_draft_prefix(&mut self, key: SessionKey, scope: Option<&str>, source: &CachedSession<S>, now: Instant) {
        // A snapshot fork costs allocations, copies and a device sync on the
        // completion path, ahead of reading the next turn. Skip it when the
        // finished session has no proposal prefix to hand over.
        if !source.session.has_draft_prefix() { return; }
        let Some(index) = self.entries.iter().enumerate().filter(|(_, e)| e.checkpoint.is_none() && e.key == key
            && e.scope.as_deref() == scope && source.tokens.starts_with(&e.state.tokens))
            .max_by_key(|(_, e)| e.state.tokens.len()).map(|(i, _)| i) else { return; };
        let entry = &self.entries[index].state;
        let Ok(Some(mut state)) = entry.fork_prefix(&entry.tokens) else { return; };
        if state.session.retain_draft_prefix(&source.session) != Ok(true) { return; }
        // Only the draft changes. Keep the prompt's original target pages and
        // logits: generated target KV cannot substitute for cold BF16 prefill.
        self.fund_pages(state.session.prefix_allocations(state.tokens.len()).unwrap_or_default());
        self.fund_pages(state.session.funded_snapshot_allocations(None));
        self.remove(index);
        let _ = self.insert_scoped(key, scope, state, now);
    }
    pub(crate) fn snapshot_scoped(&mut self, key: SessionKey, scope: Option<&str>, state: &CachedSession<S>, now: Instant, prompt_end: bool) {
        // Reject before evicting another conversation or allocating a KV copy.
        if state.tokens.is_empty()
            || !state.session.can_fork()
            || state.session.checkpoint_reusable_tokens(state.tokens.len()) > state.tokens.len()
            || state.tokens.len() > self.limits.tokens_per_entry
            || state.next_logits.len() > self.limits.logits_per_entry
            || state.next_logits.iter().any(|v| !v.is_finite())
            || state.reserved_session_bytes == 0
        {
            return;
        }
        if self.covered(&key, scope, &state.tokens, !state.next_logits.is_empty()) {
            return;
        }
        let reserved = state
            .session
            .retained_bytes()
            .unwrap_or(state.reserved_session_bytes);
        let allocations = state.session.retained_allocations();
        if allocations.as_ref().is_some_and(|a| !self.valid_allocations(a)) { return; }
        let charge = if allocations.is_some() { state.session.retained_private_bytes() } else { reserved }
            .saturating_add(state.tokens.len().saturating_mul(4))
            .saturating_add(state.next_logits.len().saturating_mul(4))
            .saturating_add(index_charge(state.tokens.len(), scope))
            .saturating_add(allocations.as_ref().map_or(0, |a| a.len() * 96));
        if charge > self.limits.retained_bytes {
            return;
        }
        // A new prompt replaces older turns; its completion keeps this prompt's
        // logits so replay can reuse the whole prompt without another head pass.
        // The final prompt tile can reveal an already-funded in-place arena.
        // Credit it before make_room, not only after the snapshot has forked.
        self.fund_pages(state.session.funded_snapshot_allocations(None));
        self.drop_prefixes(&key, scope, &state.tokens, prompt_end && !state.next_logits.is_empty());
        if !self.make_room(charge, &allocations.unwrap_or_default()) { return; }
        let Ok(session) = state.session.fork() else {
            return;
        };
        self.fund_pages(session.funded_snapshot_allocations(None));
        let reserved = session.retained_bytes().unwrap_or(reserved);
        let _ = self.insert_scoped(
            key,
            scope,
            CachedSession {
                checkpoint_lease: None,
                session,
                tokens: state.tokens.clone(),
                next_logits: state.next_logits.clone(),
                reserved_session_bytes: reserved,
            },
            now,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};
    #[test]
    fn snapshot_credits_live_storage_revealed_at_prompt_end() {
        #[derive(Clone)] struct Funded(bool);
        impl crate::Session for Funded {
            fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
            fn fork(&self) -> Result<Self, String> { Ok(Self(false)) }
            fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> { Some(vec![(42, 4096)]) }
            fn retained_private_bytes(&self) -> usize { 32 }
            fn funded_snapshot_allocations(&self, _: Option<usize>) -> Vec<(usize, usize)> {
                if self.0 { vec![(42, 4096)] } else { Vec::new() }
            }
        }
        let now = Instant::now();
        let mut cache = SessionCache::new(CacheLimits { retained_bytes: 8192, ..limits() }).unwrap();
        cache.rebalance(2048, BTreeSet::new(), now);
        let state = CachedSession { checkpoint_lease: None, session: Funded(true), tokens: vec![1,2],
            next_logits: vec![1.,2.], reserved_session_bytes: 4096 };
        cache.snapshot_scoped(SessionKey::generate().unwrap(), None, &state, now, true);
        assert_eq!(cache.len(), 1);
        assert!(cache.retained_bytes() > 4096 && cache.unfunded_bytes() < 2048);
        cache.rebalance(8192, BTreeSet::new(), now);
        assert_eq!(cache.len(), 1);
    }
    #[test]
    fn checkpoints_share_charge_and_enforce_scope_release_lru_ttl() {
        #[derive(Clone)] struct Shared(Rc<[u8; 1024]>);
        impl crate::Session for Shared {
            fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
            fn fork(&self) -> Result<Self, String> { Ok(self.clone()) }
            fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> { Some(vec![(Rc::as_ptr(&self.0) as usize, 1024)]) }
        }
        let now = Instant::now();
        let mut c = SessionCache::new(CacheLimits { entries: 32, retained_bytes: 4096, ..limits() }).unwrap();
        let state = CachedSession { checkpoint_lease: None, session: Shared(Rc::new([0; 1024])), tokens: vec![1, 2], next_logits: vec![1., 2.], reserved_session_bytes: 1024 };
        let a = c.checkpoint(Some("a"), Some("a"), &state, now).unwrap();
        c.limits.retained_bytes = 2 * (3 * a.retained_bytes - 2 * 1024);
        let b = c.checkpoint(Some("a"), Some("a"), &state, now).unwrap();
        assert_eq!(b.scope_retained_bytes, 2 * a.retained_bytes - 1024);
        assert_eq!(c.fork_checkpoint(&a.handle, Some("b"), &[1, 2], now).1, "unavailable");
        assert_eq!(c.fork_checkpoint(&a.handle, Some("a"), &[3, 1], now).1, "prompt_mismatch");
        assert_eq!(c.fork_checkpoint(&a.handle, Some("a"), &[1, 2, 3], now).1, "hit");
        for _ in 0..2 { c.checkpoint(Some("a"), Some("a"), &state, now).unwrap(); }
        assert_eq!(c.fork_checkpoint(&b.handle, Some("a"), &[1, 2], now).1, "unavailable");
        c.release_checkpoint(&a.handle, Some("b"));
        assert_eq!(c.fork_checkpoint(&a.handle, Some("a"), &[1, 2], now).1, "hit");
        c.release_checkpoint(&a.handle, Some("a"));
        assert_eq!(c.fork_checkpoint(&a.handle, Some("a"), &[1, 2], now).1, "unavailable");
        c.limits.retained_bytes = 1 << 20; c.available_bytes = 1 << 20;
        for _ in 0..17 { c.checkpoint(Some("a"), Some("a"), &state, now).unwrap(); }
        assert_eq!(c.len(), 16);
        c.expire_checkpoints(now + Duration::from_secs(5));
        assert_eq!((c.len(), c.retained_bytes()), (0, 0));
        for entries in [2, 16] { let forks = Rc::new(Cell::new(0)); let mut c = SessionCache::new(CacheLimits { entries, retained_bytes: 8192, ..limits() }).unwrap();
            let a = c.checkpoint(None, None, &kv(&[1, 2], &forks), now).unwrap(); let b = c.checkpoint(None, None, &kv(&[3, 4], &forks), now).unwrap(); c.limits.retained_bytes = 2 * (a.retained_bytes + b.retained_bytes + 128);
            let next = c.checkpoint(None, None, &kv(&[3, 4, 5], &forks), now).unwrap(); assert_eq!((c.fork_checkpoint(&a.handle, None, &[1, 2], now).1, c.fork_checkpoint(&b.handle, None, &[3, 4], now).1, c.fork_checkpoint(&next.handle, None, &[3, 4, 5], now).1, c.len()), ("hit", "unavailable", "hit", 2)); }
    }
    #[test]
    fn exact_replay_falls_back_to_the_recurrent_checkpoint() {
        // A backend that forks only at its checkpoint (Flash-Next: the last 64-aligned prompt row).
        struct Aligned(usize);
        impl crate::Session for Aligned {
            fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
            fn can_fork_prefix(&self, position: usize) -> bool { position == self.0 }
            fn checkpoint_reusable_tokens(&self, tokens: usize) -> usize { if self.0 <= tokens { self.0 } else { 0 } }
            fn fork(&self) -> Result<Self, String> { Ok(Aligned(self.0)) }
            fn fork_prefix(&self, _: usize) -> Result<Self, String> { Ok(Aligned(self.0)) }
        }
        let entry = |checkpoint| CachedSession { checkpoint_lease: None, session: Aligned(checkpoint), tokens: vec![1, 2, 3],
            next_logits: vec![1., 2.], reserved_session_bytes: 1 };
        // Exact replay of [1, 2, 3] with a checkpoint at 2: restore 2 rows, no stale logits.
        let fork = entry(2).fork_prefix(&[1, 2, 3]).unwrap().unwrap();
        assert_eq!((fork.tokens, fork.next_logits.len()), (vec![1, 2], 0));
        // A checkpoint at the prompt end keeps the whole exact replay with its logits.
        let fork = entry(3).fork_prefix(&[1, 2, 3]).unwrap().unwrap();
        assert_eq!((fork.tokens, fork.next_logits.len()), (vec![1, 2, 3], 2));
        // Appends behave as before: the checkpoint at 2 restores for a longer prompt.
        assert_eq!(entry(2).fork_prefix(&[1, 2, 3, 4]).unwrap().unwrap().tokens, vec![1, 2]);
    }
    #[test]
    fn live_children_pin_superseded_checkpoint_until_drop() {
        let (now, forks) = (Instant::now(), Rc::new(Cell::new(0)));
        let mut c = SessionCache::new(CacheLimits { entries: 2, retained_bytes: 1 << 20, ..limits() }).unwrap();
        let base = c.checkpoint(None, None, &kv(&[1, 2], &forks), now).unwrap();
        let child = c.fork_checkpoint(&base.handle, None, &[1, 2, 3], now).0.unwrap();
        c.checkpoint(None, None, &kv(&[1, 2, 3], &forks), now).unwrap();
        c.checkpoint(None, None, &kv(&[4, 5], &forks), now).unwrap();
        let other = c.fork_checkpoint(&base.handle, None, &[1, 2, 6], now).0.unwrap();
        c.rebalance(0, BTreeSet::new(), now);
        assert_eq!(c.len(), 1);
        assert!(!c.make_room(1, &[]));
        drop(child); c.rebalance(0, BTreeSet::new(), now); assert_eq!(c.len(), 1);
        drop(other);
        c.rebalance(0, BTreeSet::new(), now);
        assert!(c.is_empty());
    }
    struct DropCount(Rc<Cell<usize>>);
    impl crate::Session for DropCount {
        fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
    }
    impl Drop for DropCount {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    fn key(n: u8) -> SessionKey {
        SessionKey([n; 32])
    }
    fn limits() -> CacheLimits {
        CacheLimits {
            entries: 2,
            retained_bytes: 1024,
            tokens_per_entry: 16,
            logits_per_entry: 16,
            idle_ttl: Duration::from_secs(5),
        }
    }
    fn state(d: &Rc<Cell<usize>>, tokens: Vec<u32>) -> CachedSession<DropCount> {
        CachedSession {
            checkpoint_lease: None,
            session: DropCount(d.clone()),
            tokens,
            next_logits: vec![1., 2.],
            reserved_session_bytes: 64,
        }
    }
    #[test]
    fn capability_is_strict_and_debug_redacts() {
        let k = key(0xab);
        assert_eq!(SessionKey::parse(&k.expose()).unwrap(), k);
        for s in [
            "a".repeat(63),
            "g".repeat(64),
            "A".repeat(64),
            "é".repeat(32),
        ] {
            assert!(SessionKey::parse(&s).is_err());
        }
        assert_eq!(format!("{k:?}"), "SessionKey(<redacted>)");
    }
    #[cfg(unix)]
    #[test]
    fn capabilities_use_os_randomness() {
        assert_ne!(
            SessionKey::generate().unwrap(),
            SessionKey::generate().unwrap()
        );
    }
    #[test]
    fn idle_entries_survive_until_memory_pressure() {
        let d = Rc::new(Cell::new(0));
        let now = Instant::now();
        let mut c = SessionCache::new(limits()).unwrap();
        c.insert_snapshot(key(1), state(&d, vec![1]), now).unwrap();
        c.rebalance(1024, BTreeSet::new(), now + Duration::from_secs(5));
        assert_eq!(c.len(), 1);
        c.rebalance(0, BTreeSet::new(), now + Duration::from_secs(5));
        assert!(c.is_empty());
        assert_eq!(d.get(), 1);
    }
    #[test]
    fn budget_counts_capacity_and_eviction() {
        let d = Rc::new(Cell::new(0));
        let now = Instant::now();
        let mut l = limits();
        l.retained_bytes = 1024;
        let mut c = SessionCache::new(l).unwrap();
        c.insert_snapshot(key(1), state(&d, vec![1]), now).unwrap();
        c.insert_snapshot(key(2), state(&d, vec![2]), now).unwrap();
        c.rebalance(350, BTreeSet::new(), now);
        assert_eq!(c.len(), 1);
        let mut s = state(&d, vec![3]);
        s.tokens.reserve(100);
        assert!(c.insert_snapshot(key(3), s, now).is_err());
        assert_eq!(c.len(), 1);
        assert_eq!(d.get(), 2);
    }
    #[test]
    fn malformed_replacement_preserves_cache_and_overflow_drops() {
        let d = Rc::new(Cell::new(0));
        let now = Instant::now();
        let mut c = SessionCache::new(limits()).unwrap();
        c.insert_snapshot(key(1), state(&d, vec![1]), now).unwrap();
        let mut bad = state(&d, vec![1]);
        bad.next_logits[0] = f32::NAN;
        assert!(c.insert_snapshot(key(1), bad, now).is_err());
        assert_eq!(c.len(), 1);
        assert_eq!(d.get(), 1);
        let mut bad = state(&d, vec![1]);
        bad.reserved_session_bytes = usize::MAX;
        assert!(c.insert_snapshot(key(1), bad, now).is_err());
        assert_eq!(d.get(), 2);
    }
    struct Kv {
        data: Vec<u32>,
        forks: Rc<Cell<usize>>,
        fail_truncate: bool,
    }
    impl crate::Session for Kv {
        fn step(&mut self, _: u32) -> Result<Vec<f32>, String> {
            unreachable!()
        }
        fn fork(&self) -> Result<Self, String> {
            self.forks.set(self.forks.get() + 1);
            Ok(Self {
                data: self.data.clone(),
                forks: self.forks.clone(),
                fail_truncate: self.fail_truncate,
            })
        }
        fn truncate(&mut self, n: usize) -> Result<(), String> {
            if self.fail_truncate {
                return Err("truncate failed".into());
            }
            self.data.truncate(n);
            Ok(())
        }
        fn retained_bytes(&self) -> Option<usize> {
            Some(self.data.capacity() * 4)
        }
    }
    fn kv(tokens: &[u32], forks: &Rc<Cell<usize>>) -> CachedSession<Kv> {
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(tokens);
        CachedSession {
            checkpoint_lease: None,
            session: Kv {
                data,
                forks: forks.clone(),
                fail_truncate: false,
            },
            tokens: tokens.to_vec(),
            next_logits: vec![1., 2.],
            reserved_session_bytes: 128,
        }
    }
    #[test]
    fn superseded_automatic_entry_yields_first() {
        let (now, forks) = (Instant::now(), Rc::new(Cell::new(0)));
        let mut c = SessionCache::new(limits()).unwrap();
        let h = c.checkpoint(None, None, &kv(&[1, 2, 3], &forks), now).unwrap();
        assert_eq!(c.checkpoint_state(&h.handle, None, &[1, 2, 4], now).unwrap().1, 2);
        let (s, why) = c.fork_checkpoint(&h.handle, None, &[1, 2, 4], now);
        let s = s.unwrap(); assert_eq!((why, s.tokens, s.session.data), ("hit", vec![1, 2], vec![1, 2]));
        assert_eq!(c.fork_checkpoint(&h.handle, None, &[1, 2, 3], now).0.unwrap().tokens, [1, 2, 3]);
        c.release_checkpoint(&h.handle, None);
        c.insert_snapshot(key(1), kv(&[1, 2], &forks), now).unwrap();
        c.insert_snapshot(key(2), kv(&[5, 6], &forks), now).unwrap();
        assert!(c.fork_automatic(None, &[1, 2], now).unwrap().is_some());
        assert!(c.fork_automatic(None, &[5, 6, 7], now).unwrap().is_some());
        c.insert_snapshot(key(3), kv(&[5, 6, 7], &forks), now).unwrap();
        assert_eq!((c.reusable_tokens(None, &[1, 2, 3]), c.entries[0].key), (2, key(1)));
    }
    #[test]
    fn completion_without_draft_prefix_never_forks() {
        let (now, forks) = (Instant::now(), Rc::new(Cell::new(0)));
        let mut c = SessionCache::new(limits()).unwrap();
        c.insert_snapshot(key(1), kv(&[1, 2], &forks), now).unwrap();
        c.retain_draft_prefix(key(1), None, &kv(&[1, 2, 3], &forks), now);
        assert_eq!((forks.get(), c.len()), (0, 1));
    }
    #[test]
    fn replay_snapshot_yields_before_any_entry() {
        struct Pages(Vec<(usize, usize)>);
        impl crate::Session for Pages {
            fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
            fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> { Some(self.0.clone()) }
            fn release_replay_snapshot(&mut self) { if self.0.len() == 3 { self.0.remove(1); } }
        }
        let pages = |tokens: Vec<u32>, a: Vec<(usize, usize)>| CachedSession { checkpoint_lease: None, session: Pages(a), tokens, next_logits: vec![1., 2.], reserved_session_bytes: 1 };
        let now = Instant::now();
        let mut c = SessionCache::new(CacheLimits { entries: 4, retained_bytes: 1900, ..limits() }).unwrap();
        c.insert_snapshot(key(1), pages(vec![1], vec![(1, 200), (2, 200), (3, 200)]), now).unwrap();
        c.insert_snapshot(key(2), pages(vec![2], vec![(4, 200)]), now).unwrap();
        c.insert_snapshot(key(3), pages(vec![3], vec![(5, 200)]), now).unwrap();
        assert_eq!((c.len(), c.entries[0].allocations.clone(), c.retained_bytes()), (3, vec![(1, 200), (3, 200)], 3 * 468 + 296));
        for a in [vec![(0,200)],vec![(1,300)],vec![(6,200),(6,200)]] { let before=c.retained_bytes(); assert!(c.insert_snapshot(key(4),pages(vec![4],a),now).is_err()); assert_eq!((c.len(),c.retained_bytes()),(3,before)); }
    }
    #[test]
    fn rejected_snapshot_never_evicts_or_forks() {
        struct Pending; impl crate::Session for Pending { fn step(&mut self,_:u32)->Result<Vec<f32>,String>{unreachable!()} fn can_fork(&self)->bool{false} }
        let mut pending=SessionCache::new(limits()).unwrap(); assert!(pending.insert_snapshot(key(1),CachedSession{checkpoint_lease:None,session:Pending,tokens:vec![1],next_logits:vec![0.],reserved_session_bytes:1},Instant::now()).is_err()); assert_eq!((pending.len(),pending.retained_bytes()),(0,0));
        let now = Instant::now();
        let forks = Rc::new(Cell::new(0));
        let mut l = limits();
        l.entries = 1;
        let mut c = SessionCache::new(l).unwrap();
        c.snapshot(key(1), &kv(&[1, 2], &forks), now);
        let charge = c.retained_bytes();
        for kind in 0..4 {
            let mut invalid = kv(&[3, 4], &forks);
            match kind {
                0 => invalid.tokens.resize(17, 3),
                1 => invalid.next_logits.resize(17, 0.),
                2 => invalid.next_logits[0] = f32::NAN,
                _ => invalid.session.data.reserve(2048),
            }
            c.snapshot(key(2), &invalid, now);
            assert_eq!(c.retained_bytes(), charge);
            assert_eq!(c.entries[0].key, key(1));
            assert_eq!(forks.get(), 1);
        }
    }
}
