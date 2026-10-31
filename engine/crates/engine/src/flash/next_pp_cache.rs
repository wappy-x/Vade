//! Ordered prefix ownership on every pipeline stage. Tickets outlive session slots;
//! their last front-side owner queues a release, drained before cache work or allocation.
use super::*;

pub(in super::super) const CAPTURE: u32 = 9;
pub(in super::super) const PUBLISH: u32 = 10;
pub(in super::super) const RESTORE: u32 = 11;
pub(in super::super) const RELEASE: u32 = 12;
const ROOM: u32 = 13;
pub(in super::super) fn command(kind: u32) -> bool { matches!(kind, CAPTURE | PUBLISH | RESTORE | RELEASE | ROOM) }
pub(in super::super) fn charge(rows: &[f32]) -> usize {
    (u64::from(rows[0].to_bits()) | u64::from(rows[1].to_bits()) << 32) as usize
}

pub(in super::super) fn set_charge(rows: &mut [f32], bytes: usize) {
    rows[0] = f32::from_bits(bytes as u32);
    rows[1] = f32::from_bits((bytes as u64 >> 32) as u32);
}
#[derive(Default)]
pub(in super::super) struct Tickets {
    next: u32, owners: Vec<(u32, usize, Arc<AtomicBool>)>, room: Option<(Instant, i128)>,
}
pub(in super::super) struct Lease {
    id: u32, meta: [u32; 4], pub(in super::super) bytes: usize, released: Arc<AtomicBool>,
}
impl Drop for Lease {
    fn drop(&mut self) { self.released.store(true, Ordering::Release); }
}
impl Link {
    fn cache_command(&mut self, kind: u32, meta: [u32; 4]) -> Result<usize> {
        self.cache_command_live(kind, meta, 0)
    }
    fn cache_command_live(&mut self, kind: u32, meta: [u32; 4], live: u32) -> Result<usize> {
        if self.poisoned { return Err("pipeline prefix link failed earlier".into()); }
        let mut rows = vec![0.0; W];
        rows[6] = f32::from_bits(live);
        // No pass runs: the request carries no tile or token-split bit of the last pass.
        let bits = (std::mem::take(&mut self.tp_tile), std::mem::take(&mut self.tp_seq));
        let exchanged = self.exchange(kind, &[meta], &mut rows);
        (self.tp_tile, self.tp_seq) = bits;
        exchanged.map_err(|e| self.fail(e))?;
        let mut room = [0u8; 16];
        for (part, word) in room.chunks_exact_mut(4).zip(&rows[2..6]) { part.copy_from_slice(&word.to_bits().to_le_bytes()); }
        self.cache.room = Some((Instant::now(), i128::from_le_bytes(room)));
        Ok(charge(&rows))
    }
    pub(super) fn cache_releases(&mut self) -> Result<()> {
        let released: Vec<u32> = self.cache.owners.iter().filter(|(_, _, done)| done.load(Ordering::Acquire)).map(|&(id, _, _)| id).collect();
        for id in released {
            self.cache_command(RELEASE, [0, 0, 0, id])?;
            self.cache.owners.retain(|&(held, _, _)| held != id);
        }
        Ok(())
    }
    /// The worker's room for prefix copies, sampled at most once per second while a prefix
    /// exists. A probe is an ordered exchange: it would wait for every queued prompt tile, so
    /// while a tile is queued the last sample stays (None before the first). The worker still
    /// refuses a copy that does not fit its own room.
    fn cache_room(&mut self) -> Result<Option<i128>> {
        if self.cache.owners.is_empty() { return Ok(None); }
        if self.pending == 0 && self.cache.room.is_none_or(|(at, _)| at.elapsed() >= Duration::from_secs(1)) {
            self.cache_command(ROOM, [0, 0, 0, 1])?;
        }
        Ok(self.cache.room.map(|(_, room)| room))
    }
    /// Releases of dropped prefixes. A release is an ordered exchange, so unless `drain` it
    /// waits while a prompt tile is queued. With `drain` (no request is active or queued), it
    /// first reads any reply that a canceled request's tile left unread.
    fn cache_releases_when(&mut self, drain: bool) -> Result<()> {
        if drain || self.pending == 0 { self.cache_releases()?; }
        Ok(())
    }
    fn cache_capture(&mut self, mut meta: [u32; 4], live: u32) -> Result<Option<Lease>> {
        self.cache_releases()?;
        self.cache.next = self.cache.next.checked_add(1).ok_or("pipeline prefix ticket overflow")?;
        meta[3] = self.cache.next;
        let bytes = self.cache_command_live(CAPTURE, meta, live)?;
        if bytes == 0 {
            self.cache_command(RELEASE, [0, 0, 0, meta[3]])?;
            return Ok(None);
        }
        let released = Arc::new(AtomicBool::new(false));
        self.cache.owners.push((meta[3], bytes, released.clone()));
        Ok(Some(Lease { id: meta[3], meta, bytes, released }))
    }
}

// Pending tickets hold only recurrent checkpoints; publication copies the source
// session's still-intact attention rows before that slot can be reused.
#[cfg(target_os = "linux")]
pub(in super::super) enum Saved<'a> {
    Pending { slot: u32, checkpoint: prefix::Checkpoint<'a>, bytes: usize },
    Published(NextPrefix<'a>),
}
#[cfg(target_os = "linux")]
impl Saved<'_> {
    fn bytes(&self) -> usize { match self { Self::Pending { bytes, .. } => *bytes, Self::Published(p) => p.device_bytes() } }
}
#[cfg(target_os = "linux")]
impl<'a> NextModel<'a> {
    pub fn tp_prefix_supported(&self) -> bool { self.pp.borrow().as_ref().is_some_and(|l| l.tp) && self.tp_pair() }
    pub fn pp_cache_releases(&self) -> Result<()> {
        if let Some(link) = self.pp.borrow_mut().as_mut() { link.cache_releases()?; }
        Ok(())
    }
    /// Worker room is sampled only while a prefix exists and no prompt tile is queued.
    /// Ordinary uncached turns send no probe.
    pub fn pp_cache_room(&self) -> Result<Option<i128>> {
        self.pp.borrow_mut().as_mut().map_or(Ok(None), Link::cache_room)
    }
    /// Releases of dropped prefixes during memory maintenance (Link::cache_releases_when;
    /// `idle`: no request is active or queued). Cache commands and new sessions send the rest first.
    pub fn pp_cache_maintain(&self, idle: bool) -> Result<()> {
        self.pp.borrow_mut().as_mut().map_or(Ok(()), |link| link.cache_releases_when(idle))
    }
    pub(in super::super) fn pp_cache_capture(&self, slot: u32, capacity: usize, position: usize) -> Result<Option<Lease>> {
        let live = self.cache_live_slots();
        self.pp.borrow_mut().as_mut().map_or(Ok(None), |l| l.cache_capture([slot, capacity as u32, position as u32, 0], live))
    }
    fn cache_live_slots(&self) -> u32 {
        self.pool.borrow().used.iter().enumerate().skip(1).filter(|&(_, used)| *used).fold(0, |mask, (slot, _)| mask | 1 << (slot - 1))
    }
    pub(in super::super) fn pp_cache_publish(&self, lease: &Lease) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().ok_or("prefix has no pipeline link")?;
        link.cache_releases()?;
        if link.cache_command(PUBLISH, lease.meta)? == 0 { return Err("pipeline stage could not publish a prefix".into()); }
        Ok(())
    }
    pub(in super::super) fn pp_cache_restore(&self, lease: &Lease, slot: u32, capacity: usize, position: usize) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().ok_or("prefix has no pipeline link")?;
        link.cache_releases()?;
        link.cache_command_live(RESTORE, [slot, capacity as u32, position as u32, lease.id], self.cache_live_slots())?;
        self.tp_prefix_restored(slot);
        Ok(())
    }
    pub(in super::super) fn pp_cache_step<'w>(&'w self, b: &Backend<'a>, sessions: &mut BTreeMap<u32, NextSession<'a, 'w>>,
        prefixes: &mut BTreeMap<u32, Saved<'a>>, (budget, room): (usize, usize), kind: u32, count: usize, meta: &[[u32; 4]], rows: &mut [f32], tile: usize, context: usize) -> Result<()> {
        if count != 1 || meta.len() != 1 || meta[0][3] == 0 { return Err("invalid pipeline prefix command".into()); }
        let [slot, capacity, position, id] = meta[0];
        let (capacity, position) = (capacity as usize, position as usize);
        if kind == RELEASE { prefixes.remove(&id); return cache_reply(b, rows, 1); }
        if kind == ROOM { return cache_reply(b, rows, 1); }
        if slot == 0 || slot as usize > POOL_SIZE || capacity > context || position == 0
            || !position.is_multiple_of(64) || position > capacity || kind == RESTORE && position == capacity {
            return Err("invalid pipeline prefix position or slot".into());
        }
        if matches!(kind, CAPTURE | RESTORE) {
            let live = rows[6].to_bits();
            if !valid_live_mask(live, slot) { return Err("invalid pipeline prefix live slots".into()); }
            b.synchronize()?;
            let before = sessions.len();
            // A pending checkpoint still refers to its source arena. Published copies do not.
            sessions.retain(|held, _| live & (1 << (*held - 1)) != 0
                || prefixes.values().any(|p| matches!(p, Saved::Pending { slot: source, .. } if source == held)));
            if sessions.len() != before {
                engine_model::events::note("prefix_cache_worker_reclaim", format_args!("released_sessions={} retained_sessions={}", before - sessions.len(), sessions.len()));
            }
        }
        let bytes = match kind {
            CAPTURE => {
                if prefixes.len() >= 2 * POOL_SIZE { return cache_reply(b, rows, 0); }
                if prefixes.contains_key(&id) { return Err("pipeline prefix ticket already exists".into()); }
                let s = sessions.get_mut(&slot).ok_or("pipeline prefix source missing")?;
                if s.retained > 0 && s.valid { s.commit_back(b, position)?; }
                if !s.valid || s.pos != position || s.capacity != capacity { return Err("pipeline prefix source position differs".into()); }
                let checkpoint_bytes = s.checkpoint_allocation_bytes();
                // Against the whole room: the front evicts older prefixes for this one only when it
                // publishes (pp_cache_publish sends their releases first), where the rest is checked.
                if !cache_room_for(b, checkpoint_bytes, checkpoint_bytes, room as i128)? { return cache_reply(b, rows, 0); }
                s.set_prefix_checkpoint(true);
                let captured = s.capture_checkpoint(b).map(|()| s.prefix_bytes());
                let checkpoint = s.checkpoint.borrow_mut().take();
                s.set_prefix_checkpoint(false);
                let bytes = captured?;
                match checkpoint {
                    // Older prefixes count only at PUBLISH, after the front's releases for this one.
                    Some(checkpoint) if checkpoint_bytes <= budget => {
                        prefixes.insert(id, Saved::Pending { slot, checkpoint, bytes: checkpoint_bytes }); bytes
                    }
                    Some(_) => 0,
                    None => 0,
                }
            }
            PUBLISH => {
                match prefixes.remove(&id) {
                    Some(Saved::Pending { slot: source, checkpoint, .. }) if source == slot => {
                        let s = sessions.get(&slot).ok_or("pipeline prefix source vanished")?;
                        *s.checkpoint.borrow_mut() = Some(checkpoint);
                        let bytes = s.prefix_bytes();
                        if !s.valid || s.pos < position || s.capacity != capacity
                            || prefixes.values().try_fold(bytes, |n, p| n.checked_add(p.bytes())).is_none_or(|n| n > budget)
                            || !cache_room_for(b, bytes, s.checkpoint_allocation_bytes(), owned_room(room, prefixes))? {
                            s.checkpoint.borrow_mut().take();
                            return cache_reply(b, rows, 0);
                        }
                        match s.save_prefix(b) {
                            Ok(prefix) => { prefixes.insert(id, Saved::Published(prefix)); 1 }
                            Err(_) => { s.checkpoint.borrow_mut().take(); 0 }
                        }
                    }
                    Some(Saved::Published(prefix)) => { prefixes.insert(id, Saved::Published(prefix)); 1 }
                    _ => 0,
                }
            }
            RESTORE => {
                let prefix = match prefixes.get(&id) {
                    Some(Saved::Published(prefix)) if prefix.position() == position => prefix,
                    _ => return Err("pipeline prefix ticket missing or position differs".into()),
                };
                sessions.remove(&slot);
                sessions.insert(slot, self.session_from_prefix(b, capacity, tile, prefix)?);
                1
            }
            _ => return Err("unknown pipeline prefix command".into()),
        };
        b.synchronize()?;
        engine_model::events::note("prefix_cache_worker", format_args!("command={kind} ticket={id} slot={slot} position={position} bytes={bytes} entries={}", prefixes.len()));
        cache_reply(b, rows, bytes)
    }
}

#[cfg(target_os = "linux")]
fn cache_room(b: &Backend<'_>) -> Result<i128> {
    b.refresh_kv_pressure()?;
    let allocator = b.allocation_budget() as i128 - b.accounted_bytes() as i128;
    Ok(b.kv_reserve_room()?.map_or(allocator, |(room, _)| room.min(allocator)))
}
/// What is left of the startup room (prefix_switch::cache_room) after the prefixes held.
#[cfg(target_os = "linux")]
fn owned_room(room: usize, prefixes: &BTreeMap<u32, Saved<'_>>) -> i128 {
    room as i128 - prefixes.values().map(Saved::bytes).sum::<usize>() as i128
}
/// Host pressure bounds retention, except inside the startup room (`owned`: what is left of
/// it): the plan funds those bytes. The allocator bounds both, with the slack.
#[cfg(target_os = "linux")]
fn cache_room_for(b: &Backend<'_>, bytes: usize, slack: usize, owned: i128) -> Result<bool> {
    b.prepare_cache_allocation(bytes.saturating_add(slack))?;
    let room = cache_room(b)?;
    let allocator = b.allocation_budget() as i128 - b.accounted_bytes() as i128;
    let fits = bytes as i128 <= room.saturating_sub(slack as i128)
        || bytes as i128 <= owned && bytes.saturating_add(slack) as i128 <= allocator;
    if !fits { engine_model::events::note("prefix_cache_worker_refused", format_args!("reason=no_room bytes={bytes} slack_bytes={slack} room_bytes={room} owned_room_bytes={owned}")); }
    Ok(fits)
}
#[cfg(target_os = "linux")]
fn cache_reply(b: &Backend<'_>, rows: &mut [f32], bytes: usize) -> Result<()> {
    set_charge(rows, bytes);
    for (word, part) in rows[2..6].iter_mut().zip(cache_room(b)?.to_le_bytes().chunks_exact(4)) {
        *word = f32::from_bits(u32::from_le_bytes(part.try_into().unwrap()));
    }
    Ok(())
}
fn valid_live_mask(mask: u32, slot: u32) -> bool {
    slot != 0 && slot as usize <= POOL_SIZE && mask >> POOL_SIZE == 0 && mask & (1 << (slot - 1)) != 0
}

#[cfg(target_os = "linux")]
impl<'a> NextModel<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn pp_back_step<'w>(
        &'w self,
        b: &Backend<'a>,
        sessions: &mut BTreeMap<u32, NextSession<'a, 'w>>,
        prefixes: &mut BTreeMap<u32, Saved<'a>>,
        prefix_budget: (usize, usize),
        kind: u32,
        count: usize,
        meta: &[[u32; 4]],
        tile: usize,
        context: usize,
        residual: &mut [f32],
    ) -> Result<()> {
        if command(kind) { return self.pp_cache_step(b, sessions, prefixes, prefix_budget, kind, count, meta, residual, tile, context); }
        for &[slot, word, pos, len] in meta {
            let (capacity, kv_rows, pos) = ((word & ((1 << 19) - 1)) as usize, (word >> 19) as usize * 64, pos as usize);
            if len == 0 || slot == 0 {
                return Err("pipeline row without a slot".into());
            }
            if capacity == 0 || capacity > context {
                return Err(format!("pipeline session capacity {capacity} outside 1..={context}"));
            }
            if sessions.get(&slot).is_none_or(|s| s.capacity != capacity) {
                sessions.remove(&slot);
                crate::flash::qsa::set_paged_rows(kv_rows);
                let session = self.session_with_chunk(b, capacity, tile);
                crate::flash::qsa::set_paged_rows(0);
                sessions.insert(slot, session?);
            }
            let s = sessions.get_mut(&slot).unwrap();
            if pos == 0 && (s.pos != 0 || !s.valid) {
                s.reset(b)?;
            }
            // After a verify, the chat's next request carries the position the
            // front committed; the back commits the same accepted prefix first.
            if s.retained > 0 && s.valid {
                s.commit_back(b, pos)?;
            }
            if s.pos != pos || !s.valid {
                return Err(format!("pipeline session slot {slot}: back position {} valid {} vs front {pos}", s.pos, s.valid));
            }
        }
        let mut by_slot: Vec<(u32, &mut NextSession<'a, 'w>)> = sessions.iter_mut().map(|(k, v)| (*k, v)).collect();
        let mut ordered: Vec<&mut NextSession<'a, 'w>> = Vec::with_capacity(meta.len());
        for m in meta {
            let i = by_slot.iter().position(|(k, _)| *k == m[0]).ok_or("pipeline cohort repeats a slot")?;
            ordered.push(by_slot.swap_remove(i).1);
        }
        let lens: Vec<usize> = meta.iter().map(|m| m[3] as usize).collect();
        match kind {
            KIND_ROWS | KIND_ROWS_ASYNC | KIND_ROWS_KEEP | KIND_ROWS_FINAL | KIND_VERIFY | crate::flash_tp_switch::KIND_VERIFY => {
                if lens.iter().sum::<usize>() != count {
                    return Err("pipeline rows do not match the residual".into());
                }
                for (s, &n) in ordered.iter_mut().zip(&lens) { s.fit_kv(b, s.pos + n)?; }
                NextSession::pp_back_rows(b, &mut ordered, &lens, residual, matches!(kind, KIND_VERIFY | crate::flash_tp_switch::KIND_VERIFY))
            }
            KIND_POOLED => {
                if lens.iter().any(|&n| n != 1) {
                    return Err("pipeline pooled step needs one row per chat".into());
                }
                for s in ordered.iter_mut() { s.fit_kv(b, s.pos + 1)?; }
                NextSession::pp_back_pooled(b, &mut ordered, count, residual)
            }
            crate::flash_tp_switch::KIND_HANDOFF => if self.tp_split_back(b, &mut ordered, &lens, residual)? { Ok(()) } else { self.tp_handoff(b, &mut ordered, &lens) }, _ => Err("pipeline request kind unknown".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::BTreeMap, net::TcpListener};

    /// Runs the production prompt/control wire twice with a CPU recurrence in place of
    /// CUDA. The source slot is overwritten between turns; a ticket restores a new slot.
    /// The room probe never waits on a queued prompt tile.
    /// Device recurrence, drafting and token exactness still need the Spark gate.
    #[test]
    fn paired_prefix_survives_slot_reuse_and_optional_refusal() {
        static STOP: AtomicBool = AtomicBool::new(false);
        static ROOMS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        assert!(valid_live_mask(1, 1) && valid_live_mask((1 << POOL_SIZE) - 1, POOL_SIZE as u32));
        for (mask, slot) in [(0, 1), (2, 1), (1, 0), (1, POOL_SIZE as u32 + 1), (u32::MAX, 1)] { assert!(!valid_live_mask(mask, slot)); }
        for (tp, ranks) in [(true, 2)] {
            let (mut outgoing, mut incoming) = (Vec::new(), Vec::new());
            for _ in 1..ranks {
                let listener = TcpListener::bind("127.0.0.1:0").unwrap();
                outgoing.push(Some(TcpStream::connect(listener.local_addr().unwrap()).unwrap()));
                incoming.push(Some(listener.accept().unwrap().0));
            }
            let front = outgoing[0].take().unwrap();
            let recipe = Recipe { limit: Duration::from_secs(2), asynchronous: true, mtp: true, tp };
            let mut workers = Vec::new();
            for rank in 1..ranks {
                let back = incoming[rank - 1].take().unwrap();
                let next = outgoing.get_mut(rank).and_then(Option::take).map(|s| (s, None));
                workers.push(std::thread::spawn(move || {
                    let mut live = BTreeMap::<u32, (usize, u32)>::new();
                    let mut saved = BTreeMap::<u32, (usize, u32)>::new();
                    serve_stage(back, None, next, Stage::new(rank, ranks, None).unwrap(), 512, 8, 64, &STOP, recipe,
                        |_| Ok(()), || Ok(()), |kind, _, meta, rows| {
                            let [slot, _, pos, id] = meta[0];
                            let bits = crate::flash_tp_switch::TILE | crate::flash_tp_switch::SEQ;
                            // A prefix command carries no bit of the pass before it.
                            assert!(!command(kind & !bits) || kind & bits == 0, "prefix command {kind:#x} carries pass bits");
                            let kind = kind & !bits;
                            if matches!(kind, CAPTURE | RESTORE) { assert!(valid_live_mask(rows[6].to_bits(), slot)); }
                            match kind {
                                CAPTURE => { assert_eq!(live[&slot].0, pos as usize); if id != 2 { saved.insert(id, live[&slot]); } set_charge(rows, if id == 2 { 0 } else { 128 }); }
                                PUBLISH => { assert!(saved.contains_key(&id)); set_charge(rows, 1); }
                                RESTORE => { live.insert(slot, *saved.get(&id).ok_or("missing ticket")?); set_charge(rows, 1); }
                                RELEASE => { saved.remove(&id); set_charge(rows, 1); }
                                ROOM => {
                                    ROOMS.fetch_add(1, Ordering::SeqCst);
                                    for (word, part) in rows[2..6].iter_mut().zip(4096i128.to_le_bytes().chunks_exact(4)) {
                                        *word = f32::from_bits(u32::from_le_bytes(part.try_into().unwrap()));
                                    }
                                    set_charge(rows, 1);
                                }
                                KIND_ROWS | KIND_ROWS_ASYNC => {
                                    let state = live.entry(slot).or_default();
                                    if pos == 0 { *state = (0, 0); }
                                    assert_eq!(state.0, pos as usize);
                                    for row in rows.chunks_mut(W) {
                                        state.1 = state.1.rotate_left(5).wrapping_add(row[0].to_bits()).wrapping_mul(17);
                                        state.0 += 1;
                                        row[0] = f32::from_bits(state.1);
                                    }
                                }
                                _ => return Err("unexpected CPU prompt command".into()),
                            }
                            Ok(())
                        }).unwrap();
                    assert!(saved.is_empty(), "last owner did not release its checkpoint");
                }));
            }
            let mut link = Link::connect(front, None, Stage::new(0, ranks, None).unwrap(), 512, 8, 64, &STOP, recipe).unwrap();
            let prompt: Vec<u32> = (0..128).map(|i| i * 7 + 11).collect();
            let prefill = |link: &mut Link, slot, pos, tokens: &[u32]| {
                let mut rows = vec![0.0; tokens.len() * W];
                for (row, &token) in rows.chunks_mut(W).zip(tokens) { row[0] = f32::from_bits(token); }
                link.exchange(KIND_ROWS, &[[slot, 512, pos, tokens.len() as u32]], &mut rows).unwrap();
                rows.chunks(W).map(|row| row[0].to_bits()).collect::<Vec<_>>()
            };
            let mut first = vec![0.0; 64 * W];
            for (row, &token) in first.chunks_mut(W).zip(&prompt) { row[0] = f32::from_bits(token); }
            link.post(&[[1, 512, 0, 64]], first, None).unwrap();
            let lease = std::rc::Rc::new(link.cache_capture([1, 512, 64, 0], 1).unwrap().unwrap());
            assert!(link.cache_capture([1, 512, 64, 0], 1).unwrap().is_none(), "optional refusal must preserve the link");
            assert!(!link.poisoned && link.cache.owners.len() == 1);
            let other_owner = lease.clone();
            drop(lease);
            link.cache_releases().unwrap();
            let lease = other_owner;
            assert_eq!(lease.bytes, 128);
            assert!(link.cache_command(PUBLISH, lease.meta).unwrap() > 0);
            // A stale sample and a queued tile: no probe, the last sample stays.
            link.cache.room = Some((Instant::now().checked_sub(Duration::from_secs(2)).unwrap(), 77));
            let mut tile = vec![0.0; W];
            tile[0] = f32::from_bits(5);
            link.post(&[[4, 512, 0, 1]], tile, None).unwrap();
            assert_eq!((link.cache_room().unwrap(), link.pending, ROOMS.load(Ordering::SeqCst)), (Some(77), 1, 0));
            prefill(&mut link, 4, 1, &[6]);
            assert_eq!((link.cache_room().unwrap(), ROOMS.load(Ordering::SeqCst)), (Some(4096), 1), "an idle link probes");
            // A sample taken now (here dated later, so the check does not depend on host timing) is reused.
            link.cache.room = Some((Instant::now() + Duration::from_secs(60), 4096));
            assert_eq!((link.cache_room().unwrap(), ROOMS.load(Ordering::SeqCst)), (Some(4096), 1), "at most once per second");
            let uncached_ids = prefill(&mut link, 1, 64, &prompt[64..]);
            prefill(&mut link, 1, 0, &[999]);
            let cached_tokens = lease.meta[2];
            (link.tp_tile, link.tp_seq) = (true, true);
            link.cache_command_live(RESTORE, [2, 512, cached_tokens, lease.id], 2).unwrap();
            assert!(link.tp_tile && link.tp_seq, "a prefix command keeps the last pass's bits");
            (link.tp_tile, link.tp_seq) = (false, false);
            let cached_ids = prefill(&mut link, 2, cached_tokens, &prompt[cached_tokens as usize..]);
            assert_eq!(cached_tokens, 64);
            assert_eq!(cached_ids, uncached_ids);
            // A canceled request's tile can leave its reply unread: maintenance under load keeps
            // the release queued, and the idle server reads the reply and sends it.
            let mut tile = vec![0.0; W];
            tile[0] = f32::from_bits(9);
            link.post(&[[5, 512, 0, 1]], tile, None).unwrap();
            drop(lease);
            link.cache_releases_when(false).unwrap();
            assert_eq!((link.cache.owners.len(), link.pending), (1, 1), "a release waits while a tile is queued");
            link.cache_releases_when(true).unwrap();
            assert_eq!((link.cache.owners.len(), link.pending), (0, 0), "the idle server sends the release");
            prefill(&mut link, 3, 0, &[17]);
            drop(link);
            for worker in workers { worker.join().unwrap(); }
        }
    }
}
