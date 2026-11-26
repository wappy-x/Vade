use super::*;
use std::{cell::Cell, rc::Rc};
// Regression values from the kg3-w1c-1 server log. Device storage is a byte count only.
const ROOM_POOL: usize = 19_161_848_420;
const ROOM_RETAINED: usize = 9_580_924_210;
const ROOM_LIVE: usize = 4_440_712_516;
const ROOM_RESERVE_CAP: usize = 16_959_722_768;
const ROOM_CHARGE: usize = 2_117_492_316;
const ROOM_SLACK: usize = 1 << 30;
#[derive(Clone)]
struct RoomSnapshot { bytes: usize, forks: Rc<Cell<usize>> }
impl crate::Session for RoomSnapshot {
    fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
    fn retained_bytes(&self) -> Option<usize> { Some(self.bytes) }
    fn fork(&self) -> Result<Self, String> {
        self.forks.set(self.forks.get() + 1);
        Ok(self.clone())
    }
}
fn room_fixture() -> (SessionCache<RoomSnapshot>, CachedSession<RoomSnapshot>) {
    let tokens = 32_103;
    let cache = SessionCache::new(CacheLimits { entries: 16, retained_bytes: ROOM_RETAINED,
        tokens_per_entry: 65_536, logits_per_entry: 1, idle_ttl: Duration::from_secs(3600) }).unwrap();
    let state = CachedSession { checkpoint_lease: None,
        session: RoomSnapshot { bytes: ROOM_CHARGE - tokens * 4 - 4 - index_charge(tokens, None),
            forks: Rc::new(Cell::new(0)) }, tokens: vec![1; tokens], next_logits: vec![1.],
        reserved_session_bytes: ROOM_LIVE };
    (cache, state)
}
#[test]
fn snapshot_uses_room_without_funding_unused_reserve_capacity() {
    let (mut cache, state) = room_fixture();
    let now = Instant::now();
    // Synthetic physical room puts the logged snapshot exactly at the growth limit.
    let free = crate::prefix_switch::cache_physical_room(
        Some(((ROOM_CHARGE + ROOM_SLACK) as i128, ROOM_RESERVE_CAP)), ROOM_SLACK);
    cache.rebalance_within(ROOM_POOL - ROOM_LIVE - (128 << 20), free, BTreeSet::new(), now);
    cache.snapshot(SessionKey::generate().unwrap(), &state, now);
    assert_eq!((cache.len(), cache.retained_bytes(), state.session.forks.get()), (1, ROOM_CHARGE, 1));
    // The copied snapshot consumes room. Retirement only transfers existing allocations.
    let free = crate::prefix_switch::cache_physical_room(
        Some((ROOM_SLACK as i128, ROOM_RESERVE_CAP - 3_623_878_656)), ROOM_SLACK);
    cache.rebalance_within(ROOM_POOL, free, BTreeSet::new(), now);
    assert_eq!((cache.len(), cache.retained_bytes()), (1, ROOM_CHARGE));
    // A new reservation that fills the logical pool still reclaims the whole snapshot.
    cache.rebalance_within(0, free, BTreeSet::new(), now);
    assert_eq!((cache.len(), cache.retained_bytes()), (0, 0));
    cache.snapshot(SessionKey::generate().unwrap(), &state, now);
    assert_eq!((cache.len(), state.session.forks.get()), (0, 1));
}
#[test]
fn snapshot_room_keeps_growth_slack_and_yields_to_actual_pressure() {
    let (mut cache, state) = room_fixture();
    let now = Instant::now();
    let free = crate::prefix_switch::cache_physical_room(
        Some(((ROOM_CHARGE + ROOM_SLACK - 1) as i128, ROOM_RESERVE_CAP)), ROOM_SLACK);
    cache.rebalance_within(ROOM_POOL - ROOM_LIVE, free, BTreeSet::new(), now);
    cache.snapshot(SessionKey::generate().unwrap(), &state, now);
    assert_eq!((cache.len(), state.session.forks.get()), (0, 0));
    cache.rebalance_within(ROOM_POOL - ROOM_LIVE, None, BTreeSet::new(), now);
    cache.snapshot(SessionKey::generate().unwrap(), &state, now);
    assert_eq!(cache.retained_bytes(), ROOM_CHARGE);
    let free = crate::prefix_switch::cache_physical_room(Some((-1, ROOM_RESERVE_CAP)), ROOM_SLACK);
    cache.rebalance_within(ROOM_POOL, free, BTreeSet::new(), now);
    assert_eq!((cache.len(), cache.retained_bytes()), (0, 0));
}
#[test]
fn startup_room_keeps_a_snapshot_under_pressure_and_a_full_pool() {
    // A two-Spark soak under host pressure (no physical room) skipped every snapshot.
    let (cache, state) = room_fixture();
    let mut cache = cache.with_room(ROOM_CHARGE);
    let now = Instant::now();
    let pressure = crate::prefix_switch::cache_physical_room(Some((-1, ROOM_RESERVE_CAP)), ROOM_SLACK);
    cache.rebalance_within(0, pressure, BTreeSet::new(), now);
    cache.snapshot(SessionKey::generate().unwrap(), &state, now);
    assert_eq!((cache.len(), cache.retained_bytes(), state.session.forks.get()), (1, ROOM_CHARGE, 1));
    // Neither admission (no pool bytes left) nor pressure reclaims the room.
    cache.rebalance_within(0, pressure, BTreeSet::new(), now);
    assert_eq!((cache.len(), cache.retained_bytes()), (1, ROOM_CHARGE));
    // The room holds one such snapshot: a new conversation replaces the older one.
    let second = CachedSession { checkpoint_lease: None, session: state.session.clone(),
        tokens: std::iter::once(2).chain(state.tokens[1..].iter().copied()).collect(),
        next_logits: state.next_logits.clone(), reserved_session_bytes: state.reserved_session_bytes };
    cache.snapshot(SessionKey::generate().unwrap(), &second, now);
    assert_eq!((cache.len(), cache.retained_bytes(), state.session.forks.get()), (1, ROOM_CHARGE, 2));
    cache.count_lookup(32_064);
    cache.count_lookup(0);
    assert_eq!(cache.counts(), Counts { hits: 1, misses: 1, reused_tokens: 32_064, snapshots: 2, evicted: 1, expired: 0, skipped: [0; 4] });
    let status = cache.status_json();
    assert!(status.starts_with(r#"{"hits":1,"misses":1,"reused_tokens":32064,"snapshots":2,"evicted":1,"expired":0,"entries":1,"#)
        && status.ends_with(r#""skipped":{"no_room":0,"too_large":0,"fork_failed":0,"insert_refused":0}}"#), "{status}");
    // Without a room the same pressure keeps nothing.
    let (mut plain, state) = room_fixture();
    plain.rebalance_within(0, pressure, BTreeSet::new(), now);
    plain.snapshot(SessionKey::generate().unwrap(), &state, now);
    assert_eq!((plain.len(), plain.counts().skipped), (0, [1, 0, 0, 0]));
}
#[test]
fn the_status_line_follows_an_eviction_outside_a_snapshot() {
    // A rebalance that changes the entries or retained bytes prints a new status line.
    let (mut cache, state) = room_fixture();
    let now = Instant::now();
    cache.snapshot(SessionKey::generate().unwrap(), &state, now);
    assert_eq!(cache.noted, (1, ROOM_CHARGE));
    let pressure = crate::prefix_switch::cache_physical_room(Some((-1, ROOM_RESERVE_CAP)), ROOM_SLACK);
    cache.rebalance_within(0, pressure, BTreeSet::new(), now);
    assert_eq!((cache.len(), cache.noted), (0, (0, 0)));
    // Health says the entry was evicted for memory, not that it expired.
    assert_eq!((cache.counts().evicted, cache.counts().expired), (1, 0));
    assert!(cache.status_json().contains("\"evicted\":1,\"expired\":0,"), "{}", cache.status_json());
    cache.rebalance_within(ROOM_RETAINED, None, BTreeSet::new(), now);
    cache.snapshot(SessionKey::generate().unwrap(), &state, now);
    cache.expire(now + Duration::from_secs(3600 * 24));
    assert_eq!((cache.len(), cache.counts().evicted, cache.counts().expired), (0, 1, 1));
    // A checkpoint removed at its expiry on a scheduler pass counts as expired too.
    cache.checkpoint(None, None, &state, now).unwrap();
    cache.rebalance_within(ROOM_RETAINED, None, BTreeSet::new(), now + Duration::from_secs(3600));
    assert_eq!((cache.len(), cache.counts().evicted, cache.counts().expired), (0, 1, 2));
}
#[test]
fn an_in_place_entry_yields_to_pressure_only_after_the_server_is_idle() {
    // own-cache-screen-x4 1s-p1 (one Spark, context 32768): under host pressure each retired
    // turn-1 entry was evicted at the next pass, but device_bytes did not move: the funded
    // reserves kept its arenas, so the yield freed no host memory and the next turn missed.
    const RESERVATION: usize = 2_276_963_652;
    const ROOM: usize = 916_359_676;
    const ARENA: usize = 7;
    #[derive(Clone)]
    struct InPlace;
    impl crate::Session for InPlace {
        fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
        fn fork(&self) -> Result<Self, String> { Ok(Self) }
        fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> { Some(vec![(ARENA, RESERVATION)]) }
        fn retained_private_bytes(&self) -> usize { 196_725_164 }
    }
    let mut cache = SessionCache::new(CacheLimits { entries: 16, retained_bytes: ROOM_RETAINED,
        tokens_per_entry: 65_536, logits_per_entry: 1, idle_ttl: Duration::from_secs(3600) }).unwrap().with_room(ROOM);
    let state = CachedSession { checkpoint_lease: None, session: InPlace, tokens: vec![1; 32_099],
        next_logits: vec![1.], reserved_session_bytes: RESERVATION };
    let pressure = crate::prefix_switch::cache_physical_room(Some((-1, ROOM_RESERVE_CAP)), ROOM_SLACK);
    let now = Instant::now();
    // Turn 1 is live: its request funds the arena; the entry's own state fits the room.
    cache.note_busy(now);
    cache.rebalance_within(ROOM_POOL - RESERVATION, pressure, BTreeSet::from([ARENA]), now);
    cache.snapshot_scoped(SessionKey::generate().unwrap(), None, &state, now, true);
    assert_eq!(cache.len(), 1);
    // Turn 1 retires: what an eviction returns stays in the reserves, so the entry stays.
    for idle in [1, 29] {
        cache.rebalance_within(ROOM_POOL, pressure, BTreeSet::new(), now + Duration::from_secs(idle));
        assert_eq!((cache.len(), cache.counts().evicted), (1, 0), "idle {idle} s");
    }
    // After 30 s with no live request the reserves release spares: the entry yields.
    cache.rebalance_within(ROOM_POOL, pressure, BTreeSet::new(), now + Duration::from_secs(30));
    assert_eq!((cache.len(), cache.retained_bytes(), cache.counts().evicted), (0, 0, 1));
}
#[test]
fn a_snapshot_evicts_for_the_storage_its_fork_adds() {
    // The FP8 ring session funds a new snapshot copy from its own budget (the cache ceiling,
    // reserved at launch) only when it forks. After a long served run, a card's entries fill
    // the cache (FP8 screen: 10 entries, 8,705,753,848 of 8,883,411,194 bytes); a long chat's
    // snapshot must still be funded, by evicting older entries first. Bytes scaled by 10^4.
    const LIMIT: usize = 888_341;
    struct Store { used: Cell<usize>, next: Cell<usize> }
    struct Held(Rc<Store>, usize, usize);
    impl Drop for Held { fn drop(&mut self) { self.0.used.set(self.0.used.get() - self.2); } }
    #[derive(Clone)]
    struct Ring { store: Rc<Store>, copy: usize, saved: Vec<Rc<Held>> }
    impl crate::Session for Ring {
        fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
        fn fork(&self) -> Result<Self, String> {
            let s = &self.store;
            if s.used.get() + self.copy > LIMIT { return Err("Flash snapshot cache is unfunded".into()); }
            s.used.set(s.used.get() + self.copy);
            s.next.set(s.next.get() + 1);
            let mut saved = self.saved.clone();
            saved.push(Rc::new(Held(s.clone(), s.next.get(), self.copy)));
            Ok(Self { store: s.clone(), copy: 0, saved })
        }
        fn retained_allocations(&self) -> Option<Vec<(usize, usize)>> { Some(self.saved.iter().map(|h| (h.1, h.2)).collect()) }
        fn fork_storage_bytes(&self) -> usize { self.copy }
    }
    let store = Rc::new(Store { used: Cell::new(0), next: Cell::new(0) });
    let mut cache = SessionCache::new(CacheLimits { entries: 128, retained_bytes: LIMIT, tokens_per_entry: 16,
        logits_per_entry: 1, idle_ttl: Duration::from_secs(3600) }).unwrap();
    let now = Instant::now();
    let chat = |token, copy| CachedSession { checkpoint_lease: None, session: Ring { store: store.clone(), copy, saved: Vec::new() },
        tokens: vec![token], next_logits: vec![1.], reserved_session_bytes: copy };
    for token in 1..=10 { cache.snapshot(SessionKey::generate().unwrap(), &chat(token, 87_000), now); }
    assert_eq!((cache.len(), store.used.get()), (10, 870_000));
    // The long chat: the oldest card entries yield, the chat is kept, the budget holds.
    cache.snapshot(SessionKey::generate().unwrap(), &chat(11, 300_000), now);
    assert_eq!((cache.counts().snapshots, cache.counts().skipped), (11, [0; 4]));
    assert!(cache.entries.iter().any(|e| e.state.tokens == [11]) && cache.len() < 10);
    assert_eq!(cache.counts().evicted, 11 - cache.len());
    assert!(store.used.get() <= LIMIT && cache.retained_bytes() <= LIMIT, "{} {}", store.used.get(), cache.retained_bytes());
}
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
#[test]
fn disabled_cache_accepts_the_full_service_context() {
    for (retained_bytes,tokens_per_entry,valid) in [(0,524288,true),(0,1_048_576,true),
        (0,1_048_577,false),(1024,262144,true),(1024,262145,false)] {
        assert_eq!(CacheLimits {retained_bytes,tokens_per_entry,..limits()}.validate().is_ok(),valid);
    }
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
fn idle_entries_expire_without_memory_pressure_at_the_deadline() {
    let d = Rc::new(Cell::new(0));
    let now = Instant::now();
    let mut c = SessionCache::new(CacheLimits { idle_ttl: Duration::from_secs(300), ..limits() }).unwrap();
    c.insert_snapshot(key(1), state(&d, vec![1]), now).unwrap();
    c.rebalance(1024, BTreeSet::new(), now + Duration::from_secs(299));
    assert_eq!((c.len(), d.get()), (1, 0));
    c.rebalance(1024, BTreeSet::new(), now + Duration::from_secs(300));
    assert_eq!((c.len(), c.retained_bytes(), c.counts().expired, c.counts().evicted), (0, 0, 1, 0));
    assert_eq!(d.get(), 1);
}
#[test]
fn a_scheduler_pass_scans_for_idle_prompts_at_most_once_a_second_and_a_lookup_at_once() {
    let (start, forks) = (Instant::now(), Rc::new(Cell::new(0))); let at = |ms| start + Duration::from_millis(ms);
    for lookup in [false, true] {
        let mut c = SessionCache::new(CacheLimits { idle_ttl: Duration::from_secs(300), ..limits() }).unwrap();
        c.insert_snapshot(key(1), kv(&[1, 2], &forks), at(0)).unwrap();
        // A pass at 299.5 s scans and finds nothing. The pass at 300.2 s is the same second: two compares, no scan.
        c.rebalance(1024, BTreeSet::new(), at(299_500)); c.rebalance(1024, BTreeSet::new(), at(300_200));
        assert_eq!((c.len(), c.counts().expired), (1, 0), "a pass inside the second after a scan does not scan");
        if lookup { assert!(c.fork_prefix(&key(1), &[1, 2], at(300_200)).state.is_none()); } else { c.rebalance(1024, BTreeSet::new(), at(300_500)); }
        assert_eq!((c.len(), c.counts().expired), (0, 1), "the scan one second after the last, or a lookup at once: lookup {lookup}");
        // A cache that may keep nothing (zero bytes: prompt keep off) takes no stamp: its passes do no idle work.
        let mut off = SessionCache::<Kv>::new(CacheLimits { retained_bytes: 0, ..limits() }).unwrap();
        for ms in [0, 1_500, 400_000] { off.rebalance_within(0, None, BTreeSet::new(), at(ms)); assert!(off.scanned.is_none(), "{ms}"); }
    }
}
#[test]
fn a_lookup_cannot_refresh_an_expired_prompt() {
    let (now, forks) = (Instant::now(), Rc::new(Cell::new(0)));
    let mut c = SessionCache::new(limits()).unwrap();
    c.insert_snapshot(key(1), kv(&[1, 2], &forks), now).unwrap();
    assert!(c.fork_prefix(&key(1), &[1, 2], now + Duration::from_secs(5)).state.is_none());
    assert_eq!((c.len(), c.counts().expired, forks.get()), (0, 1, 0));
}
#[test]
fn insertion_expires_old_prompts_before_the_entry_ceiling() {
    let (now, d) = (Instant::now(), Rc::new(Cell::new(0)));
    let mut c = SessionCache::new(limits()).unwrap();
    c.insert_snapshot(key(1), state(&d, vec![1]), now).unwrap();
    c.insert_snapshot(key(2), state(&d, vec![2]), now + Duration::from_secs(1)).unwrap();
    c.insert_snapshot(key(3), state(&d, vec![3]), now + Duration::from_secs(5)).unwrap();
    assert_eq!((c.len(), c.counts().expired, c.counts().evicted, d.get()), (2, 1, 0, 1));
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
    assert!(c.fork_automatic(None, &[1, 2], now).state.is_some());
    assert!(c.fork_automatic(None, &[5, 6, 7], now).state.is_some());
    c.insert_snapshot(key(3), kv(&[5, 6, 7], &forks), now).unwrap();
    assert_eq!((c.reusable_tokens(None, &[1, 2, 3]), c.entries[0].key), (2, key(1)));
}
#[test]
fn a_failed_fork_reaches_the_caller_and_a_later_candidate_still_forks() {
    let (now, forks) = (Instant::now(), Rc::new(Cell::new(0)));
    let mut c = SessionCache::new(limits()).unwrap();
    let mut broken = kv(&[1, 2, 3, 4], &forks);
    broken.session.fail_truncate = true;
    c.insert_snapshot(key(1), broken, now).unwrap();
    c.insert_snapshot(key(2), kv(&[1, 2], &forks), now).unwrap();
    // The longer entry is tried first; its truncate fails, so the search goes on to the shorter one.
    let forked = c.fork_automatic(None, &[1, 2, 3, 9], now);
    assert_eq!(forked.failed, ["truncate failed"]);
    assert_eq!(forked.state.unwrap().tokens, [1, 2]);
    let forked = c.fork_automatic(None, &[1, 2, 3, 8], now);
    assert_eq!((forked.failed.len(), forked.state.map(|s| s.tokens)), (1, Some(vec![1, 2])));
    let mut c = SessionCache::new(limits()).unwrap();
    let mut broken = kv(&[1, 2, 3, 4], &forks);
    broken.session.fail_truncate = true;
    c.insert_snapshot(key(1), broken, now).unwrap();
    let forked = c.fork_scoped(&key(1), None, &[1, 2, 3, 9], now);
    assert!(forked.state.is_none(), "every candidate failed");
    assert_eq!(forked.failed, ["truncate failed"], "the miss keeps the error");
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
