//! Host-testable bounds and pressure decisions for returned device arenas.
use std::time::{Duration, Instant};

/// Idle time after which a reserve releases its spares (a funded one only under pressure).
pub(crate) const IDLE_RELEASE: Duration = Duration::from_secs(30);
pub(crate) fn vector_capacity(bytes:usize,available:usize)->usize {
    let rounded=bytes.checked_add(4095).map_or(bytes,|rounded|rounded/4096*4096);
    if rounded<=available {rounded} else {bytes}
}

/// Cumulative ordinary KV arena events. Segmented virtual allocations are excluded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvArenaStats {
    pub allocated_count: usize, pub allocated_bytes: usize,
    pub reused_count: usize, pub reused_bytes: usize,
    pub returned_count: usize, pub returned_bytes: usize,
    pub direct_freed_count: usize, pub direct_freed_bytes: usize,
    pub evicted_count: usize, pub evicted_bytes: usize,
}
impl KvArenaStats {
    /// Ordinary bytes the process holds now, live or spare.
    pub(crate) fn held_bytes(&self) -> usize {
        self.allocated_bytes.saturating_sub(self.direct_freed_bytes).saturating_sub(self.evicted_bytes)
    }
}

#[derive(Debug)]
pub(crate) struct ReservePolicy {
    pub(crate) limit: usize, pub(crate) conversation: usize, pub(crate) cohort: usize,
    pub(crate) margin: usize, pub(crate) pressure: bool,
    pub(crate) available: usize, pub(crate) accounted: usize,
    /// Admitted session limit (0: no session plan) and the held bytes before the first session.
    pub(crate) sessions: usize, pub(crate) base: usize,
    /// Start of the current idle period: no request is active, queued or being admitted.
    pub(crate) idle_since: Option<Instant>,
}
impl Default for ReservePolicy {
    fn default() -> Self {
        Self { limit: 0, conversation: 0, cohort: 4, margin: 0, pressure: false, available: 0, accounted: 0, sessions: 0, base: 0, idle_since: None }
    }
}
impl ReservePolicy {
    pub(crate) const MINIMUM: usize = 2_000_000_000;
    pub(crate) fn configure(&mut self, margin: usize) { self.limit = Self::MINIMUM; self.margin = margin; }
    pub(crate) fn cohort(&mut self, chats: usize) {
        self.cohort = self.cohort.max(chats);
        self.conversation_ended(self.conversation);
    }
    /// Memory admission funds `sessions` live sessions. The first call records `held`
    /// as the base; a later call (a smaller agreed limit) keeps that base.
    pub(crate) fn fund(&mut self, sessions: usize, held: usize) {
        if self.sessions == 0 { self.base = held; }
        self.sessions = sessions;
    }
    pub(crate) fn conversation_ended(&mut self, bytes: usize) {
        if self.limit == 0 { return; }
        self.conversation = self.conversation.max(bytes);
        self.limit = Self::MINIMUM.max(self.conversation.saturating_mul(self.cohort));
    }
    pub(crate) fn clear(&mut self) {
        self.conversation = 0;
        if self.limit != 0 { self.limit = Self::MINIMUM; }
    }
    /// True while the ordinary bytes this pool holds above its base, live or spare, fit
    /// the session plan: the session limit times the largest session reservation. A
    /// returned buffer is still held, so keeping it does not raise host use above what
    /// memory admission already funds for live sessions.
    pub(crate) fn funded(&self, held: usize) -> bool {
        self.sessions != 0 && self.conversation != 0
            && held.saturating_sub(self.base) <= self.sessions.saturating_mul(self.conversation)
    }
    /// Earlier fields of this session may have freed memory since a pressure poll.
    /// A fresh host reading must confirm relief before any later arena is kept.
    pub(crate) fn needs_refresh(&self, current_accounted: usize) -> bool {
        self.pressure && current_accounted < self.accounted
    }
    pub(crate) fn poll(&mut self, available: usize, accounted: usize) {
        self.available = available;
        self.accounted = accounted;
        self.pressure = available < self.margin;
    }
    /// Maintenance marks each scheduler pass idle or busy. Admission marks busy before it
    /// can allocate, so a pressure poll during admission never takes the idle release.
    pub(crate) fn idle(&mut self, idle: bool, now: Instant) {
        if idle { self.idle_since.get_or_insert(now); } else { self.idle_since = None; }
    }
    pub(crate) fn idle_long(&self, now: Instant) -> bool {
        self.idle_since.is_some_and(|start| now.duration_since(start) >= IDLE_RELEASE)
    }
    /// `funded`: the return passed `funded` above. It is the only exception to pressure.
    pub(crate) fn can_keep(&self, retained: usize, bytes: usize, funded: bool) -> bool {
        (!self.pressure || funded) && retained.saturating_add(bytes) <= self.limit
    }
    /// Under pressure, whether one spare must go back to the device allocator. Segmented
    /// pages always go. Ordinary spares go while the pool holds more than its session plan
    /// or after a long idle period. A pool without a session plan releases every spare.
    pub(crate) fn sheds(&self, segment: bool, held: usize, idle_long: bool) -> bool {
        segment || idle_long || !self.funded(held)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn vector_capacity_preserves_extents_without_overflow() {
        for bytes in [0,1,4,32,255,256,4095,4096,4097,16383,16384,619520,usize::MAX-4096,usize::MAX] {
            let capacity=super::vector_capacity(bytes,usize::MAX);assert!(capacity>=bytes && capacity-bytes<4096);
            if bytes<=usize::MAX-4095 {assert_eq!(capacity%4096,0);}
            assert_eq!(super::vector_capacity(bytes,bytes),bytes);
            assert_eq!(super::vector_capacity(bytes,0),bytes);
        }
        for bytes in 1..=4096 {assert_eq!(super::vector_capacity(bytes,4096),4096);}
    }
    use super::*;

    #[test]
    fn cohort_cap_tracks_observed_highwater_and_survives_pressure_clear() {
        let mut p = ReservePolicy::default();
        p.cohort(7);
        p.conversation_ended(3_000_000_000);
        assert_eq!(p.limit, 0);
        p.configure(500);
        p.conversation_ended(3_000_000_000);
        assert_eq!(p.limit, 21_000_000_000);
        p.cohort(9);
        p.cohort(2);
        assert_eq!((p.cohort, p.limit), (9, 27_000_000_000));
        p.poll(499, 1000);
        p.clear();
        assert_eq!((p.cohort, p.limit, p.conversation), (9, 2_000_000_000, 0));
        p.conversation_ended(3_000_000_000);
        assert_eq!(p.limit, 27_000_000_000);
        assert!(!p.can_keep(0, 1, false));
        p.poll(500, 900);
        assert!(p.can_keep(26_000_000_000, 1_000_000_000, false));
        assert!(!p.can_keep(26_000_000_000, 1_000_000_001, false));
    }

    #[test]
    fn freed_device_bytes_require_a_new_host_read_before_retention() {
        let mut p = ReservePolicy::default();
        p.configure(500);
        p.poll(400, 1000);
        assert!(!p.needs_refresh(1000));
        assert!(!p.needs_refresh(1100));
        assert!(p.needs_refresh(900));
        // A failed read leaves the last verdict intact.
        assert!(!p.can_keep(0, 100, false));
        p.poll(450, 900);
        assert!(!p.can_keep(0, 100, false));
        assert!(!p.needs_refresh(900));
        assert!(p.needs_refresh(800));
        p.poll(500, 800);
        assert!(p.can_keep(0, 100, false));
        assert!(!p.needs_refresh(700));
    }

    #[test]
    fn return_refresh_restores_large_bound_after_an_intermediate_pressure_clear() {
        let mut p = ReservePolicy::default();
        p.configure(500);
        p.cohort(8);
        p.conversation_ended(4_000_000_000);
        p.poll(400, 1000);
        let conversation = p.conversation;
        assert!(p.needs_refresh(900));
        p.poll(450, 900);
        p.clear();
        p.conversation_ended(conversation);
        assert_eq!(p.limit, 32_000_000_000);
        assert!(!p.can_keep(0, 3_000_000_000, false));
        assert!(p.needs_refresh(800));
        p.poll(500, 800);
        p.conversation_ended(conversation);
        assert!(p.can_keep(3_000_000_000, 3_000_000_000, false));
    }

    #[test]
    fn funded_returns_survive_pressure_only_inside_the_session_plan() {
        let mut p = ReservePolicy::default();
        p.configure(500);
        p.conversation_ended(1000);
        assert!(!p.funded(0));
        p.fund(3, 700);
        p.fund(2, 9000);
        assert_eq!((p.sessions, p.base), (2, 700));
        p.poll(400, 5000);
        assert!(p.funded(2700) && !p.funded(2701));
        assert!(p.can_keep(0, 100, p.funded(2700)));
        assert!(!p.can_keep(0, 100, p.funded(2701)));
        assert!(!p.can_keep(p.limit, 1, true));
        assert!(p.sheds(true, 700, false));
        assert!(!p.sheds(false, 2700, false) && p.sheds(false, 2701, false));
        assert!(p.sheds(false, 700, true));
        // A cleared bound funds nothing until the serving loop restores it.
        p.clear();
        assert!(!p.funded(700));
        p.conversation_ended(1000);
        assert!(p.funded(2700));
    }

    // Replays the saved C8 receipt r2-c8-pressure for 2fcbcbed (rank 0 server.log): margin,
    // session reservation, session limit, arena and vector sizes, and MemAvailable readings.
    const MARGIN: usize = 12_132_553_932;
    const RESERVATION: usize = 2_276_963_652;
    const SESSIONS: usize = 9;
    const ARENA: usize = 150_994_944;
    const ARENAS: usize = 12;
    const VECTOR: usize = 25_064_772;
    const VECTORS: usize = 25;
    /// MemAvailable with eight live chats at lines 57, 187, 308 and 429.
    const PEAKS: [usize; 4] = [10_891_083_776, 11_574_718_464, 11_523_444_736, 11_176_812_544];
    /// Host bytes one retirement freed outside device accounting at line 76:
    /// 12179644416 - 10891083776 - 25 vectors - 1 arena.
    const HOST: usize = 510_946_396;

    /// Mirrors Reserve's keep, take, refresh and pressure release with these predicates.
    struct Pool { policy: ReservePolicy, stats: KvArenaStats, spares: Vec<usize> }
    impl Pool {
        fn new(sessions: usize) -> Self {
            let mut policy = ReservePolicy::default();
            policy.configure(MARGIN);
            policy.cohort(8);
            policy.conversation_ended(RESERVATION);
            if sessions != 0 { policy.fund(sessions, 0); }
            Self { policy, stats: Default::default(), spares: Vec::new() }
        }
        fn take(&mut self, bytes: usize) {
            if let Some(i) = self.spares.iter().position(|&b| b == bytes) {
                self.spares.swap_remove(i);
                self.stats.reused_count += 1;
            } else {
                self.stats.allocated_count += 1;
                self.stats.allocated_bytes += bytes;
            }
        }
        fn poll(&mut self, available: usize, accounted: usize) {
            self.policy.poll(available, accounted);
            if !self.policy.pressure { return; }
            if self.policy.sessions == 0 {
                for bytes in std::mem::take(&mut self.spares) { self.evict(bytes); }
                self.policy.clear();
                // Serving restores the session bound after every maintenance pass.
                self.policy.conversation_ended(RESERVATION);
                return;
            }
            let mut i = 0;
            while i < self.spares.len() {
                if self.policy.sheds(false, self.stats.held_bytes(), false) {
                    let bytes = self.spares.swap_remove(i);
                    self.evict(bytes);
                } else { i += 1; }
            }
        }
        fn evict(&mut self, bytes: usize) { self.stats.evicted_count += 1; self.stats.evicted_bytes += bytes; }
        fn give(&mut self, bytes: usize, host: &dyn Fn(usize) -> usize, accounted: usize) {
            let funded = self.policy.funded(self.stats.held_bytes());
            if self.policy.needs_refresh(accounted) && !funded {
                let conversation = self.policy.conversation;
                self.poll(host(accounted), accounted);
                self.policy.conversation_ended(conversation);
            }
            let retained = self.spares.iter().sum();
            if self.policy.can_keep(retained, bytes, funded) {
                self.spares.push(bytes);
                self.stats.returned_count += 1;
            } else {
                self.stats.direct_freed_count += 1;
                self.stats.direct_freed_bytes += bytes;
            }
        }
    }
    /// MemAvailable falls by every device byte and by `host` bytes for each live chat.
    fn reading(host: usize, peak: usize, accounted: usize, live: usize) -> usize {
        let full = 8 * (ARENAS * ARENA + VECTORS * VECTOR);
        (peak + full + 8 * host).saturating_sub(accounted + live * host)
    }
    struct Host { kv: Pool, vectors: Pool, host: usize }
    impl Host {
        fn new(sessions: usize, host: usize) -> Self { Self { kv: Pool::new(sessions), vectors: Pool::new(sessions), host } }
        fn accounted(&self) -> usize { self.kv.stats.held_bytes() + self.vectors.stats.held_bytes() }
        fn poll(&mut self, peak: usize, live: usize) {
            let a = self.accounted();
            let r = reading(self.host, peak, a, live);
            self.kv.poll(r, a);
            self.vectors.poll(r, a);
        }
        fn admit(&mut self) {
            for _ in 0..VECTORS { self.vectors.take(VECTOR); }
            for _ in 0..ARENAS { self.kv.take(ARENA); }
        }
        /// One chat retires with `live` chats left: host bytes, then vectors, then arenas,
        /// as in the saved retirement order.
        fn retire(&mut self, peak: usize, live: usize) {
            let host = self.host;
            let read = move |accounted: usize| reading(host, peak, accounted, live);
            for _ in 0..VECTORS { let a = self.accounted(); self.vectors.give(VECTOR, &read, a); }
            for _ in 0..ARENAS { let a = self.accounted(); self.kv.give(ARENA, &read, a); }
            self.poll(peak, live);
        }
        /// Warm-up plus three measured rounds of eight chats. Returns fresh arenas per
        /// measured round and the arenas freed or evicted.
        fn c8(sessions: usize, host: usize) -> (Vec<usize>, usize) {
            let mut h = Self::new(sessions, host);
            let mut fresh = Vec::new();
            for (round, &peak) in PEAKS.iter().enumerate() {
                let before = h.kv.stats.allocated_count;
                for _ in 0..8 { h.admit(); }
                if round != 0 { fresh.push(h.kv.stats.allocated_count - before); }
                h.poll(peak, 8);
                assert!(h.kv.policy.pressure, "round {round} peak must be under pressure");
                for live in (0..8).rev() { h.retire(peak, live); }
            }
            (fresh, h.kv.stats.direct_freed_count + h.kv.stats.evicted_count)
        }
    }

    #[test]
    fn recorded_c8_trace_reproduces_the_old_refusal() {
        // The first retirement of the saved warm-up: 25 vectors and 1 arena are freed,
        // then MemAvailable clears the margin and 11 arenas return (lines 76 to 81).
        let mut h = Host::new(0, HOST);
        for _ in 0..8 { h.admit(); }
        h.poll(PEAKS[0], 8);
        h.retire(PEAKS[0], 7);
        assert_eq!((h.vectors.stats.direct_freed_count, h.vectors.stats.returned_count), (25, 0));
        assert_eq!((h.kv.stats.direct_freed_count, h.kv.stats.returned_count), (1, 11));
        assert!(!h.kv.policy.pressure);
        // The next round's eighth chat allocates 1 arena, as chat 16 did (lines 171-172).
        assert_eq!(Host::c8(0, HOST).0[0], 1);
        // Without host bytes per chat, more arenas go before the margin clears.
        let (fresh, lost) = Host::c8(0, 0);
        assert!(fresh[0] > 1 && lost > fresh[0], "{fresh:?} {lost}");
    }

    #[test]
    fn recorded_c8_trace_allocates_no_arena_after_warm_up_with_the_session_plan() {
        for host in [HOST, 0] {
            assert_eq!(Host::c8(SESSIONS, host), (vec![0, 0, 0], 0));
            // Each pool is bounded by its own held bytes: 8 x 12 arenas need more than five
            // session reservations, so a plan of five sessions frees arenas again.
            assert_eq!(Host::c8(8, host), (vec![0, 0, 0], 0));
            let (fresh, lost) = Host::c8(5, host);
            assert!(lost > 0 && fresh.iter().all(|&n| n > 0), "{fresh:?} {lost}");
        }
    }

    #[test]
    fn recorded_c8_pressure_still_releases_spares_above_the_plan_and_after_idle() {
        let mut h = Host::new(SESSIONS, 0);
        for _ in 0..8 { h.admit(); }
        for live in (0..8).rev() { h.retire(PEAKS[0], live); }
        assert_eq!((h.kv.spares.len(), h.vectors.spares.len()), (8 * ARENAS, 8 * VECTORS));
        assert!(h.kv.policy.pressure);
        // A spare of another size raises held bytes above the plan; pressure releases
        // spares until the pool fits it again.
        let extra = SESSIONS * RESERVATION;
        h.kv.stats.allocated_bytes += extra;
        h.kv.spares.push(extra);
        h.poll(PEAKS[0], 0);
        assert!(h.kv.policy.funded(h.kv.stats.held_bytes()));
        assert!(h.kv.stats.evicted_count >= 1);
        assert!(h.kv.policy.sheds(false, h.kv.stats.held_bytes(), true));
    }

    #[test]
    fn admission_ends_the_idle_period_before_a_pressure_poll() {
        let start = Instant::now();
        let mut p = ReservePolicy::default();
        p.configure(MARGIN);
        p.cohort(8);
        p.conversation_ended(RESERVATION);
        p.fund(SESSIONS, 0);
        p.poll(PEAKS[0], 0);
        // Eight chats' arenas wait as spares inside the plan.
        let held = 8 * ARENAS * ARENA;
        assert!(p.pressure && p.funded(held));
        // Later idle passes keep the start of the idle period.
        p.idle(true, start);
        p.idle(true, start + IDLE_RELEASE / 2);
        let late = start + IDLE_RELEASE;
        assert!(p.idle_long(late) && p.sheds(false, held, p.idle_long(late)));
        // Admission marks busy. A pressure poll during admission keeps the spares.
        p.idle(false, late);
        let poll = late + IDLE_RELEASE;
        assert!(!p.idle_long(poll) && !p.sheds(false, held, p.idle_long(poll)));
        assert!(p.sheds(true, held, false) && p.sheds(false, SESSIONS * RESERVATION + 1, false));
        // The next idle pass starts a new period.
        p.idle(true, poll);
        assert!(!p.idle_long(poll + IDLE_RELEASE / 2) && p.idle_long(poll + IDLE_RELEASE));
    }

    #[test]
    fn worker_startup_restores_the_session_bound_cleared_before_funding() {
        // A worker sets the bound, then its workspace allocation polls pressure before any
        // session plan: the reserve clears, as Reserve::shed does without a plan.
        let mut p = ReservePolicy::default();
        p.configure(MARGIN);
        p.conversation_ended(RESERVATION);
        p.poll(PEAKS[0], 0);
        assert!(p.pressure && p.sessions == 0);
        p.clear();
        let held = ARENAS * ARENA;
        p.fund(SESSIONS, 0);
        assert!(!p.funded(held));
        // The worker restores the bound after the workspace, then the plan holds.
        p.conversation_ended(RESERVATION);
        p.fund(SESSIONS, 0);
        assert!(p.funded(held) && p.can_keep(0, ARENA, p.funded(held)));
    }
}
