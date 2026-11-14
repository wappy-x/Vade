//! Host ownership counts through the serving loop's production retirement drain.
use super::*;
use std::{cell::RefCell, collections::BTreeSet, rc::Rc};

#[allow(dead_code)]
#[path = "../../gpu/src/reserve_policy.rs"]
mod reserve_policy;
use reserve_policy::ReservePolicy;

const ATTENTION_ARENAS: usize = 12;
const ARENA_BYTES: usize = 256;
const TRANSIENT_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts { allocated: usize, returned: usize, freed: usize, reused: usize }

struct Device {
    policy: ReservePolicy,
    available: usize,
    accounted: usize,
    reserve: Vec<usize>,
    live: BTreeSet<usize>,
    counts: Counts,
    reads: usize,
    fail_reads: bool,
    arena_bytes: usize,
    transient_bytes: usize,
    refresh_returns: bool,
}
impl Device {
    fn new(chats: usize, margin: usize, available: usize, fail_reads: bool) -> Rc<RefCell<Self>> {
        let mut policy = ReservePolicy::default();
        policy.configure(margin);
        policy.cohort(chats);
        policy.conversation_ended(ATTENTION_ARENAS * ARENA_BYTES + TRANSIENT_BYTES);
        Rc::new(RefCell::new(Self { policy, available, accounted: 0, reserve: Vec::new(),
            live: BTreeSet::new(), counts: Counts::default(), reads: 0, fail_reads,
            arena_bytes: ARENA_BYTES, transient_bytes: TRANSIENT_BYTES, refresh_returns: true }))
    }
    fn poll(&mut self) -> Result<(), String> {
        self.reads += 1;
        if self.fail_reads { return Err("injected pressure read failure".into()); }
        self.policy.poll(self.available, self.accounted);
        if !self.policy.pressure { return Ok(()); }
        if self.policy.sessions == 0 { self.policy.clear(); return Ok(()); }
        while !self.reserve.is_empty() && self.policy.sheds(false, self.held(), false) {
            self.reserve.pop();
            self.free_arena();
        }
        Ok(())
    }
    fn refresh(&mut self) -> Result<(), String> {
        let result = self.poll();
        self.policy.conversation_ended(ATTENTION_ARENAS * self.arena_bytes + self.transient_bytes);
        result
    }
    fn allocate(&mut self) -> usize {
        let id = if let Some(id) = self.reserve.pop() {
            self.counts.reused += 1;
            id
        } else {
            let id = self.counts.allocated;
            self.counts.allocated += 1;
            self.accounted += self.arena_bytes;
            id
        };
        assert!(self.live.insert(id), "an arena cannot have two live owners");
        id
    }
    /// Ordinary arena bytes held, live or spare, as Reserve counts them.
    fn held(&self) -> usize { (self.counts.allocated - self.counts.freed) * self.arena_bytes }
    fn free_arena(&mut self) {
        self.accounted -= self.arena_bytes;
        self.available += self.arena_bytes;
        self.counts.freed += 1;
    }
    fn release(&mut self, id: usize) {
        assert!(self.live.remove(&id), "an arena cannot return twice");
        let funded = self.policy.funded(self.held());
        if self.refresh_returns && self.policy.needs_refresh(self.accounted) && !funded {
            let conversation = self.policy.conversation;
            let _ = self.poll();
            self.policy.conversation_ended(conversation);
        }
        if self.policy.can_keep(self.reserve.len() * self.arena_bytes, self.arena_bytes, funded) {
            self.reserve.push(id);
            self.counts.returned += 1;
        } else {
            self.free_arena();
        }
        self.check_ownership();
    }
    fn check_ownership(&self) {
        assert_eq!(self.counts.allocated, self.live.len() + self.reserve.len() + self.counts.freed);
        assert_eq!(self.reserve.len(), self.reserve.iter().copied().collect::<BTreeSet<_>>().len());
    }
}
struct Transient(Rc<RefCell<Device>>);
impl Drop for Transient {
    fn drop(&mut self) {
        let mut device = self.0.borrow_mut();
        let bytes = device.transient_bytes;
        device.accounted -= bytes;
        device.available += bytes;
    }
}
struct Arena { device: Rc<RefCell<Device>>, id: usize }
impl Drop for Arena {
    fn drop(&mut self) { self.device.borrow_mut().release(self.id); }
}
struct FakeSession {
    // NextSession drops draft, rollback, PLE and GDN state before attention state.
    _transient: Transient,
    _attention: Vec<Arena>,
}
impl FakeSession {
    fn new(device: &Rc<RefCell<Device>>) -> Self {
        {
            let mut device = device.borrow_mut();
            let bytes = device.transient_bytes;
            device.accounted += bytes;
        }
        let attention = (0..ATTENTION_ARENAS).map(|_| {
            let id = device.borrow_mut().allocate();
            Arena { device: device.clone(), id }
        }).collect();
        Self { _transient: Transient(device.clone()), _attention: attention }
    }
}
impl Session for FakeSession {
    fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
}
fn retired(device: &Rc<RefCell<Device>>, chats: usize) -> Vec<Retired<FakeSession>> {
    let finished = Instant::now();
    (0..chats).map(|id| Retired {
        state: CachedSession { checkpoint_lease: None, session: FakeSession::new(device),
            tokens: Vec::new(), next_logits: Vec::new(),
            reserved_session_bytes: {
                let device = device.borrow();
                ATTENTION_ARENAS * device.arena_bytes + device.transient_bytes
            } },
        finished, id: id as u64,
    }).collect()
}

fn recovered_pressure_retirement(chats: usize) {
    let device = Device::new(chats, 1024, 960, false);
    let mut sessions = retired(&device, chats);
    device.borrow_mut().refresh().unwrap();
    assert!(device.borrow().policy.pressure);
    let mut releases = 0;
    let mut refresh = retirement_refresh_hook(true, || {
        releases += 1;
        let mut device = device.borrow_mut();
        assert_eq!(device.counts.returned, releases * ATTENTION_ARENAS);
        assert_eq!(device.counts.freed, 0);
        assert_eq!(device.live.len(), (chats - releases) * ATTENTION_ARENAS);
        device.check_ownership();
        eprintln!("arena_retired chat={releases} allocations={} reserve_returns={} direct_frees={} reuse_hits={} live={} reserve={}",
            device.counts.allocated, device.counts.returned, device.counts.freed, device.counts.reused,
            device.live.len(), device.reserve.len());
        device.refresh()
    }).unwrap();
    reclaim_retired_refresh(&mut sessions, Duration::MAX, &mut refresh);
    drop(refresh);
    assert!(sessions.is_empty());
    assert_eq!(releases, chats);
    let before = device.borrow().counts;
    assert_eq!(before, Counts { allocated: chats * ATTENTION_ARENAS,
        returned: chats * ATTENTION_ARENAS, freed: 0, reused: 0 });
    assert!(!device.borrow().policy.pressure);
    let ids = device.borrow().reserve.iter().copied().collect::<BTreeSet<_>>();
    let next = (0..chats).map(|_| FakeSession::new(&device)).collect::<Vec<_>>();
    {
        let device = device.borrow();
        assert_eq!(device.live, ids);
        assert!(device.reserve.is_empty());
        assert_eq!(device.counts.allocated, before.allocated);
        assert_eq!(device.counts.reused, before.returned);
        device.check_ownership();
    }
    drop(next);
    let device = device.borrow();
    assert!(device.live.is_empty());
    assert_eq!(device.counts.returned, before.returned * 2);
    device.check_ownership();
}

#[test]
fn pair_retirement_returns_each_attention_arena_once_before_readmission() {
    recovered_pressure_retirement(2);
}

#[test]
fn single_retirement_refreshes_after_transient_memory_is_freed() {
    recovered_pressure_retirement(1);
}

#[test]
fn post_session_refresh_alone_loses_the_first_retired_chats_attention_arenas() {
    let device = Device::new(2, 1024, 960, false);
    device.borrow_mut().refresh_returns = false;
    let mut sessions = retired(&device, 2);
    device.borrow_mut().refresh().unwrap();
    let mut releases = 0;
    reclaim_retired_refresh(&mut sessions, Duration::MAX, || {
        releases += 1;
        let mut device = device.borrow_mut();
        assert_eq!(device.counts.freed, ATTENTION_ARENAS);
        assert_eq!(device.counts.returned, (releases - 1) * ATTENTION_ARENAS);
        device.check_ownership();
        device.refresh().unwrap();
    });
    assert_eq!(releases, 2);
    assert!(sessions.is_empty());
    let device = device.borrow();
    assert_eq!(device.counts, Counts { allocated: 24, returned: 12, freed: 12, reused: 0 });
    eprintln!("arena_old_retirement allocations=24 reserve_returns=12 direct_frees=12");
    assert!(device.live.is_empty());
    device.check_ownership();
}

#[test]
fn pair_retirement_frees_every_arena_when_pressure_remains_or_reads_fail() {
    for fail_reads in [false, true] {
        let device = Device::new(2, 1 << 20, 0, fail_reads);
        let mut sessions = retired(&device, 2);
        {
            let mut device = device.borrow_mut();
            let accounted = device.accounted;
            device.policy.poll(0, accounted);
        }
        let mut releases = 0;
        let mut refresh = retirement_refresh_hook(true, || {
            releases += 1;
            let mut device = device.borrow_mut();
            assert_eq!(device.counts.freed, releases * ATTENTION_ARENAS);
            assert_eq!(device.counts.returned, 0);
            device.check_ownership();
            device.refresh()
        }).unwrap();
        reclaim_retired_refresh(&mut sessions, Duration::MAX, &mut refresh);
        drop(refresh);
        assert_eq!(releases, 2);
        assert!(sessions.is_empty());
        let device = device.borrow();
        assert!(device.policy.pressure);
        assert_eq!(device.accounted, 0);
        assert_eq!(device.counts, Counts { allocated: 2 * ATTENTION_ARENAS,
            returned: 0, freed: 2 * ATTENTION_ARENAS, reused: 0 });
        assert!(device.reserve.is_empty());
        assert!(device.live.is_empty());
    }
}

#[test]
fn pair_retirement_keeps_the_large_conversation_cap_after_an_intermediate_pressure_poll() {
    let device = Device::new(2, 250_000_000, 0, false);
    {
        let mut device = device.borrow_mut();
        // Synthetic sizes only; the host test allocates handles, not these bytes.
        device.arena_bytes = 300_000_000;
        device.transient_bytes = 100_000_000;
    }
    let mut sessions = retired(&device, 2);
    device.borrow_mut().refresh().unwrap();
    let mut releases = 0;
    reclaim_retired_refresh(&mut sessions, Duration::MAX, || {
        releases += 1;
        let mut device = device.borrow_mut();
        // The first transient free cannot relieve pressure. The first attention
        // arena frees too. Its next sibling sees relief and keeps the full cap.
        assert_eq!(device.counts.freed, 1);
        assert_eq!(device.counts.returned, releases * ATTENTION_ARENAS - 1);
        assert_eq!(device.policy.conversation, 3_700_000_000);
        assert!(device.reserve.len() * device.arena_bytes > 2_000_000_000);
        device.check_ownership();
        device.refresh().unwrap();
    });
    assert_eq!(releases, 2);
    assert!(sessions.is_empty());
    let device = device.borrow();
    assert_eq!(device.counts, Counts { allocated: 24, returned: 23, freed: 1, reused: 0 });
    assert!(device.live.is_empty());
    device.check_ownership();
}

#[test]
fn funded_pair_retirement_keeps_every_arena_when_pressure_remains_or_reads_fail() {
    for fail_reads in [false, true] {
        let device = Device::new(2, 1 << 20, 0, fail_reads);
        device.borrow_mut().policy.fund(2, 0);
        let mut sessions = retired(&device, 2);
        {
            let mut device = device.borrow_mut();
            let accounted = device.accounted;
            device.policy.poll(0, accounted);
        }
        let mut releases = 0;
        let mut refresh = retirement_refresh_hook(true, || {
            releases += 1;
            let mut device = device.borrow_mut();
            assert_eq!(device.counts.returned, releases * ATTENTION_ARENAS);
            assert_eq!(device.counts.freed, 0);
            device.check_ownership();
            device.refresh()
        }).unwrap();
        reclaim_retired_refresh(&mut sessions, Duration::MAX, &mut refresh);
        drop(refresh);
        assert_eq!(releases, 2);
        assert!(sessions.is_empty());
        // Pressure remains. The funded arenas stay through the poll and serve the next pair.
        assert!(device.borrow().policy.pressure);
        if !fail_reads { device.borrow_mut().refresh().unwrap(); }
        let ids = device.borrow().reserve.iter().copied().collect::<BTreeSet<_>>();
        let next = (0..2).map(|_| FakeSession::new(&device)).collect::<Vec<_>>();
        {
            let device = device.borrow();
            assert_eq!(device.live, ids);
            assert_eq!(device.counts, Counts { allocated: 2 * ATTENTION_ARENAS,
                returned: 2 * ATTENTION_ARENAS, freed: 0, reused: 2 * ATTENTION_ARENAS });
            device.check_ownership();
        }
        drop(next);
        // A plan of one session holds 12 of the 24 arenas: pressure releases the rest.
        let mut device = device.borrow_mut();
        device.policy.fund(1, 0);
        if !fail_reads {
            device.refresh().unwrap();
            assert_eq!((device.reserve.len(), device.counts.freed), (ATTENTION_ARENAS, ATTENTION_ARENAS));
        }
        device.check_ownership();
    }
}
