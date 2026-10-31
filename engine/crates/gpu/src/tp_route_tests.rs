use super::*;
use std::{collections::VecDeque, ops::Range, sync::Mutex};

/// Every origin's part reaches every other rank exactly once, through neighbors only, on paths
/// and rings of 1 to 8 ranks: walking back from rank r along the side it arrives on reaches the
/// origin in `hops` steps, and each rank on the way writes it on toward r.
#[test]
fn routes_reach_every_rank_once_through_neighbors() {
    for world in 1..=8 {
        for ring in [false, true] {
            let links = Links::new(world, ring).unwrap();
            for o in 0..world {
                assert_eq!(links.arrives(o, o), None);
                let mut covered = vec![false; world];
                for side in links.sends(o) { covered[links.neighbor(o, side).unwrap()] = true; }
                for r in (0..world).filter(|&r| r != o) {
                    let (from, hops) = links.arrives(o, r).unwrap_or_else(|| panic!("{world} {ring} {o} -> {r}"));
                    let (mut at, mut steps) = (r, 0);
                    // Walk back to the origin: each rank before r writes the part on, toward r.
                    loop {
                        let back = links.neighbor(at, from).unwrap();
                        steps += 1;
                        if back == o { assert!(links.sends(o).any(|s| links.neighbor(o, s) == Some(at))); break; }
                        assert_eq!(links.arrives(o, back).map(|a| a.0), Some(from), "{world} {ring} {o} -> {r} via {back}");
                        assert_eq!(links.onward(o, back).map(|s| links.neighbor(back, s)), Some(Some(at)));
                        at = back;
                    }
                    assert_eq!(steps, hops);
                    if links.onward(o, r).is_none() { assert!(hops >= 1); }
                    covered[r] = true;
                }
                assert!(covered.iter().enumerate().all(|(r, &c)| c || r == o));
            }
            let most = if ring { world / 2 } else { world - 1 };
            assert_eq!(links.max_hops(), most, "{world} {ring}");
        }
    }
    assert!(Links::new(0, false).is_err());
    // Four in a row: the ends get their farthest part after 3 hops, the middle ranks after 2.
    let path = Links::new(4, false).unwrap();
    assert_eq!((0..4).map(|o| path.hops(o, 0)).collect::<Vec<_>>(), [0, 1, 2, 3]);
    assert_eq!((0..4).map(|o| path.hops(o, 1)).max(), Some(2));
    assert_eq!((path.onward(0, 1), path.onward(0, 2), path.onward(0, 3), path.onward(3, 1)), (Some(Side::Right), Some(Side::Right), None, Some(Side::Left)));
}

/// One hop keeps the pair's mailbox word for word; a relayed layout has 8 slots per channel, a
/// frame, and header words and slots that never overlap.
#[test]
fn channels_keep_the_pair_layout_and_never_overlap() {
    let pair = Channels::new(&Links::new(2, false).unwrap(), 64).unwrap();
    assert_eq!((pair.slots, pair.frame, pair.slot_words, pair.words()), (TP_MAILBOX_SLOTS, 0, 64, HEADER + 2 * TP_MAILBOX_SLOTS * 64));
    for (o, r) in [(0, 1), (1, 0)] {
        assert_eq!(pair.channel(o, r), 0);
        for slot in 0..TP_MAILBOX_SLOTS {
            assert_eq!((pair.send_slot(slot), pair.recv_slot(0, slot), pair.recv_seq(0, slot)),
                (HEADER + slot * 64, HEADER + (TP_MAILBOX_SLOTS + slot) * 64, RECV_SEQ + slot));
        }
    }
    for (world, ring) in [(3, false), (4, false), (4, true), (5, false), (5, true)] {
        let links = Links::new(world, ring).unwrap();
        let l = Channels::new(&links, 1000).unwrap();
        assert_eq!((l.slots, l.frame, l.slot_words), (2 * Channels::AHEAD.next_power_of_two(), Channels::FRAME, 1000 + Channels::FRAME));
        assert!(l.slots >= 2 * Channels::AHEAD && l.slot_words.is_multiple_of(4));
        for r in 0..world {
            let mut channels: Vec<usize> = (0..world).filter(|&o| o != r).map(|o| l.channel(o, r)).collect();
            channels.sort_unstable();
            assert_eq!(channels, (0..world - 1).collect::<Vec<_>>());
        }
        let mut seqs: Vec<usize> = (0..world - 1).flat_map(|c| (0..l.slots).map(move |s| (c, s))).map(|(c, s)| l.recv_seq(c, s)).collect();
        seqs.sort_unstable();
        seqs.dedup();
        assert_eq!(seqs.len(), (world - 1) * l.slots);
        assert!(seqs.iter().all(|&w| (RECV_SEQ..ERROR).contains(&w)));
        assert!(SEND_SEQ + l.slots <= SEND_LEN && SEND_LEN + l.slots <= RECV_SEQ && FLAG_OUT + l.slots <= FAIL_OUT);
        let last = l.recv_slot(world - 2, l.slots - 1) + l.slot_words;
        assert_eq!((l.recv_slot(0, 0), last), (l.send_slot(l.slots - 1) + l.slot_words, l.words()));
    }
    // Six on a ring need 40 receive sequence words and eight on a path 56: the header has 32,
    // so both are refused.
    assert!(Channels::new(&Links::new(6, true).unwrap(), 64).is_err());
    assert!(Channels::new(&Links::new(8, false).unwrap(), 64).is_err());
    assert!(Channels::new(&Links::new(4, false).unwrap(), 0).is_err());
}

fn tag(o: usize, s: u64) -> u32 { (s as u32) ^ ((o as u32) << 28) }

/// A fake NIC on one link: a post snapshots the words it names; the write is carried out some
/// reaps later and completes later still (pseudo-random delays). The NIC flags a source word that
/// changed between the post and the write (a slot rewritten while the NIC reads it).
struct Nic<'m> { local: &'m [AtomicU32], peer: &'m [AtomicU32], queue: VecDeque<(Vec<u32>, Range<usize>, usize, usize, usize, u64, u32)>,
    flying: VecDeque<(u64, u32)>, seed: u32, torn: &'m AtomicBool, posts: usize, refuse_after: usize }
impl Nic<'_> {
    fn delay(&mut self, most: u32) -> u32 { self.seed ^= self.seed << 13; self.seed ^= self.seed >> 17; self.seed ^= self.seed << 5; self.seed % most }
}
impl TpDoorbell for Nic<'_> {
    fn post(&mut self, from: Range<usize>, to: usize, flag: usize, flag_to: usize, id: u64) -> Result<(), String> {
        if id != u64::MAX {
            self.posts += 1;
            if self.posts > self.refuse_after { return Err("post refused".into()); }
        }
        let wait = self.delay(30);
        let snapshot = self.local[from.clone()].iter().map(|w| w.load(Ordering::Relaxed)).collect();
        self.queue.push_back((snapshot, from, to, flag, flag_to, id, wait));
        Ok(())
    }
    fn reap(&mut self, done: &mut dyn FnMut(u64)) -> Result<(), String> {
        if let Some(head) = self.queue.front_mut() { head.6 = head.6.saturating_sub(1); }
        while self.queue.front().is_some_and(|q| q.6 == 0) {
            let (snapshot, from, to, flag, flag_to, id, _) = self.queue.pop_front().unwrap();
            for (i, w) in from.enumerate() {
                let now = self.local[w].load(Ordering::Relaxed);
                if now != snapshot[i] { self.torn.store(true, Ordering::Relaxed); }
                self.peer[to + i].store(now, Ordering::Relaxed);
            }
            self.peer[flag_to].store(self.local[flag].load(Ordering::Relaxed), Ordering::Release);
            let late = self.delay(120);
            self.flying.push_back((id, late));
        }
        if let Some(head) = self.flying.front_mut() { head.1 = head.1.saturating_sub(1); }
        while self.flying.front().is_some_and(|f| f.1 == 0) { done(self.flying.pop_front().unwrap().0); }
        Ok(())
    }
    fn notify(&mut self, from: usize, to: usize) -> Result<(), String> { self.post(from..from + 1, to, from, to, u64::MAX) }
}

/// One rank's GPU stream: puts its parts tagged (rank, sequence) and takes each origin's parts
/// from that origin's channel, checking every word. A take that waits past `limit` sets ERROR,
/// as the take kernel does.
struct Gpu<'m> { m: &'m [AtomicU32], l: Channels, rank: usize, sent: u64, taken: Vec<u64>, limit: Duration, seed: u32, slow: bool }
impl Gpu<'_> {
    fn nap(&mut self) {
        if !self.slow { return; }
        self.seed ^= self.seed << 13; self.seed ^= self.seed >> 17; self.seed ^= self.seed << 5;
        if self.seed.is_multiple_of(4) { std::thread::sleep(Duration::from_micros(u64::from(self.seed % 300))); }
    }
    fn put(&mut self, words: usize) {
        self.nap();
        self.sent += 1;
        let slot = self.sent as usize % self.l.slots;
        let rows = self.l.send_slot(slot) + self.l.frame;
        for w in &self.m[rows..rows + words] { w.store(tag(self.rank, self.sent), Ordering::Relaxed); }
        self.m[SEND_LEN + slot].store((words * 4) as u32, Ordering::Relaxed);
        self.m[SEND_SEQ + slot].store(self.sent as u32, Ordering::Release);
    }
    fn take(&mut self, o: usize, words: usize) -> Result<(), String> {
        self.nap();
        self.taken[o] += 1;
        let s = self.taken[o];
        let (c, slot) = (self.l.channel(o, self.rank), s as usize % self.l.slots);
        let deadline = Instant::now() + self.limit;
        while self.m[self.l.recv_seq(c, slot)].load(Ordering::Acquire) != s as u32 {
            if self.m[ERROR].load(Ordering::Acquire) != 0 { return Err(format!("rank {} error before take {s} of {o}", self.rank)); }
            if Instant::now() > deadline { self.m[ERROR].store(1, Ordering::Release); return Err(format!("rank {} take {s} of {o} timed out", self.rank)); }
            std::thread::yield_now();
        }
        let rows = self.l.recv_slot(c, slot) + self.l.frame;
        if self.m[rows..rows + words].iter().any(|w| w.load(Ordering::Relaxed) != tag(o, s)) { return Err(format!("rank {} take {s} of {o} torn", self.rank)); }
        Ok(())
    }
    /// The split's call order for `passes` passes of `layers` layers: rank 0's residual post (all
    /// take it), per layer a gather pair (both puts, then both takes from every origin) and a single
    /// gather of a changing size, then the head (ranks other than 0 put, every rank takes).
    fn passes(&mut self, world: usize, passes: usize, layers: usize, most: usize) -> Result<(), String> {
        let others: Vec<usize> = (0..world).filter(|&o| o != self.rank).collect();
        for p in 0..passes {
            if self.rank == 0 { self.put(most); } else { self.take(0, most)?; }
            for layer in 0..layers {
                let (x, y, z) = (1 + (p * 7 + layer * 13) % most, 1 + layer % 5, 1 + (layer * 31) % most);
                self.put(x);
                self.put(y);
                for &o in &others { self.take(o, x)?; }
                for &o in &others { self.take(o, y)?; }
                self.put(z);
                for &o in &others { self.take(o, z)?; }
            }
            if self.rank != 0 { self.put(3); }
            for &o in others.iter().filter(|&&o| o != 0) { self.take(o, 3)?; }
        }
        Ok(())
    }
}

/// Build `world` mailboxes, run each rank's relay over fake NICs and each rank's GPU stream.
/// `refuse`: (rank, side, posts) whose NIC refuses data writes after that many. `dead`: a rank
/// whose relay never runs. Returns the GPU streams' results, the relays' results (None: never
/// ran) and whether a NIC saw a torn slot.
fn simulate(links: Links, first: u64, refuse: Option<(usize, Side, usize)>, dead: Option<usize>, limit: Duration)
    -> (Vec<Result<(), String>>, Vec<Option<Result<(), String>>>, bool) {
    let world = links.world();
    let most = 40;
    let l = Channels::new(&links, most).unwrap();
    let memory: Vec<Vec<AtomicU32>> = (0..world).map(|_| (0..l.words()).map(|_| AtomicU32::new(0)).collect()).collect();
    let (stop, torn) = (AtomicBool::new(false), AtomicBool::new(false));
    let relays = Mutex::new(vec![None; world]);
    let timing = Ring { reuse: Duration::from_secs(5), notify: Duration::from_millis(200), ..Ring::default() };
    let gpus = std::thread::scope(|scope| {
        struct Stop<'s>(&'s AtomicBool);
        impl Drop for Stop<'_> { fn drop(&mut self) { self.0.store(true, Ordering::Relaxed); } }
        let _stop = Stop(&stop);
        for rank in (0..world).filter(|&r| Some(r) != dead) {
            let (memory, relays, stop, torn) = (&memory, &relays, &stop, &torn);
            scope.spawn(move || {
                let nic = |side: Side| links.neighbor(rank, side).filter(|&n| n != rank).map(|peer| Nic { local: &memory[rank], peer: &memory[peer],
                    queue: VecDeque::new(), flying: VecDeque::new(), seed: 7 + 31 * rank as u32 + side.index() as u32, torn,
                    posts: 0, refuse_after: refuse.filter(|r| (r.0, r.1) == (rank, side)).map_or(usize::MAX, |r| r.2) });
                let (mut left, mut right) = (nic(Side::Left), nic(Side::Right));
                let mut relay = Relay::new(&memory[rank], links, l, rank).unwrap();
                relay.first = first;
                let wires = [left.as_mut().map(|n| n as &mut dyn TpDoorbell), right.as_mut().map(|n| n as &mut dyn TpDoorbell)];
                let result = relay.run(wires, stop, timing);
                relays.lock().unwrap()[rank] = Some(result);
            });
        }
        let streams: Vec<_> = (0..world).map(|rank| {
            let memory = &memory;
            scope.spawn(move || {
                let mut g = Gpu { m: &memory[rank], l, rank, sent: first - 1, taken: vec![first - 1; world], limit, seed: 3 + rank as u32, slow: rank == 2 };
                g.passes(world, 4, 10, most)
            })
        }).collect();
        let results: Vec<Result<(), String>> = streams.into_iter().map(|s| s.join().unwrap()).collect();
        stop.store(true, Ordering::Relaxed);
        results
    });
    (gpus, relays.into_inner().unwrap(), torn.load(Ordering::Relaxed))
}

/// Four ranks on a path and on a ring (and three and five on a path): every take gets its
/// origin's whole part, with late writes and later completions, a slow rank, and sequence words
/// that wrap at 32 bits; no NIC reads a slot while it is rewritten; every relay stops cleanly.
#[test]
fn ranks_gather_whole_parts_through_relays() {
    for (world, ring) in [(4, false), (4, true), (3, false), (5, false)] {
        let links = Links::new(world, ring).unwrap();
        let first = u64::from(u32::MAX) - 50;
        let (gpus, relays, torn) = simulate(links, first, None, None, Duration::from_secs(20));
        for (rank, g) in gpus.iter().enumerate() { assert_eq!(g, &Ok(()), "{world} {ring} rank {rank}"); }
        for (rank, r) in relays.iter().enumerate() { assert_eq!(r, &Some(Ok(())), "{world} {ring} relay {rank}"); }
        assert!(!torn, "{world} {ring}: a NIC read a slot while it was rewritten");
    }
}

/// A wire that fails in the middle (a middle rank's right link refuses writes) fails every
/// rank's takes through the ERROR words, and no stream computes on missing rows.
#[test]
fn a_failed_wire_fails_every_rank() {
    let links = Links::new(4, false).unwrap();
    let (gpus, relays, _) = simulate(links, 1, Some((1, Side::Right, 25)), None, Duration::from_secs(20));
    for (rank, g) in gpus.iter().enumerate() {
        let error = g.as_ref().unwrap_err();
        assert!(error.contains("error before take"), "rank {rank}: {error}");
    }
    assert!(relays[1].as_ref().unwrap().as_ref().unwrap_err().contains("post refused"));
    for rank in [0, 2, 3] { assert!(relays[rank].as_ref().unwrap().is_err(), "relay {rank} forwards the failure"); }
}

/// A rank whose wire thread never runs: the takes that need it time out, and the timeout's ERROR
/// reaches every other rank, so all four streams fail within the take limit (none hangs).
#[test]
fn a_dead_relay_fails_every_rank_by_timeout() {
    let links = Links::new(4, false).unwrap();
    let at = Instant::now();
    let (gpus, relays, _) = simulate(links, 1, None, Some(1), Duration::from_millis(500));
    assert!(gpus.iter().all(|g| g.is_err()), "{gpus:?}");
    assert!(relays[1].is_none() && relays.iter().flatten().all(|r| r.is_err()));
    assert!(at.elapsed() < Duration::from_secs(10), "failure took {:?}", at.elapsed());
}

/// A relay needs a neighbor on each side its routes use, and a frame (one-hop layouts stay on
/// the pair's doorbell).
#[test]
fn relay_refuses_missing_links_and_unframed_layouts() {
    let links = Links::new(4, false).unwrap();
    let l = Channels::new(&links, 8).unwrap();
    let memory: Vec<AtomicU32> = (0..l.words()).map(|_| AtomicU32::new(0)).collect();
    let mut relay = Relay::new(&memory, links, l, 1).unwrap();
    let error = relay.run([None, None], &AtomicBool::new(false), Ring::default()).unwrap_err();
    assert!(error.contains("needs its"), "{error}");
    assert_eq!(memory[ERROR].load(Ordering::Acquire), 1);
    assert!(Relay::new(&memory, links, l, 4).is_err());
    let pair = Links::new(2, false).unwrap();
    let p = Channels::new(&pair, 8).unwrap();
    let small: Vec<AtomicU32> = (0..p.words()).map(|_| AtomicU32::new(0)).collect();
    assert!(Relay::new(&small, pair, p, 0).is_err());
}
