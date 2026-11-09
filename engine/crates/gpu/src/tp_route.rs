//! The tensor split's gathers over more than two ranks, host side (std only). Each rank owns one
//! part of a vector and every other rank needs it: an all-gather of raw words, never a sum. The
//! ranks sit on a path 0-1-..-(world - 1) or a ring, each between at most two neighbors, so a part
//! reaches a rank that is not a neighbor through the ranks between them: each one's wire thread
//! writes the part on from its own receive slot (a relay), with no GPU copy and no host copy.
//!
//! Mailbox (Channels): the pair's layout (tp_mailbox_host) with one receive channel per origin
//! rank. Message s of origin o lands in channel `channel(o, r)`, slot s % slots, on every rank r.
//! Only the neighbor on o's side writes that channel, so RC order on that one link keeps it in
//! order, and a take reads a fixed channel per origin: no origin field and no arrival order. A
//! relayed slot starts with a FRAME of four words; the first holds the message's byte length, so
//! a relay forwards exactly the published extent. The rows start FRAME words in (16-byte aligned
//! for the four-word copies). One hop keeps the pair's layout as it is (no frame, its slots).
//!
//! Slots: every rank puts its part of a gather once and takes every other origin's part, and a
//! rank puts at most AHEAD messages before it takes the earlier ones (a gather pair plus the
//! pass's residual post). Origin o puts message s only after it took a part that rank r put after
//! r took o's message s - 2 AHEAD. So with at least 2 AHEAD slots no write lands in a slot whose
//! message rank r has not taken, and a relay's read of a slot is done before the next rank took
//! that message. Each wire also waits for the completion of its previous write to a slot before
//! it writes that slot again, as the pair's doorbell does. A failed rank writes 1 into each
//! neighbor's ERROR word; a relay that sees its own ERROR fails and writes on, so the failure
//! crosses the whole path, and the takes' timeout stays the backstop.
use crate::tp_mailbox_host::{pace, Ring, TpDoorbell, ERROR, FAIL_OUT, FLAG_OUT, HEADER, RECV_SEQ, SEND_LEN, SEND_SEQ, TP_MAILBOX_SLOTS};
use std::{sync::atomic::{AtomicBool, AtomicU32, Ordering}, time::{Duration, Instant}};

/// A rank's link toward lower ranks (Left) or higher ranks (Right; a ring wraps both ways).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side { Left, Right }
impl Side {
    fn index(self) -> usize { match self { Side::Left => 0, Side::Right => 1 } }
}

/// How the ranks are linked: a path, or a ring that also links the two ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Links { world: usize, ring: bool }
impl Links {
    pub fn new(world: usize, ring: bool) -> Result<Self, String> {
        std::num::NonZeroUsize::new(world).ok_or("a gather needs at least one rank")?;
        Ok(Self { world, ring })
    }
    pub fn world(&self) -> usize { self.world }
    /// How many ranks origin `o`'s part covers going right and going left. A ring sends the
    /// longer half to the right.
    fn reach(&self, o: usize) -> (usize, usize) {
        let others = self.world - 1;
        if self.ring { (others - others / 2, others / 2) } else { (others - o, o) }
    }
    /// Steps from origin `o` to rank `r` going right, and going left.
    fn distance(&self, o: usize, r: usize) -> (Option<usize>, Option<usize>) {
        if self.ring { (Some((r + self.world - o) % self.world), Some((o + self.world - r) % self.world)) } else { (r.checked_sub(o), o.checked_sub(r)) }
    }
    /// The side on which origin `o`'s part enters rank `r` and its hops, None for `o` itself.
    pub fn arrives(&self, o: usize, r: usize) -> Option<(Side, usize)> {
        let ((right, left), (to_right, to_left)) = (self.reach(o), self.distance(o, r));
        match (to_right.filter(|&d| (1..=right).contains(&d)), to_left.filter(|&d| (1..=left).contains(&d))) {
            (Some(d), _) => Some((Side::Left, d)),
            (None, Some(d)) => Some((Side::Right, d)),
            (None, None) => None,
        }
    }
    pub fn hops(&self, o: usize, r: usize) -> usize { self.arrives(o, r).map_or(0, |(_, d)| d) }
    /// The side on which rank `r` writes origin `o`'s part on, if a farther rank still needs it.
    pub fn onward(&self, o: usize, r: usize) -> Option<Side> {
        let (from, d) = self.arrives(o, r)?;
        let (right, left) = self.reach(o);
        match from { Side::Left => (d < right).then_some(Side::Right), Side::Right => (d < left).then_some(Side::Left) }
    }
    /// The sides on which origin `o` writes its own part.
    pub fn sends(&self, o: usize) -> impl Iterator<Item = Side> {
        let (right, left) = self.reach(o);
        [(Side::Left, left), (Side::Right, right)].into_iter().filter(|&(_, n)| n > 0).map(|(side, _)| side)
    }
    /// Rank `r`'s neighbor on `side`.
    pub fn neighbor(&self, r: usize, side: Side) -> Option<usize> {
        match (self.ring, side) {
            (true, Side::Right) => Some((r + 1) % self.world),
            (true, Side::Left) => Some((r + self.world - 1) % self.world),
            (false, Side::Right) => Some(r + 1).filter(|&n| n < self.world),
            (false, Side::Left) => r.checked_sub(1),
        }
    }
    /// The most hops any part takes.
    pub fn max_hops(&self) -> usize {
        (0..self.world).flat_map(|o| (0..self.world).map(move |r| (o, r))).map(|(o, r)| self.hops(o, r)).max().unwrap_or(0)
    }
}

/// One rank's mailbox layout for a gather over `world` ranks (the module comment).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Channels { pub world: usize, pub slots: usize, pub frame: usize, pub slot_words: usize }
impl Channels {
    /// Words before a relayed message's rows; the first holds its byte length.
    pub const FRAME: usize = 4;
    /// The most messages a rank puts before it takes the earlier ones (a pair and the residual).
    pub const AHEAD: usize = 3;
    /// The layout for these links and messages of at most `message_words` words.
    pub fn new(links: &Links, message_words: usize) -> Result<Self, String> {
        let relayed = links.max_hops() > 1;
        let (slots, frame) = if relayed { ((2 * Self::AHEAD).next_power_of_two(), Self::FRAME) } else { (TP_MAILBOX_SLOTS, 0) };
        let layout = Self { world: links.world, slots, frame, slot_words: message_words + frame };
        if message_words == 0 || slots > SEND_LEN - SEND_SEQ || slots > FAIL_OUT - FLAG_OUT || (links.world - 1) * slots > ERROR - RECV_SEQ {
            return Err(format!("the tensor split mailbox header has no room for {} ranks of {slots} slots", links.world));
        }
        Ok(layout)
    }
    /// Rank `r`'s receive channel for origin `o` (not `r`).
    pub fn channel(&self, o: usize, r: usize) -> usize { if o < r { o } else { o - 1 } }
    pub fn send_slot(&self, slot: usize) -> usize { HEADER + slot * self.slot_words }
    pub fn recv_slot(&self, channel: usize, slot: usize) -> usize { HEADER + (self.slots * (1 + channel) + slot) * self.slot_words }
    pub fn recv_seq(&self, channel: usize, slot: usize) -> usize { RECV_SEQ + channel * self.slots + slot }
    /// The mailbox's words.
    pub fn words(&self) -> usize { HEADER + self.slots * self.world * self.slot_words }
}

/// One link of a relaying rank: its queue pair and the write ids it posted and completed.
struct Wire<'w> { link: &'w mut dyn TpDoorbell, peer: usize, next: u64, done: u64, last: Vec<u64> }
impl Wire<'_> {
    /// Reap completions (each once, in order).
    fn reap(&mut self) -> Result<(), String> {
        let (mut order, mut done) = (true, self.done);
        self.link.reap(&mut |id| { order &= id == done + 1; done = id; })?;
        self.done = done;
        if order { Ok(()) } else { Err(format!("tensor split write completion out of order (after {done})")) }
    }
}

/// The wire thread of one rank of a relayed gather: it writes this rank's messages to its
/// neighbors and other origins' messages on, as the routes ask.
pub struct Relay<'m> { words: &'m [AtomicU32], links: Links, layout: Channels, rank: usize, first: u64 }
impl<'m> Relay<'m> {
    pub fn new(words: &'m [AtomicU32], links: Links, layout: Channels, rank: usize) -> Result<Self, String> {
        if rank >= links.world || words.len() != layout.words() || layout.world != links.world || layout.frame == 0 {
            return Err("tensor split relay: rank or mailbox outside the layout".into());
        }
        Ok(Self { words, links, layout, rank, first: 1 })
    }
    fn error(&self) -> bool { self.words[ERROR].load(Ordering::Acquire) != 0 }
    /// Until `stop` or an error: `wires` are the left and right neighbors' queue pairs (None where
    /// this rank has no neighbor). On an error this rank fails its mailbox and writes 1 into each
    /// neighbor's ERROR word.
    pub fn run(&mut self, wires: [Option<&mut dyn TpDoorbell>; 2], stop: &AtomicBool, timing: Ring) -> Result<(), String> {
        let mut wires = wires.into_iter().zip([Side::Left, Side::Right]).map(|(link, side)| match (link, self.links.neighbor(self.rank, side)) {
            (Some(link), Some(peer)) => Ok(Some(Wire { link, peer, next: 0, done: 0, last: vec![0; self.layout.world * self.layout.slots] })),
            (None, _) => Ok(None),
            (Some(_), None) => Err(format!("tensor split relay: rank {} has no {side:?} neighbor", self.rank)),
        }).collect::<Result<Vec<_>, String>>()?;
        let result = self.relay(&mut wires, stop, timing);
        if result.is_err() {
            self.words[ERROR].store(1, Ordering::Release);
            self.words[FAIL_OUT].store(1, Ordering::Release);
            for wire in wires.iter_mut().flatten() {
                if wire.link.notify(FAIL_OUT, ERROR).is_ok() {
                    let (until, mut out) = (Instant::now() + timing.notify, false);
                    while !out && Instant::now() < until && wire.link.reap(&mut |id| out |= id == u64::MAX).is_ok() { std::thread::yield_now(); }
                }
            }
        }
        result
    }
    fn relay(&mut self, wires: &mut [Option<Wire<'_>>], stop: &AtomicBool, timing: Ring) -> Result<(), String> {
        let (l, rank) = (self.layout, self.rank);
        let sends: Vec<Side> = self.links.sends(rank).collect();
        let relays: Vec<(usize, Side)> = (0..l.world).filter_map(|o| self.links.onward(o, rank).map(|side| (o, side))).collect();
        for side in sends.iter().chain(relays.iter().map(|(_, s)| s)) {
            if wires[side.index()].is_none() { return Err(format!("tensor split relay: rank {rank} needs its {side:?} link")); }
        }
        let (mut next, mut onward) = (self.first, vec![self.first; l.world]);
        let (mut active, mut pause, mut paced) = (Instant::now(), Duration::from_micros(1), Instant::now());
        while !stop.load(Ordering::Relaxed) {
            for wire in wires.iter_mut().flatten() { wire.reap()?; }
            if self.error() { return Err(format!("mailbox failed before message {next}")); }
            let mut moved = false;
            let slot = next as usize % l.slots;
            if self.words[SEND_SEQ + slot].load(Ordering::Acquire) == next as u32 {
                let len = self.words[SEND_LEN + slot].load(Ordering::Relaxed) as usize;
                if len == 0 || len > (l.slot_words - l.frame) * 4 || !len.is_multiple_of(4) { return Err("tensor split mailbox message length".into()); }
                for side in &sends {
                    let wire = wires[side.index()].as_mut().unwrap();
                    let c = l.channel(rank, wire.peer);
                    if !self.free(wire, c, slot, next, stop, timing)? { return Ok(()); }
                }
                // Both wires completed the slot's previous message: its frame and flag are free.
                self.words[l.send_slot(slot)].store(len as u32, Ordering::Relaxed);
                self.words[FLAG_OUT + slot].store(next as u32, Ordering::Release);
                for side in &sends {
                    let wire = wires[side.index()].as_mut().unwrap();
                    let c = l.channel(rank, wire.peer);
                    post(wire, l.send_slot(slot)..l.send_slot(slot) + l.frame + len / 4, l.recv_slot(c, slot), FLAG_OUT + slot, l.recv_seq(c, slot), c * l.slots + slot)?;
                }
                next += 1;
                moved = true;
            }
            for &(o, side) in &relays {
                let s = onward[o];
                let (c, slot) = (l.channel(o, rank), s as usize % l.slots);
                if self.words[l.recv_seq(c, slot)].load(Ordering::Acquire) != s as u32 { continue; }
                let len = self.words[l.recv_slot(c, slot)].load(Ordering::Relaxed) as usize;
                if len == 0 || len > (l.slot_words - l.frame) * 4 || !len.is_multiple_of(4) { return Err(format!("tensor split relay: message {s} of rank {o} has a bad length")); }
                let wire = wires[side.index()].as_mut().unwrap();
                let pc = l.channel(o, wire.peer);
                if !self.free(wire, pc, slot, s, stop, timing)? { return Ok(()); }
                post(wire, l.recv_slot(c, slot)..l.recv_slot(c, slot) + l.frame + len / 4, l.recv_slot(pc, slot), l.recv_seq(c, slot), l.recv_seq(pc, slot), pc * l.slots + slot)?;
                onward[o] += 1;
                moved = true;
            }
            if moved {
                (active, pause) = (Instant::now(), Duration::from_micros(1));
            } else if active.elapsed() < timing.active {
                pace(&mut paced, timing.spin);
            } else {
                std::thread::sleep(pause);
                pause = (pause * 2).min(timing.idle_max);
            }
        }
        // A stop after a failure still reports it.
        if self.error() { Err(format!("mailbox failed before message {next}")) } else { Ok(()) }
    }
    /// Wait until `wire`'s previous write into the peer's channel `c` slot `slot` completed
    /// (false: `stop` came first).
    fn free(&self, wire: &mut Wire<'_>, c: usize, slot: usize, s: u64, stop: &AtomicBool, timing: Ring) -> Result<bool, String> {
        let (want, started, mut paced) = (wire.last[c * self.layout.slots + slot], Instant::now(), Instant::now());
        while wire.done < want {
            if stop.load(Ordering::Relaxed) { return Ok(false); }
            if self.error() { return Err(format!("mailbox failed before message {s}")); }
            if started.elapsed() > timing.reuse { return Err(format!("tensor split write before message {s} incomplete when it reuses its slot")); }
            pace(&mut paced, timing.spin);
            wire.reap()?;
        }
        Ok(true)
    }
}
/// Post one message's two writes on `wire` and record the write id for the slot `at`.
fn post(wire: &mut Wire<'_>, from: std::ops::Range<usize>, to: usize, flag: usize, flag_to: usize, at: usize) -> Result<(), String> {
    wire.next += 1;
    wire.last[at] = wire.next;
    wire.link.post(from, to, flag, flag_to, wire.next)
}

#[cfg(test)]
#[path = "tp_route_tests.rs"]
mod tests;
