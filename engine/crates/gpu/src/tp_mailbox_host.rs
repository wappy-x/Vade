//! VADE_W2_TP=1 in-graph mailbox, host side (std only; the mapped allocation and the
//! kernel launches are spark-gpu-probe driver::TpMailbox): mapped host slots that kernels fill and drain inside the
//! stream (tp_mailbox_put / tp_mailbox_take), and the one host endpoint a proxy thread uses
//! to move each slot over RDMA. The host side touches the mapping only as `[AtomicU32]`:
//! payload words with Relaxed loads/stores, each slot's sequence word with Release (publish)
//! and Acquire (consume), so no host access is a data race. Layout in 32-bit words: one
//! 1024-word header (send sequence words, send lengths, receive sequence words and the
//! error word on separate 64-byte lines), then SLOTS send slots, then SLOTS receive slots.
//! Message s (1, 2, ... per direction, counted in u64) uses slot s % SLOTS and publishes the
//! low 32 bits of s; a slot is reused only four messages later, and the split's lockstep keeps
//! at most two messages in flight per direction, so a slot is never rewritten while in use.
use std::{ops::Range, sync::{atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering}, OnceLock}, thread::Thread, time::{Duration, Instant}};

/// Slots per direction.
pub const TP_MAILBOX_SLOTS: usize = 4;
/// Mailbox layout in 32-bit words (see the module comment).
pub const HEADER: usize = 1024;
pub const SEND_SEQ: usize = 0;
pub const SEND_LEN: usize = 16;
pub const RECV_SEQ: usize = 32;
pub const ERROR: usize = 64;
/// VADE_W2_TP_WIRE=write: host-owned staging words the NIC reads (never written by a kernel):
/// FLAG_OUT + slot holds the sequence word a message's second write puts in the peer's RECV_SEQ,
/// FAIL_OUT the 1 a failing rank writes into the peer's ERROR word.
pub const FLAG_OUT: usize = 96;
pub const FAIL_OUT: usize = 112;
/// The longest nanosleep (ns) between a take's polls, written by the host before any take is
/// queued (0, as zeroed: 2048); the launch passes it in the high 16 bits of the take's limit.
pub const TAKE_PAUSE: usize = 128;
/// 1: puts and takes copy four words at a time where the shape allows (host-written like TAKE_PAUSE;
/// 0, as zeroed: one word at a time); launches pass it in bit 32 of the put's sequence and bit 47
/// of the take's limit.
pub const WIDE: usize = 129;
/// The most CTAs a take launches (host-written like TAKE_PAUSE; 0, as zeroed: 96). Each CTA's
/// first thread polls the sequence word at system scope, and those polls cost about 0.3 us per
/// CTA on GB10 (tp-gather-bench, 2026-10-26), so fewer CTAs that copy more each wait less.
pub const TAKE_CTAS: usize = 130;

/// VADE_W2_TP_DOORBELL=wait: the host's count of queued puts, which lets the write doorbell block
/// (thread park, a futex on Linux) while every queued message is posted, instead of polling. The
/// GPU cannot wake a host thread, but the host queues each put before the GPU publishes it.
#[derive(Default)]
pub struct Bell { queued: AtomicU64, parked: AtomicBool, thread: OnceLock<Thread>, parks: AtomicU64, parked_ns: AtomicU64 }
impl Bell {
    /// The host queued the put of message `seq`; a parked doorbell wakes. A no-op until a waiting
    /// doorbell registered (the spin doorbell and the proxy never park).
    pub fn queued(&self, seq: u64) {
        if let Some(thread) = self.thread.get() {
            self.queued.fetch_max(seq, Ordering::SeqCst);
            if self.parked.load(Ordering::SeqCst) { thread.unpark(); }
        }
    }
    /// The calling thread is the doorbell from now on (once).
    fn register(&self) -> Result<(), String> {
        self.thread.set(std::thread::current()).map_err(|_| "tensor split doorbell already registered".to_string())
    }
    /// Park for at most `most` unless message `next` is queued (or a put races in: `queued` then
    /// sees `parked` and unparks, and a park after an unpark returns at once).
    fn rest(&self, next: u64, most: Duration) {
        self.parked.store(true, Ordering::SeqCst);
        if self.queued.load(Ordering::SeqCst) < next {
            let at = Instant::now();
            std::thread::park_timeout(most);
            self.parks.fetch_add(1, Ordering::Relaxed);
            self.parked_ns.fetch_add(at.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        self.parked.store(false, Ordering::SeqCst);
    }
    /// How often the doorbell parked and for how long in all (ns).
    pub fn stats(&self) -> (u64, u64) { (self.parks.load(Ordering::Relaxed), self.parked_ns.load(Ordering::Relaxed)) }
    fn in_flight(&self, next: u64) -> bool { self.queued.load(Ordering::SeqCst) >= next }
}

/// The host side of a mailbox, owned by one proxy thread.
pub struct TpMailboxHost<'m> { mailbox: &'m [AtomicU32], slot_words: usize, bell: Option<&'m Bell> }
impl<'m> TpMailboxHost<'m> {
    /// The mailbox's words (header, then the send and receive slots of `slot_words` each).
    pub fn new(mailbox: &'m [AtomicU32], slot_words: usize) -> Self { Self { mailbox, slot_words, bell: None } }
    /// The same, with the bell the mailbox's put launches ring (a waiting doorbell needs it).
    pub fn with_bell(mailbox: &'m [AtomicU32], slot_words: usize, bell: &'m Bell) -> Self { Self { mailbox, slot_words, bell: Some(bell) } }
    /// Low sequence word the GPU last published in send slot `seq % SLOTS`.
    pub fn sent(&mut self, seq: u64) -> u32 { self.mailbox[SEND_SEQ + seq as usize % TP_MAILBOX_SLOTS].load(Ordering::Acquire) }
    /// Copy message `seq` (1, 2, ...) out of its send slot: exactly the byte length the GPU
    /// published with it, and only once its sequence word is published. An overwrite during
    /// the copy (the sequence word moved on) is an error.
    pub fn read_send(&mut self, seq: u64, out: &mut Vec<u8>) -> Result<(), String> {
        let slot = seq as usize % TP_MAILBOX_SLOTS;
        if seq == 0 || self.mailbox[SEND_SEQ + slot].load(Ordering::Acquire) != seq as u32 { return Err("tensor split message not published".into()); }
        let len = self.mailbox[SEND_LEN + slot].load(Ordering::Relaxed) as usize;
        if len > self.slot_words * 4 || !len.is_multiple_of(4) { return Err("tensor split mailbox message length".into()); }
        let base = HEADER + slot * self.slot_words;
        out.clear();
        out.extend(self.mailbox[base..base + len / 4].iter().flat_map(|w| w.load(Ordering::Relaxed).to_le_bytes()));
        if self.mailbox[SEND_SEQ + slot].load(Ordering::Acquire) != seq as u32 { return Err("tensor split send slot overwritten during its copy".into()); }
        Ok(())
    }
    /// Write message `seq` into receive slot `seq % SLOTS`, then publish its sequence word.
    pub fn deliver(&mut self, seq: u64, bytes: &[u8]) -> Result<(), String> {
        if seq == 0 || bytes.len() > self.slot_words * 4 || !bytes.len().is_multiple_of(4) { return Err("tensor split mailbox message length or sequence".into()); }
        let slot = seq as usize % TP_MAILBOX_SLOTS;
        let base = HEADER + (TP_MAILBOX_SLOTS + slot) * self.slot_words;
        for (w, c) in self.mailbox[base..].iter().zip(bytes.as_chunks::<4>().0) { w.store(u32::from_le_bytes(*c), Ordering::Relaxed); }
        self.mailbox[RECV_SEQ + slot].store(seq as u32, Ordering::Release);
        Ok(())
    }
    /// The put launches' bell, when this view has it.
    pub fn bell(&self) -> Option<&'m Bell> { self.bell }
    /// The mailbox's words, for the one-sided wire that registers them (VADE_W2_TP_WIRE=write).
    pub fn words(&self) -> &'m [AtomicU32] { self.mailbox }
    /// Send slot `slot`'s first word and receive slot `slot`'s first word.
    pub fn send_slot(&self, slot: usize) -> usize { HEADER + slot * self.slot_words }
    pub fn recv_slot(&self, slot: usize) -> usize { HEADER + (TP_MAILBOX_SLOTS + slot) * self.slot_words }
    /// The byte length the GPU published with message `seq` (published: its sequence word is
    /// in place), checked against the slot.
    pub fn send_len(&mut self, seq: u64) -> Result<usize, String> {
        let slot = seq as usize % TP_MAILBOX_SLOTS;
        if seq == 0 || self.mailbox[SEND_SEQ + slot].load(Ordering::Acquire) != seq as u32 { return Err("tensor split message not published".into()); }
        let len = self.mailbox[SEND_LEN + slot].load(Ordering::Relaxed) as usize;
        if len == 0 || len > self.slot_words * 4 || !len.is_multiple_of(4) { return Err("tensor split mailbox message length".into()); }
        Ok(len)
    }
    /// Stage message `seq`'s sequence word for the NIC to write into the peer's RECV_SEQ.
    pub fn stage_flag(&mut self, seq: u64) { self.mailbox[FLAG_OUT + seq as usize % TP_MAILBOX_SLOTS].store(seq as u32, Ordering::Release); }
    /// The takes' longest poll pause from now on (ns, 1..=65535; queued takes keep theirs).
    pub fn set_take_pause(&mut self, ns: u16) { self.mailbox[TAKE_PAUSE].store(ns.into(), Ordering::Relaxed); }
    /// Four-word copies from now on (queued kernels keep theirs).
    pub fn set_wide(&mut self, wide: bool) { self.mailbox[WIDE].store(wide.into(), Ordering::Relaxed); }
    /// The takes' CTA cap from now on (1..=96; queued takes keep theirs).
    pub fn set_take_ctas(&mut self, ctas: u32) { self.mailbox[TAKE_CTAS].store(ctas.clamp(1, 96), Ordering::Relaxed); }
    /// Stage the 1 the NIC writes into the peer's ERROR word.
    pub fn stage_fail(&mut self) { self.mailbox[FAIL_OUT].store(1, Ordering::Release); }
    /// Stop every waiting take (sticky).
    pub fn fail(&mut self) { self.mailbox[ERROR].store(1, Ordering::Release); }
    pub fn error(&self) -> bool { self.mailbox[ERROR].load(Ordering::Acquire) != 0 }
}

/// VADE_W2_TP_WIRE=write: what the doorbell needs from its queue pair (crate::tp_rdma::TpRdma).
pub trait TpDoorbell {
    /// Words `from` to the peer's words from `to`, then staging word `flag` to the peer's word
    /// `flag_to`, in that order; the second write completes with id `id`.
    fn post(&mut self, from: Range<usize>, to: usize, flag: usize, flag_to: usize, id: u64) -> Result<(), String>;
    /// The ids completed since the last call, in posting order, without waiting.
    fn reap(&mut self, done: &mut dyn FnMut(u64)) -> Result<(), String>;
    /// Staging word `from` (holding 1) into the peer's word `to`; completes with id u64::MAX.
    fn notify(&mut self, from: usize, to: usize) -> Result<(), String>;
}
/// The doorbell's timing: up to `active` after the last message it polls, spinning at most `spin`
/// at a time and then yielding the core (yield_now: any runnable thread goes first), then sleeps
/// with a doubling backoff up to `idle_max`; a slot's previous message must complete within
/// `reuse` (RC retries included), and after a failure the ERROR write to the peer gets `notify`.
/// `rest` (VADE_W2_TP_DOORBELL=wait; None: poll as above): while no queued message is waiting to be
/// posted and every posted one completed, block on the mailbox's Bell for at most `rest` at a time
/// instead; the `active` window and the backoff then cover only queued messages the GPU has not
/// published yet.
#[derive(Clone, Copy)]
pub struct Ring { pub active: Duration, pub spin: Duration, pub idle_max: Duration, pub reuse: Duration, pub notify: Duration, pub rest: Option<Duration> }
impl Default for Ring {
    fn default() -> Self {
        Self { active: Duration::from_millis(20), spin: Duration::from_micros(1000), idle_max: Duration::from_micros(100), reuse: Duration::from_secs(2), notify: Duration::from_millis(100), rest: None }
    }
}
/// Spin until `spin` has passed since `since`, then yield the core once and restart the count.
fn pace(since: &mut Instant, spin: Duration) {
    if since.elapsed() < spin { std::hint::spin_loop(); } else { std::thread::yield_now(); *since = Instant::now(); }
}
impl TpMailboxHost<'_> {
    /// The write doorbell, until `stop` or an error: when the GPU publishes message s, post its
    /// published bytes from send slot s % SLOTS into the peer's receive slot s % SLOTS, then its
    /// staged sequence word into the peer's RECV_SEQ. The slot's previous message (s - SLOTS) must
    /// have completed first; the lockstep already implies the peer placed it, so this waits only
    /// for its acknowledgement. Completions arrive once each, in order.
    /// With `Ring::rest` it parks on the mailbox's Bell while nothing is in flight (every queued
    /// message posted and completed); a put launch or the caller's unpark (after `stop`) wakes it.
    pub fn ring(&mut self, link: &mut dyn TpDoorbell, stop: &AtomicBool, timing: Ring) -> Result<(), String> {
        let (mut next, mut done, slots) = (1u64, 0u64, TP_MAILBOX_SLOTS as u64);
        let (mut active, mut pause, mut paced) = (Instant::now(), Duration::from_micros(1), Instant::now());
        let bell = match timing.rest {
            Some(_) => { let bell = self.bell.ok_or("the waiting tensor split doorbell needs the mailbox's bell")?; bell.register()?; Some(bell) }
            None => None,
        };
        while !stop.load(Ordering::Relaxed) {
            reap(link, &mut done)?;
            if self.sent(next) == next as u32 {
                if self.error() {
                    return Err(format!("mailbox failed before message {next}"));
                }
                let len = self.send_len(next)?;
                let started = Instant::now();
                while done + slots < next {
                    if stop.load(Ordering::Relaxed) { return Ok(()); }
                    if self.error() { return Err(format!("mailbox failed before message {next}")); }
                    if started.elapsed() > timing.reuse {
                        return Err(format!("tensor split message {} incomplete when message {next} reuses its slot", next - slots));
                    }
                    pace(&mut paced, timing.spin);
                    reap(link, &mut done)?;
                }
                let slot = next as usize % TP_MAILBOX_SLOTS;
                let from = self.send_slot(slot);
                self.stage_flag(next);
                link.post(from..from + len / 4, self.recv_slot(slot), FLAG_OUT + slot, RECV_SEQ + slot, next)?;
                next += 1;
                (active, pause) = (Instant::now(), Duration::from_micros(1));
                continue;
            }
            if self.error() {
                return Err(format!("mailbox failed before message {next}"));
            }
            if let (Some(bell), Some(most)) = (bell, timing.rest) {
                if done + 1 >= next && !bell.in_flight(next) {
                    bell.rest(next, most);
                    (active, pause) = (Instant::now(), Duration::from_micros(1));
                    continue;
                }
            }
            if active.elapsed() < timing.active {
                pace(&mut paced, timing.spin);
            } else {
                std::thread::sleep(pause);
                pause = (pause * 2).min(timing.idle_max);
            }
        }
        Ok(())
    }
    /// After a doorbell error: fail this mailbox, then (best effort, within `Ring::notify`) write 1
    /// into the peer's ERROR word, which its takes poll; their timeout stays the backstop.
    pub fn fail_peer(&mut self, link: &mut dyn TpDoorbell) {
        self.fail();
        self.stage_fail();
        if link.notify(FAIL_OUT, ERROR).is_ok() {
            let (until, mut out) = (Instant::now() + Ring::default().notify, false);
            while !out && Instant::now() < until && link.reap(&mut |id| out |= id == u64::MAX).is_ok() { std::thread::yield_now(); }
        }
    }
}
/// Reap completions into `done`, the last completed message (each completes once, in order).
fn reap(link: &mut dyn TpDoorbell, done: &mut u64) -> Result<(), String> {
    let mut order = true;
    link.reap(&mut |id| { order &= id == *done + 1; *done = id; })?;
    if order { Ok(()) } else { Err(format!("tensor split write completion out of order (after {done})")) }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn view(memory: &[AtomicU32], slot_words: usize) -> TpMailboxHost<'_> { TpMailboxHost::new(memory, slot_words) }
    /// Wait (at most 10 s, then fail the test instead of hanging) until `ready`.
    fn until(mut ready: impl FnMut() -> bool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !ready() { assert!(std::time::Instant::now() < deadline, "mailbox test timed out"); std::thread::yield_now(); }
    }
    #[test]
    fn mailbox_header_words_sit_on_separate_lines() {
        assert!(SEND_LEN >= SEND_SEQ + TP_MAILBOX_SLOTS && RECV_SEQ - SEND_LEN >= 16 && ERROR - RECV_SEQ >= 16 && ERROR < HEADER);
        // The write wire's staging words: own lines, clear of the put's counters (words 80..84).
        assert!(FLAG_OUT - ERROR >= 32 && FLAG_OUT + TP_MAILBOX_SLOTS <= FAIL_OUT && FAIL_OUT + 16 <= TAKE_PAUSE && WIDE == TAKE_PAUSE + 1 && TAKE_CTAS == WIDE + 1 && TAKE_CTAS < HEADER);
    }
    /// One thread plays the GPU (fills send slots, publishes words, takes deliveries), the other
    /// is the endpoint (copies published messages out, delivers replies): every message arrives
    /// whole and in order, sequence words wrap at 32 bits without a false match, and an
    /// overwrite during a copy is reported instead of returning mixed rows.
    #[test]
    fn two_threads_move_whole_messages_in_order() {
        let slot_words = 64;
        let memory: Vec<AtomicU32> = (0..HEADER + 2 * TP_MAILBOX_SLOTS * slot_words).map(|_| AtomicU32::new(0)).collect();
        let first = u32::MAX as u64 - 40;
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let (mut out, mut host) = (Vec::new(), view(&memory, slot_words));
                for seq in first..first + 200 {
                    until(|| host.sent(seq) == seq as u32);
                    host.read_send(seq, &mut out).unwrap();
                    let words: Vec<u32> = out.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
                    assert!(words.iter().all(|&w| w == seq as u32), "message {seq} arrived whole");
                    host.deliver(seq, &out).unwrap();
                }
            });
            for seq in first..first + 200 {
                let slot = seq as usize % TP_MAILBOX_SLOTS;
                let base = HEADER + slot * slot_words;
                // At most one message in flight: wait for the previous reply (the lockstep).
                if seq > first { until(|| memory[RECV_SEQ + (seq as usize - 1) % TP_MAILBOX_SLOTS].load(Ordering::Acquire) == (seq - 1) as u32); }
                for w in &memory[base..base + slot_words] { w.store(seq as u32, Ordering::Relaxed); }
                memory[SEND_LEN + slot].store((slot_words * 4) as u32, Ordering::Relaxed);
                memory[SEND_SEQ + slot].store(seq as u32, Ordering::Release);
            }
            let last = first + 199;
            until(|| memory[RECV_SEQ + last as usize % TP_MAILBOX_SLOTS].load(Ordering::Acquire) == last as u32);
        });
        let mut host = view(&memory, slot_words);
        let mut out = Vec::new();
        assert!(host.read_send(first + 300, &mut out).is_err(), "unpublished message");
        assert!(host.deliver(1, &vec![0; slot_words * 4 + 4]).is_err());
        assert!(host.deliver(0, &[0; 4]).is_err(), "sequence 0 is never a message");
        assert!(!host.error());
        host.fail();
        assert!(host.error());
    }
    /// The endpoint returns only the published extent of a published message: a zeroed mailbox
    /// (sequence words 0) has no message 0, a 4-byte message in an 8-byte slot reads 4 bytes,
    /// and a published length beyond the slot is refused.
    #[test]
    fn reads_only_the_published_extent_of_published_messages() {
        let slot_words = 2;
        let memory: Vec<AtomicU32> = (0..HEADER + 2 * TP_MAILBOX_SLOTS * slot_words).map(|_| AtomicU32::new(0)).collect();
        let mut out = Vec::new();
        assert!(view(&memory, slot_words).read_send(0, &mut out).is_err());
        let base = HEADER + slot_words;
        memory[base].store(7, Ordering::Relaxed);
        memory[base + 1].store(9, Ordering::Relaxed);
        memory[SEND_LEN + 1].store(4, Ordering::Relaxed);
        memory[SEND_SEQ + 1].store(1, Ordering::Release);
        view(&memory, slot_words).read_send(1, &mut out).unwrap();
        assert_eq!(out, 7u32.to_le_bytes());
        memory[SEND_LEN + 1].store(12, Ordering::Relaxed);
        assert!(view(&memory, slot_words).read_send(1, &mut out).is_err());
    }
    /// A fake NIC for one rank: posted writes are carried out (reading the local words then)
    /// only some reaps later, and complete still later, both by a fixed pseudo-random delay.
    /// Each payload read checks that the send slot still holds the message the post named (no
    /// overwrite while the NIC reads it).
    struct Nic<'m> { local: &'m [AtomicU32], peer: &'m [AtomicU32], tag: fn(u64) -> u32, queue: std::collections::VecDeque<(Range<usize>, usize, usize, usize, u64, u32)>,
        flying: std::collections::VecDeque<(u64, u32)>, seed: u32, posts: Vec<u64>, torn: &'m AtomicBool, fail_post: bool }
    impl Nic<'_> {
        fn delay(&mut self, most: u32) -> u32 { self.seed ^= self.seed << 13; self.seed ^= self.seed >> 17; self.seed ^= self.seed << 5; self.seed % most }
    }
    impl TpDoorbell for Nic<'_> {
        fn post(&mut self, from: Range<usize>, to: usize, flag: usize, flag_to: usize, id: u64) -> Result<(), String> {
            if self.fail_post { return Err("post refused".into()); }
            let wait = self.delay(40);
            self.posts.push(id);
            self.queue.push_back((from, to, flag, flag_to, id, wait));
            Ok(())
        }
        fn reap(&mut self, done: &mut dyn FnMut(u64)) -> Result<(), String> {
            if let Some(head) = self.queue.front_mut() { head.5 = head.5.saturating_sub(1); }
            while self.queue.front().is_some_and(|q| q.5 == 0) {
                let (from, to, flag, flag_to, id, _) = self.queue.pop_front().unwrap();
                if id != u64::MAX && self.local[from.clone()].iter().any(|w| w.load(Ordering::Relaxed) != (self.tag)(id)) { self.torn.store(true, Ordering::Relaxed); }
                for (i, w) in from.enumerate() { self.peer[to + i].store(self.local[w].load(Ordering::Relaxed), Ordering::Relaxed); }
                self.peer[flag_to].store(self.local[flag].load(Ordering::Relaxed), Ordering::Release);
                let late = self.delay(200);
                self.flying.push_back((id, late));
            }
            if let Some(head) = self.flying.front_mut() { head.1 = head.1.saturating_sub(1); }
            while self.flying.front().is_some_and(|f| f.1 == 0) { done(self.flying.pop_front().unwrap().0); }
            Ok(())
        }
        fn notify(&mut self, from: usize, to: usize) -> Result<(), String> { self.post(from..from + 1, to, from, to, u64::MAX) }
    }
    /// The doorbell as the write wire's thread runs it: ring, and on an error fail both ranks.
    fn run_write_within(host: TpMailboxHost<'_>, link: impl TpDoorbell, stop: &AtomicBool, reuse: Duration) -> Result<(), String> {
        run_write_with(host, link, stop, Ring { reuse, ..Ring::default() })
    }
    fn run_write_with(mut host: TpMailboxHost<'_>, mut link: impl TpDoorbell, stop: &AtomicBool, timing: Ring) -> Result<(), String> {
        let result = host.ring(&mut link, stop, timing);
        if result.is_err() { host.fail_peer(&mut link); }
        result
    }
    fn run_write(host: TpMailboxHost<'_>, link: impl TpDoorbell, stop: &AtomicBool) -> Result<(), String> {
        run_write_within(host, link, stop, Ring::default().reuse)
    }
    fn mailbox(slot_words: usize) -> Vec<AtomicU32> {
        (0..HEADER + 2 * TP_MAILBOX_SLOTS * slot_words).map(|_| AtomicU32::new(0)).collect()
    }
    /// One rank's GPU: put fills send slot seq % SLOTS with tag(seq) and publishes; take waits for
    /// RECV_SEQ and checks the peer's whole message (no torn or stale words).
    /// `bell`: the host queues each put (rings the bell) just before the GPU publishes it.
    struct Gpu<'m> { m: &'m [AtomicU32], slot_words: usize, sent: u64, taken: u64, tag: fn(u64) -> u32, peer_tag: fn(u64) -> u32, bell: Option<&'m Bell> }
    impl Gpu<'_> {
        fn put(&mut self, words: usize) {
            self.sent += 1;
            if let Some(bell) = self.bell { bell.queued(self.sent); }
            let slot = self.sent as usize % TP_MAILBOX_SLOTS;
            for w in &self.m[HEADER + slot * self.slot_words..][..words] { w.store((self.tag)(self.sent), Ordering::Relaxed); }
            self.m[SEND_LEN + slot].store((words * 4) as u32, Ordering::Relaxed);
            self.m[SEND_SEQ + slot].store(self.sent as u32, Ordering::Release);
        }
        fn take(&mut self, words: usize) -> Result<(), String> {
            self.taken += 1;
            let slot = self.taken as usize % TP_MAILBOX_SLOTS;
            let deadline = Instant::now() + Duration::from_secs(10);
            while self.m[RECV_SEQ + slot].load(Ordering::Acquire) != self.taken as u32 {
                if self.m[ERROR].load(Ordering::Acquire) != 0 { return Err(format!("error before take {}", self.taken)); }
                if Instant::now() > deadline { return Err(format!("take {} timed out", self.taken)); }
                std::thread::yield_now();
            }
            let base = HEADER + (TP_MAILBOX_SLOTS + slot) * self.slot_words;
            if self.m[base..base + words].iter().any(|w| w.load(Ordering::Relaxed) != (self.peer_tag)(self.taken)) { return Err(format!("take {} torn", self.taken)); }
            Ok(())
        }
    }
    fn tag0(seq: u64) -> u32 { seq as u32 }
    fn tag1(seq: u64) -> u32 { seq as u32 | 1 << 31 }
    /// Both ranks' doorbells over fake NICs with late placements and later completions, on the
    /// split's call orders: per pass rank 0's residual post (one way), then layers of halves
    /// (put then take on both ranks) at changing sizes. Every message arrives whole, the NIC never
    /// reads a slot the GPU is rewriting, slots wrap many times, and both doorbells stop cleanly.
    #[test]
    fn write_doorbells_keep_slots_whole_under_late_completions() {
        for wait in [false, true] { doorbells_keep_slots_whole(wait); }
    }
    fn doorbells_keep_slots_whole(wait: bool) {
        let slot_words = 64;
        let (b0, b1) = (Bell::default(), Bell::default());
        let (bell0, bell1) = (wait.then_some(&b0), wait.then_some(&b1));
        let timing = Ring { rest: wait.then_some(Duration::from_secs(10)), ..Ring::default() };
        let (m0, m1) = (mailbox(slot_words), mailbox(slot_words));
        let (stop, torn) = (AtomicBool::new(false), AtomicBool::new(false));
        struct Stop<'s>(&'s AtomicBool);
        impl Drop for Stop<'_> { fn drop(&mut self) { self.0.store(true, Ordering::Relaxed); } }
        let (passes, layers) = (12, 9);
        let nic = |local, peer, tag, seed| Nic { local, peer, tag, queue: Default::default(), flying: Default::default(), seed, posts: Vec::new(), torn: &torn, fail_post: false };
        std::thread::scope(|scope| {
            let _stop = Stop(&stop);
            let d0 = scope.spawn(|| run_write_with(TpMailboxHost::with_bell(&m0, slot_words, &b0), nic(&m0, &m1, tag0, 7), &stop, timing));
            let d1 = scope.spawn(|| run_write_with(TpMailboxHost::with_bell(&m1, slot_words, &b1), nic(&m1, &m0, tag1, 11), &stop, timing));
            // Both bells registered (or none) before any put rings them.
            until(|| !wait || b0.thread.get().is_some() && b1.thread.get().is_some());
            let r1 = scope.spawn(|| -> Result<(), String> {
                let mut g = Gpu { m: &m1, slot_words, sent: 0, taken: 0, tag: tag1, peer_tag: tag0, bell: bell1 };
                for _ in 0..passes { g.take(slot_words)?; for l in 0..layers { g.put(1 + l * 7); g.take(1 + l * 7)?; } }
                Ok(())
            });
            let mut g = Gpu { m: &m0, slot_words, sent: 0, taken: 0, tag: tag0, peer_tag: tag1, bell: bell0 };
            for _ in 0..passes { g.put(slot_words); for l in 0..layers { g.put(1 + l * 7); g.take(1 + l * 7).unwrap(); } }
            r1.join().unwrap().unwrap();
            stop.store(true, Ordering::Relaxed);
            // A parked doorbell sees stop only once woken (the service unparks it the same way).
            for (d, bell) in [(d0, &b0), (d1, &b1)] {
                if let Some(t) = bell.thread.get() { t.unpark(); }
                d.join().unwrap().unwrap();
            }
            assert_eq!(g.sent, (passes * (layers + 1)) as u64);
        });
        assert!(!torn.load(Ordering::Relaxed), "a NIC read a slot while its GPU rewrote it");
        assert!(!TpMailboxHost::new(&m0, slot_words).error() && !TpMailboxHost::new(&m1, slot_words).error());
    }
    /// VADE_W2_TP_DOORBELL=wait: with nothing queued the doorbell parks (no polling), a queued put
    /// wakes it and its message is posted, it parks again once that write completed, and after stop
    /// an unpark ends it at once (its own recheck is 60 s away here).
    #[test]
    fn waiting_doorbell_parks_until_a_put_is_queued_and_stops_on_unpark() {
        let slot_words = 4;
        let (m0, m1, bell, stop, torn) = (mailbox(slot_words), mailbox(slot_words), Bell::default(), AtomicBool::new(false), AtomicBool::new(false));
        let nic = Nic { local: &m0, peer: &m1, tag: tag0, queue: Default::default(), flying: Default::default(), seed: 9, posts: Vec::new(), torn: &torn, fail_post: false };
        let timing = Ring { rest: Some(Duration::from_secs(60)), ..Ring::default() };
        std::thread::scope(|scope| {
            struct Stop<'s>(&'s AtomicBool, &'s Bell);
            impl Drop for Stop<'_> { fn drop(&mut self) { self.0.store(true, Ordering::Relaxed); if let Some(t) = self.1.thread.get() { t.unpark(); } } }
            let guard = Stop(&stop, &bell);
            let doorbell = scope.spawn(|| run_write_with(TpMailboxHost::with_bell(&m0, slot_words, &bell), nic, &stop, timing));
            until(|| bell.stats().0 == 0 && bell.parked.load(Ordering::SeqCst));
            let mut g = Gpu { m: &m0, slot_words, sent: 0, taken: 0, tag: tag0, peer_tag: tag1, bell: Some(&bell) };
            for seq in 1..=6u64 {
                // Parked before each put (all earlier writes completed), then woken by it.
                until(|| bell.parked.load(Ordering::SeqCst) && bell.stats().0 + 1 >= seq);
                g.put(slot_words);
                until(|| m1[RECV_SEQ + seq as usize % TP_MAILBOX_SLOTS].load(Ordering::Acquire) == seq as u32);
                assert!(m1[HEADER + (TP_MAILBOX_SLOTS + seq as usize % TP_MAILBOX_SLOTS) * slot_words..][..slot_words].iter().all(|w| w.load(Ordering::Relaxed) == seq as u32));
            }
            until(|| bell.parked.load(Ordering::SeqCst));
            let at = Instant::now();
            drop(guard);
            doorbell.join().unwrap().unwrap();
            assert!(at.elapsed() < Duration::from_secs(5), "stop ended the parked doorbell");
        });
        assert!(bell.stats().0 >= 6, "parked between messages: {:?}", bell.stats());
        assert!(!torn.load(Ordering::Relaxed));
        // A waiting doorbell needs a bell, and a bell has one doorbell.
        let m = mailbox(slot_words);
        let nic = Nic { local: &m, peer: &m1, tag: tag0, queue: Default::default(), flying: Default::default(), seed: 1, posts: Vec::new(), torn: &torn, fail_post: false };
        assert!(TpMailboxHost::new(&m, slot_words).ring(&mut { nic }, &AtomicBool::new(true), timing).is_err());
        let nic = Nic { local: &m, peer: &m1, tag: tag0, queue: Default::default(), flying: Default::default(), seed: 1, posts: Vec::new(), torn: &torn, fail_post: false };
        assert!(TpMailboxHost::with_bell(&m, slot_words, &bell).ring(&mut { nic }, &AtomicBool::new(true), timing).unwrap_err().contains("already registered"));
    }
    /// Traffic that breaks the lockstep (five puts with no take): message 5 reuses message 1's
    /// slot before 1 completed, so the doorbell refuses to post it, fails its mailbox and writes
    /// 1 into the peer's ERROR word. A refused post fails the same way.
    #[test]
    fn write_doorbell_refuses_reuse_before_completion_and_fails_both_ranks() {
        let slot_words = 8;
        let torn = AtomicBool::new(false);
        for refuse in [false, true] {
            let (m0, m1) = (mailbox(slot_words), mailbox(slot_words));
            let stop = AtomicBool::new(false);
            let mut nic = Nic { local: &m0, peer: &m1, tag: tag0, queue: Default::default(), flying: Default::default(), seed: 3, posts: Vec::new(), torn: &torn, fail_post: refuse };
            // Writes are placed but never complete (the fake's completion delay is 2^32 reaps away).
            struct Never<'n, 'm>(&'n mut Nic<'m>);
            impl TpDoorbell for Never<'_, '_> {
                fn post(&mut self, from: Range<usize>, to: usize, flag: usize, flag_to: usize, id: u64) -> Result<(), String> { self.0.post(from, to, flag, flag_to, id) }
                fn reap(&mut self, _: &mut dyn FnMut(u64)) -> Result<(), String> { self.0.reap(&mut |_| ())?; self.0.flying.iter_mut().for_each(|f| f.1 = u32::MAX); Ok(()) }
                fn notify(&mut self, from: usize, to: usize) -> Result<(), String> { self.0.notify(from, to)?; for q in &mut self.0.queue { q.5 = 1; } self.0.reap(&mut |_| ()) }
            }
            let mut g = Gpu { m: &m0, slot_words, sent: 0, taken: 0, tag: tag0, peer_tag: tag1, bell: None };
            for _ in 0..4 { g.put(slot_words); }
            // Message 5 is published once the doorbell staged message 4 (or failed).
            let result = std::thread::scope(|scope| {
                // A failed wait below stops the doorbell (no hung test).
                struct Stop<'s>(&'s AtomicBool);
                impl Drop for Stop<'_> { fn drop(&mut self) { self.0.store(true, Ordering::Relaxed); } }
                let (halt, words) = (&stop, &m0);
                scope.spawn(move || {
                    let guard = Stop(halt);
                    until(|| words[FLAG_OUT].load(Ordering::Acquire) == 4 || words[ERROR].load(Ordering::Acquire) != 0);
                    std::mem::forget(guard);
                    g.put(slot_words);
                });
                run_write_within(TpMailboxHost::new(&m0, slot_words), Never(&mut nic), &stop, Duration::from_millis(50))
            });
            let error = result.unwrap_err();
            assert!(error.contains(if refuse { "post refused" } else { "message 1 incomplete when message 5 reuses its slot" }), "{error}");
            assert_eq!(nic.posts, if refuse { vec![] } else { vec![1, 2, 3, 4, u64::MAX] });
            assert!(TpMailboxHost::new(&m0, slot_words).error());
            assert_eq!(m1[ERROR].load(Ordering::Acquire), u32::from(!refuse), "the peer's ERROR word");
        }
    }
    /// A completion out of order (or twice) is a failure, and so is a local take's timeout
    /// (ERROR set by the GPU) while the doorbell waits for the next message.
    #[test]
    fn write_doorbell_fails_on_disorder_and_on_local_error() {
        let slot_words = 4;
        struct Twice;
        impl TpDoorbell for Twice {
            fn post(&mut self, _: Range<usize>, _: usize, _: usize, _: usize, _: u64) -> Result<(), String> { Ok(()) }
            fn reap(&mut self, done: &mut dyn FnMut(u64)) -> Result<(), String> { done(1); done(1); Ok(()) }
            fn notify(&mut self, _: usize, _: usize) -> Result<(), String> { Ok(()) }
        }
        let m = mailbox(slot_words);
        let error = run_write(TpMailboxHost::new(&m, slot_words), Twice, &AtomicBool::new(false)).unwrap_err();
        assert!(error.contains("out of order"), "{error}");
        assert_eq!(m[FAIL_OUT].load(Ordering::Acquire), 1);
        let m = mailbox(slot_words);
        m[ERROR].store(1, Ordering::Release);
        let torn = AtomicBool::new(false);
        let nic = Nic { local: &m, peer: &m, tag: tag0, queue: Default::default(), flying: Default::default(), seed: 5, posts: Vec::new(), torn: &torn, fail_post: false };
        let error = run_write(TpMailboxHost::new(&m, slot_words), nic, &AtomicBool::new(false)).unwrap_err();
        assert!(error.contains("mailbox failed before message 1"), "{error}");
    }

}
