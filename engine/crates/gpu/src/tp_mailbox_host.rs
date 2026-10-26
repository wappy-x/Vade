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
use std::sync::atomic::{AtomicU32, Ordering};

/// Slots per direction.
pub const TP_MAILBOX_SLOTS: usize = 4;
/// Mailbox layout in 32-bit words (see the module comment).
pub const HEADER: usize = 1024;
pub const SEND_SEQ: usize = 0;
pub const SEND_LEN: usize = 16;
pub const RECV_SEQ: usize = 32;
pub const ERROR: usize = 64;

/// The host side of a mailbox, owned by one proxy thread.
pub struct TpMailboxHost<'m> { mailbox: &'m [AtomicU32], slot_words: usize }
impl<'m> TpMailboxHost<'m> {
    /// The mailbox's words (header, then the send and receive slots of `slot_words` each).
    pub fn new(mailbox: &'m [AtomicU32], slot_words: usize) -> Self { Self { mailbox, slot_words } }
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
    /// Stop every waiting take (sticky).
    pub fn fail(&mut self) { self.mailbox[ERROR].store(1, Ordering::Release); }
    pub fn error(&self) -> bool { self.mailbox[ERROR].load(Ordering::Acquire) != 0 }
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
}
