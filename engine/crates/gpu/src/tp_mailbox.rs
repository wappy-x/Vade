//! VADE_W2_TP=1 in-graph mailbox, device side (the host endpoint is crate::tp_mailbox_host): mapped host slots that kernels fill and drain inside the
//! stream (tp_mailbox_put / tp_mailbox_take), and the one host endpoint a proxy thread uses
//! to move each slot over RDMA. The host side touches the mapping only as `[AtomicU32]`:
//! payload words with Relaxed loads/stores, each slot's sequence word with Release (publish)
//! and Acquire (consume), so no host access is a data race. Layout in 32-bit words: one
//! 1024-word header (send sequence words, send lengths, receive sequence words and the
//! error word on separate 64-byte lines), then SLOTS send slots, then SLOTS receive slots.
//! Message s (1, 2, ... per direction, counted in u64) uses slot s % SLOTS and publishes the
//! low 32 bits of s; a slot is reused only four messages later, and the split's lockstep keeps
//! at most three messages in flight per direction (two with VADE_W2_TP_PROJ unset), so a slot is
//! never rewritten while in use.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use crate::tp_mailbox_host::{Bell, TpMailboxHost, TP_MAILBOX_SLOTS, HEADER, SEND_SEQ, SEND_LEN, RECV_SEQ, ERROR, TAKE_PAUSE, WIDE, TAKE_CTAS};

/// The mapped mailbox of one rank (device-visible host memory), `slot_bytes` per slot.
/// `bell`: every queued put rings it, so a waiting write doorbell (VADE_W2_TP_DOORBELL=wait) can park.
pub struct TpMailbox<'a> { vector: DeviceVector<'a>, host: *mut f32, slot_words: usize, host_taken: AtomicBool, bell: Bell }
impl<'a> TpMailbox<'a> {
    fn words(&self) -> &[AtomicU32] {
        // SAFETY: the mapping is HEADER + 2 * SLOTS * slot_words aligned u32 words, zeroed by the
        // driver before this view first exists (Backend::tp_mailbox), that live as long as
        // `self`; host code reads and writes them only as atomics, and kernels access them only
        // through the protocol in the module comment.
        unsafe { std::slice::from_raw_parts(self.host.cast::<AtomicU32>(), self.vector.len) }
    }
    fn send_slot(&self, slot: usize) -> usize { HEADER + slot * self.slot_words }
    fn recv_slot(&self, slot: usize) -> usize { HEADER + (TP_MAILBOX_SLOTS + slot) * self.slot_words }
    fn device(&self, word: usize) -> u64 { self.vector.device_address() + (word * 4) as u64 }
    /// The proxy thread's endpoint: handed out once, so one thread owns the host side.
    pub fn take_host(&self) -> Option<TpMailboxHost<'_>> {
        (!self.host_taken.swap(true, Ordering::AcqRel)).then(|| TpMailboxHost::with_bell(self.words(), self.slot_words, &self.bell))
    }
    /// A take timed out or the proxy failed (sticky).
    pub fn failed(&self) -> bool { self.words()[ERROR].load(Ordering::Acquire) != 0 }
    pub fn slot_bytes(&self) -> usize { self.slot_words * 4 }
    /// 1 when the host asked for four-word copies (tp_mailbox_host::WIDE).
    fn wide(&self) -> u64 { u64::from(self.words()[WIDE].load(Ordering::Relaxed) == 1) }
}

impl<'a> Backend<'a> {
    /// A zeroed mailbox with slots of `slot_bytes` (a multiple of 4).
    pub fn tp_mailbox(&self, slot_bytes: usize) -> Result<TpMailbox<'a>, String> {
        if slot_bytes == 0 || !slot_bytes.is_multiple_of(4) { return Err("tensor split mailbox slot size".into()); }
        let words = (2 * TP_MAILBOX_SLOTS).checked_mul(slot_bytes / 4).and_then(|n| n.checked_add(HEADER)).ok_or("mailbox size overflow")?;
        // Every word (header and payload) is zeroed by the driver, and the stream drained, before
        // the host view exists or any mailbox kernel runs.
        let mut vector = self.mapped_vector(words)?;
        self.zero_vector(&mut vector)?;
        self.synchronize()?;
        let host = self.mapped_exchange_host(&vector)?;
        Ok(TpMailbox { vector, host, slot_words: slot_bytes / 4, host_taken: AtomicBool::new(false), bell: Bell::default() })
    }
    /// Queue tp_mailbox_put: words [own, own + half) of each of `rows` rows of `width` words of
    /// `source` go to the send slot of message `seq`, then its sequence word is published.
    pub fn tp_mailbox_put(&self, m: &TpMailbox<'_>, source: &DeviceVector<'_>, rows: usize, width: usize, half: usize,
        own: usize, seq: u64) -> Result<(), String> {
        self.check_mailbox_owner(m, source)?;
        check_mailbox_shape(m, source.len, rows, width, half, own)?;
        let slot = seq as usize % TP_MAILBOX_SLOTS;
        let mut args = [source.buffer.address, rows as u64, width as u64, half as u64, own as u64, m.device(m.send_slot(slot)),
            m.device(SEND_LEN + slot), m.device(SEND_SEQ + slot), (seq as u32) as u64 | m.wide() << 32];
        self.tp_mailbox_launch("tp_mailbox_put", 1, 1024, &mut args)?;
        m.bell.queued(seq);
        Ok(())
    }
    /// Queue tp_mailbox_take: wait (at most `limit_ns`, below 2^47) for message `seq` in its receive slot,
    /// then its rows go to words [peer, peer + half) of each `width`-word row of `target`.
    pub fn tp_mailbox_take(&self, m: &TpMailbox<'_>, target: &mut DeviceVector<'_>, rows: usize, width: usize, half: usize,
        peer: usize, seq: u64, limit_ns: u64) -> Result<(), String> {
        self.check_mailbox_owner(m, target)?;
        check_mailbox_shape(m, target.len, rows, width, half, peer)?;
        let slot = seq as usize % TP_MAILBOX_SLOTS;
        // The limit keeps 47 bits; bit 47 asks for four-word copies and the top 16 carry the host's
        // poll pause (0 = the kernel's 2048 ns).
        let limit = limit_ns.min((1 << 47) - 1) | m.wide() << 47 | u64::from(m.words()[TAKE_PAUSE].load(Ordering::Relaxed) & 0xffff) << 48;
        let mut args = [target.buffer.address, rows as u64, width as u64, half as u64, peer as u64, m.device(m.recv_slot(slot)),
            m.device(RECV_SEQ + slot), (seq as u32) as u64, m.device(ERROR), limit];
        let most = match m.words()[TAKE_CTAS].load(Ordering::Relaxed) { 0 => 96, n => n.min(96) as usize };
        self.tp_mailbox_launch("tp_mailbox_take", (rows * half).div_ceil(256).clamp(1, most) as u32, 256, &mut args)
    }
    /// The mailbox and the rows' vector were allocated in this Backend's context: a launch on
    /// another context could outlive the memory it reads or writes.
    fn check_mailbox_owner(&self, m: &TpMailbox<'_>, rows: &DeviceVector<'_>) -> Result<(), String> {
        self.check_vector(&m.vector, m.vector.len).and_then(|()| self.check_vector(rows, rows.len))
            .map_err(|_| "tensor split mailbox or rows belong to another backend".into())
    }
    /// Both mailbox kernels take only u64 parameters (pointers, counts, the sequence word).
    fn tp_mailbox_launch(&self, name: &str, blocks: u32, threads: u32, args: &mut [u64]) -> Result<(), String> {
        let mut pointers: Vec<*mut core::ffi::c_void> = args.iter_mut().map(|a| (a as *mut u64).cast()).collect();
        // SAFETY: tp_mailbox_put / tp_mailbox_take take exactly these u64 parameters in this
        // order; every address and extent was checked against its vector or the mailbox, and
        // both outlive the ordered stream (the mailbox lives with the model).
        unsafe { self.launch_flash(name, blocks, threads, &mut pointers) }
    }
}

/// A message is `rows * half` words of a `rows * width`-word vector, inside one slot.
fn check_mailbox_shape(m: &TpMailbox<'_>, len: usize, rows: usize, width: usize, half: usize, offset: usize) -> Result<(), String> {
    let words = rows.checked_mul(width).ok_or("mailbox shape overflow")?;
    if rows == 0 || half == 0 || offset.checked_add(half).is_none_or(|end| end > width) || words > len
        || rows.checked_mul(half).is_none_or(|n| n > m.slot_words) {
        return Err("tensor split mailbox shape outside the slot or the vector".into());
    }
    Ok(())
}
