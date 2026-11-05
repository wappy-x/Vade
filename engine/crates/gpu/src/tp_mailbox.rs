//! The tensor split's in-graph mailbox, device side (the host endpoint is crate::tp_mailbox_host): mapped host slots that kernels fill and drain inside the
//! stream (`tp_mailbox_put` / `tp_mailbox_take`, or `tp_mailbox_put_pair` / `tp_mailbox_take_pair` for two parts), and the one host endpoint a proxy thread uses
//! to move each slot over RDMA. The host side touches the mapping only as `[AtomicU32]`:
//! payload words with Relaxed loads/stores, each slot's sequence word with Release (publish)
//! and Acquire (consume), so no host access is a data race. Layout in 32-bit words: one
//! 1024-word header (send sequence words, send lengths, receive sequence words and the
//! error word on separate 64-byte lines), then SLOTS send slots, then SLOTS receive slots.
//! Message s (1, 2, ... per direction, counted in u64) uses slot s % SLOTS and publishes the
//! low 32 bits of s; a slot is reused only four messages later, and the split's lockstep keeps
//! at most three messages in flight per direction, so a slot is
//! never rewritten while in use. Over more than two ranks the layout is crate::tp_route::Channels
//! (one receive channel per origin rank, and relayed slots start with a frame): `tp_mailbox_routed`.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use crate::tp_mailbox_host::{Bell, TpMailboxHost, SEND_SEQ, SEND_LEN, RECV_SEQ, ERROR, TAKE_PAUSE, WIDE, TAKE_CTAS};
use crate::tp_route::{Channels, Links};

/// The mapped mailbox of one rank (device-visible host memory), `slot_bytes` per slot.
/// `bell`: every queued put rings it, so a waiting write doorbell can park.
pub struct TpMailbox<'a> { vector: DeviceVector<'a>, host: *mut f32, layout: Channels, host_taken: AtomicBool, copy_group: AtomicBool, bell: Bell }
impl<'a> TpMailbox<'a> {
    fn words(&self) -> &[AtomicU32] {
        // SAFETY: the mapping is the layout's aligned u32 words (Channels::words), zeroed by the
        // driver before this view first exists (Backend::tp_mailbox), that live as long as
        // `self`; host code reads and writes them only as atomics, and kernels access them only
        // through the protocol in the module comment.
        unsafe { std::slice::from_raw_parts(self.host.cast::<AtomicU32>(), self.vector.len) }
    }
    fn device(&self, word: usize) -> u64 { self.vector.device_address() + (word * 4) as u64 }
    /// The proxy thread's endpoint: handed out once, so one thread owns the host side.
    pub fn take_host(&self) -> Option<TpMailboxHost<'_>> {
        (!self.host_taken.swap(true, Ordering::AcqRel)).then(|| TpMailboxHost::with_bell(self.words(), self.layout.slot_words, &self.bell))
    }
    /// A take timed out or the proxy failed (sticky).
    pub fn failed(&self) -> bool { self.words()[ERROR].load(Ordering::Acquire) != 0 }
    /// Check after the ordered readback drained queued consumers. Failure rejects the request.
    pub fn require_success(&self)->Result<(),String> {if self.failed() {Err("tensor split mailbox failed or produced nonfinite rows".into())} else {Ok(())}}
    /// Stop queued takes and ask the existing wire to notify its peer. Sticky.
    pub fn fail(&self) { self.words()[ERROR].store(1, Ordering::Release); }
    /// Bind a fresh mailbox once to an ordered copy group. Counts cannot reset
    /// while an old wire can still publish a message with the same sequence.
    pub fn claim_copy_group(&self) -> Result<(), String> {
        let words=self.words();
        let receive=self.layout.world.checked_sub(1).and_then(|n| n.checked_mul(self.layout.slots))
            .filter(|&n| n<=ERROR-RECV_SEQ).ok_or("tensor copy group has an invalid receive header")?;
        if self.layout.slots==0 || self.layout.slots>SEND_LEN-SEND_SEQ {
            return Err("tensor copy group has an invalid send header".into());
        }
        if self.failed() || words[SEND_SEQ..SEND_SEQ+self.layout.slots].iter().any(|v| v.load(Ordering::Acquire)!=0)
            || words[RECV_SEQ..RECV_SEQ+receive].iter().any(|v| v.load(Ordering::Acquire)!=0)
            || self.copy_group.swap(true,Ordering::AcqRel) {
            return Err("tensor copy group needs a fresh, unclaimed mailbox".into());
        }
        Ok(())
    }
    /// The rows one message holds, in bytes (a relayed slot's frame not counted).
    pub fn slot_bytes(&self) -> usize { (self.layout.slot_words - self.layout.frame) * 4 }
    pub fn layout(&self) -> Channels { self.layout }
    /// 1 when the host asked for four-word copies (tp_mailbox_host::WIDE).
    fn wide(&self) -> u64 { u64::from(self.words()[WIDE].load(Ordering::Relaxed) == 1) }
}

impl<'a> Backend<'a> {
    /// A zeroed mailbox with slots of `slot_bytes` (a multiple of 4).
    pub fn tp_mailbox(&self, slot_bytes: usize) -> Result<TpMailbox<'a>, String> {
        if slot_bytes == 0 || !slot_bytes.is_multiple_of(4) { return Err("tensor split mailbox slot size".into()); }
        self.tp_mailbox_routed(Channels::new(&Links::new(2, false)?, slot_bytes / 4)?)
    }
    /// A zeroed mailbox of this layout. A frame keeps the rows on 16-byte lines (four-word copies).
    pub fn tp_mailbox_routed(&self, layout: Channels) -> Result<TpMailbox<'a>, String> {
        if layout.slot_words == 0 || !layout.frame.is_multiple_of(4) || layout.frame > 0 && !layout.slot_words.is_multiple_of(4) {
            return Err("tensor split mailbox slot size".into());
        }
        let words = layout.slots.checked_mul(layout.world).and_then(|n| n.checked_mul(layout.slot_words))
            .and_then(|n| n.checked_add(crate::tp_mailbox_host::HEADER)).ok_or("mailbox size overflow")?;
        // Every word (header and payload) is zeroed by the driver, and the stream drained, before
        // the host view exists or any mailbox kernel runs.
        let mut vector = self.mapped_vector(words)?;
        self.zero_vector(&mut vector)?;
        self.synchronize()?;
        let host = self.mapped_exchange_host(&vector)?;
        Ok(TpMailbox { vector, host, layout, host_taken: AtomicBool::new(false), copy_group: AtomicBool::new(false), bell: Bell::default() })
    }
    /// Queue tp_mailbox_put: words [own, own + half) of each of `rows` rows of `width` words of
    /// `source` go to the send slot of message `seq`, then its sequence word is published.
    pub fn tp_mailbox_put(&self, m: &TpMailbox<'_>, source: &DeviceVector<'_>, rows: usize, width: usize, half: usize,
        own: usize, seq: u64) -> Result<(), String> {
        self.check_mailbox_owner(m, source)?;
        check_mailbox_shape(m, source.len, rows, width, half, own)?;
        let slot = seq as usize % m.layout.slots;
        let mut args = [source.buffer.address, rows as u64, width as u64, half as u64, own as u64, m.device(m.layout.send_slot(slot) + m.layout.frame),
            m.device(SEND_LEN + slot), m.device(SEND_SEQ + slot), (seq as u32) as u64 | m.wide() << 32];
        self.tp_mailbox_launch("tp_mailbox_put", 1, 1024, &mut args)?;
        m.bell.queued(seq);
        Ok(())
    }
    /// Floating resident rows reject nonfinite payloads through the sticky wire
    /// error. The consumer checks that error after its completion boundary.
    pub fn tp_mailbox_put_finite(&self,m:&TpMailbox<'_>,source:&DeviceVector<'_>,rows:usize,width:usize,half:usize,
        own:usize,seq:u64)->Result<(),String> {
        self.check_mailbox_owner(m,source)?;check_mailbox_shape(m,source.len,rows,width,half,own)?;
        let slot=seq as usize%m.layout.slots;
        let mut args=[source.buffer.address,rows as u64,width as u64,half as u64,own as u64,
            m.device(m.layout.send_slot(slot)+m.layout.frame),m.device(SEND_LEN+slot),m.device(SEND_SEQ+slot),seq as u32 as u64,m.device(ERROR)];
        self.tp_mailbox_launch("tp_mailbox_put_finite",1,1024,&mut args)?;m.bell.queued(seq);Ok(())
    }
    /// Queue tp_mailbox_take: wait (at most `limit_ns`, below 2^47) for message `seq` in its receive slot,
    /// then its rows go to words [peer, peer + half) of each `width`-word row of `target`.
    pub fn tp_mailbox_take(&self, m: &TpMailbox<'_>, target: &mut DeviceVector<'_>, rows: usize, width: usize, half: usize,
        peer: usize, seq: u64, limit_ns: u64) -> Result<(), String> {
        self.tp_mailbox_take_from(m, target, rows, width, half, peer, 0, seq, limit_ns)
    }
    /// tp_mailbox_take from receive channel `channel` (Channels::channel of the origin rank;
    /// the pair has only channel 0).
    #[allow(clippy::too_many_arguments)]
    pub fn tp_mailbox_take_from(&self, m: &TpMailbox<'_>, target: &mut DeviceVector<'_>, rows: usize, width: usize, half: usize,
        peer: usize, channel: usize, seq: u64, limit_ns: u64) -> Result<(), String> {
        self.check_mailbox_owner(m, target)?;
        check_mailbox_shape(m, target.len, rows, width, half, peer)?;
        if channel + 1 >= m.layout.world { return Err("tensor split mailbox channel outside the layout".into()); }
        let slot = seq as usize % m.layout.slots;
        // The limit keeps 47 bits; bit 47 asks for four-word copies and the top 16 carry the host's
        // poll pause (0 = the kernel's 2048 ns).
        let limit = limit_ns.min((1 << 47) - 1) | m.wide() << 47 | u64::from(m.words()[TAKE_PAUSE].load(Ordering::Relaxed) & 0xffff) << 48;
        let mut args = [target.buffer.address, rows as u64, width as u64, half as u64, peer as u64, m.device(m.layout.recv_slot(channel, slot) + m.layout.frame),
            m.device(m.layout.recv_seq(channel, slot)), (seq as u32) as u64, m.device(ERROR), limit];
        let most = match m.words()[TAKE_CTAS].load(Ordering::Relaxed) { 0 => 96, n => n.min(96) as usize };
        self.tp_mailbox_launch("tp_mailbox_take", (rows * half).div_ceil(256).clamp(1, most) as u32, 256, &mut args)
    }
    /// Publish two vector parts in one slot. Each tuple is (vector, width, columns, offset).
    pub fn tp_mailbox_put_pair(&self, m: &TpMailbox<'_>, parts: [(&DeviceVector<'_>, usize, usize, usize); 2],
        rows: usize, seq: u64) -> Result<(), String> {
        let [(x, xw, xc, xo), (y, yw, yc, yo)] = parts;
        self.check_mailbox_owner(m, x)?;
        self.check_mailbox_owner(m, y)?;
        crate::tp_payload::Pair::checked(rows, [(x.len, xw, xc, xo), (y.len, yw, yc, yo)], m.slot_bytes() / 4)
            .ok_or("tensor split mailbox pair outside the slot or the vectors")?;
        let slot = seq as usize % m.layout.slots;
        let mut args = [x.buffer.address, y.buffer.address, rows as u64, xw as u64, yw as u64, xc as u64, yc as u64, xo as u64, yo as u64,
            m.device(m.layout.send_slot(slot) + m.layout.frame), m.device(SEND_LEN + slot), m.device(SEND_SEQ + slot),
            (seq as u32) as u64 | m.wide() << 32];
        self.tp_mailbox_launch("tp_mailbox_put_pair", 1, 1024, &mut args)?;
        m.bell.queued(seq);
        Ok(())
    }
    /// Restore two parts from one origin message. Each tuple is (vector, width, columns, offset).
    pub fn tp_mailbox_take_pair_from(&self, m: &TpMailbox<'_>, parts: [(&mut DeviceVector<'_>, usize, usize, usize); 2],
        rows: usize, channel: usize, seq: u64, limit_ns: u64) -> Result<(), String> {
        let [(x, xw, xc, xo), (y, yw, yc, yo)] = parts;
        self.check_mailbox_owner(m, x)?;
        self.check_mailbox_owner(m, y)?;
        let p = crate::tp_payload::Pair::checked(rows, [(x.len, xw, xc, xo), (y.len, yw, yc, yo)], m.slot_bytes() / 4)
            .ok_or("tensor split mailbox pair outside the slot or the vectors")?;
        let end = |v: &DeviceVector<'_>| (v.len as u64).checked_mul(4).and_then(|bytes| v.buffer.address.checked_add(bytes));
        let (xe, ye) = (end(x).ok_or("mailbox vector address overflow")?, end(y).ok_or("mailbox vector address overflow")?);
        if x.buffer.address < ye && y.buffer.address < xe { return Err("tensor split mailbox pair targets overlap".into()); }
        if channel >= m.layout.world.saturating_sub(1) { return Err("tensor split mailbox channel outside the layout".into()); }
        let slot = seq as usize % m.layout.slots;
        let limit = limit_ns.min((1 << 47) - 1) | m.wide() << 47 | u64::from(m.words()[TAKE_PAUSE].load(Ordering::Relaxed) & 0xffff) << 48;
        let mut args = [x.buffer.address, y.buffer.address, rows as u64, xw as u64, yw as u64, xc as u64, yc as u64, xo as u64, yo as u64,
            m.device(m.layout.recv_slot(channel, slot) + m.layout.frame), m.device(m.layout.recv_seq(channel, slot)),
            (seq as u32) as u64, m.device(ERROR), limit];
        let most = match m.words()[TAKE_CTAS].load(Ordering::Relaxed) { 0 => 96, n => n.min(96) as usize };
        self.tp_mailbox_launch("tp_mailbox_take_pair", p.words().div_ceil(256).clamp(1, most) as u32, 256, &mut args)
    }
    /// The mailbox and the rows' vector were allocated in this Backend's context: a launch on
    /// another context could outlive the memory it reads or writes.
    fn check_mailbox_owner(&self, m: &TpMailbox<'_>, rows: &DeviceVector<'_>) -> Result<(), String> {
        self.check_vector(&m.vector, m.vector.len).and_then(|()| self.check_vector(rows, rows.len))
            .map_err(|_| "tensor split mailbox or rows belong to another backend".into())
    }
    /// The mailbox kernels take only u64 parameters (pointers, counts, the sequence word).
    fn tp_mailbox_launch(&self, name: &str, blocks: u32, threads: u32, args: &mut [u64]) -> Result<(), String> {
        let mut pointers: Vec<*mut core::ffi::c_void> = args.iter_mut().map(|a| (a as *mut u64).cast()).collect();
        // SAFETY: the named scalar or pair mailbox kernel takes these u64 parameters in this
        // order; every address and extent was checked against its vector or the mailbox, and
        // both outlive the ordered stream (the mailbox lives with the model).
        unsafe { self.launch_flash(name, blocks, threads, &mut pointers) }
    }
}

/// A message is `rows * half` words of a `rows * width`-word vector, inside one slot.
fn check_mailbox_shape(m: &TpMailbox<'_>, len: usize, rows: usize, width: usize, half: usize, offset: usize) -> Result<(), String> {
    let words = rows.checked_mul(width).ok_or("mailbox shape overflow")?;
    if rows == 0 || half == 0 || offset.checked_add(half).is_none_or(|end| end > width) || words > len
        || rows.checked_mul(half).is_none_or(|n| n > m.layout.slot_words - m.layout.frame) {
        return Err("tensor split mailbox shape outside the slot or the vector".into());
    }
    Ok(())
}
