//! The tensor split's write wire: the RC queue pair that moves tensor split messages with one-sided
//! writes, from this rank's mapped TP mailbox (tp_mailbox_host layout) straight into the peer's.
//! It registers exactly the mailbox's words (access LOCAL_WRITE | REMOTE_WRITE) and borrows them,
//! so the registration and the queue pair (both released on drop, queue pair first) never outlive
//! the mapping. The caller picks the device and RoCE v2 GID (sysfs) and swaps `hello` with the
//! peer over its own control link. Verbs ABI as in engine_fabric::verbs (rdma-core on aarch64).
use crate::tp_mailbox_host::TpDoorbell;
use engine_fabric::verbs::{Ah, Attr, Cap, Context, Gid, Init, Mr, Qp, Route, Send, Sge, Verbs, Wc};
use std::{ffi::{c_void, CStr}, ops::Range, ptr, sync::atomic::AtomicU32};

fn code(value: i32, what: &str) -> Result<(), String> {
    if value == 0 { Ok(()) } else { Err(format!("{what}: {}", std::io::Error::from_raw_os_error(value))) }
}
fn present<T>(p: *mut T, what: &str) -> Result<*mut T, String> {
    if p.is_null() { Err(format!("{what}: {}", std::io::Error::last_os_error())) } else { Ok(p) }
}

/// One message's writes, checked against both mailboxes (`words` long each): a non-empty payload
/// `from` to the peer's words from `to`, then the one-word sequence word `flag` to `flag_to`.
#[derive(Debug, PartialEq)]
struct Piece { from: Range<usize>, to: usize }
fn chain(words: usize, from: Range<usize>, to: usize, flag: usize, flag_to: usize) -> Result<[Piece; 2], String> {
    let fits = |r: &Range<usize>, at: usize| r.start < r.end && r.end <= words && at.checked_add(r.len()).is_some_and(|e| e <= words);
    if flag >= words || !fits(&from, to) || !fits(&(flag..flag + 1), flag_to) || from.len() > u32::MAX as usize / 4 {
        return Err("tensor split write outside the mailbox".into());
    }
    Ok([Piece { from, to }, Piece { from: flag..flag + 1, to: flag_to }])
}

/// A message that is one run of more than one word and ends in its own sequence word, at both ends of the wire
/// (the inline wire, tp_mailbox_host::TRAILER): one write carries it.
fn inline(payload: &Piece, seq: &Piece) -> bool {
    payload.from.len() > 1 && seq.from.start + 1 == payload.from.end && seq.to + 1 == payload.to + payload.from.len()
}

/// `hello`: GID (16), queue pair number (4), path MTU code (4), mailbox address (8), key (4),
/// mailbox bytes (4).
pub const HELLO: usize = 40;

pub struct TpRdma<'m> {
    verbs: &'static Verbs, context: *mut Context, pd: *mut c_void, cq: *mut c_void, qp: *mut Qp, mr: *mut Mr,
    words: &'m [AtomicU32], gid: [u8; 16], index: u8, mtu: i32, peer: Option<(u64, u32)>,
}
// One owner thread uses the queue pair; no verbs object is shared.
unsafe impl std::marker::Send for TpRdma<'_> {}
impl<'m> TpRdma<'m> {
    /// Open device `name`, RoCE v2 GID `index` (`gid`), path MTU code `mtu` (1..=5): protection
    /// domain, completion queue, RC queue pair in INIT with remote write, and `words` registered.
    pub fn open(name: &str, index: u8, gid: [u8; 16], mtu: i32, words: &'m [AtomicU32]) -> Result<Self, String> {
        let verbs = Verbs::load()?;
        if words.is_empty() || !(1..=5).contains(&mtu) { return Err("tensor split write wire setup".into()); }
        let mut r = Self { verbs, context: ptr::null_mut(), pd: ptr::null_mut(), cq: ptr::null_mut(), qp: ptr::null_mut(), mr: ptr::null_mut(),
            words, gid, index, mtu, peer: None };
        // SAFETY: verbs calls with the arguments verbs.h documents; every handle is checked
        // before use and released in Drop (in reverse order), whatever step fails here.
        unsafe {
            let mut count = 0;
            let list = present((verbs.get_device_list)(&mut count), "verbs devices")?;
            for i in 0..count {
                let device = *list.add(i as usize);
                if CStr::from_ptr((verbs.get_device_name)(device)).to_bytes() == name.as_bytes() { r.context = (verbs.open_device)(device); break; }
            }
            (verbs.free_device_list)(list);
            present(r.context, "open tensor split HCA")?;
            r.pd = present((verbs.alloc_pd)(r.context), "allocate PD")?;
            r.cq = present((verbs.create_cq)(r.context, 64, ptr::null_mut(), ptr::null_mut(), 0), "create CQ")?;
            let mut init = Init { user: ptr::null_mut(), send_cq: r.cq, recv_cq: r.cq, srq: ptr::null_mut(),
                cap: Cap { send: 64, recv: 1, send_sge: 1, recv_sge: 1, inline: 0 }, kind: 2, signal_all: 0 };
            r.qp = present((verbs.create_qp)(r.pd, &mut init), "create RC QP")?;
            let mut attr = Attr { state: 1, port: 1, access: 2, ..Default::default() };
            code((verbs.modify_qp)(r.qp, &mut attr, (1 << 0) | (1 << 3) | (1 << 4) | (1 << 5)), "QP INIT")?;
            // The words are pinned, mapped host memory that `words` keeps alive for 'm.
            r.mr = present((verbs.reg_mr)(r.pd, words.as_ptr() as *mut c_void, std::mem::size_of_val(words), 3), "register tensor split mailbox")?;
        }
        Ok(r)
    }
    pub fn hello(&self) -> [u8; HELLO] {
        let mut h = [0u8; HELLO];
        // SAFETY: qp and mr were created in open and live until drop.
        let (number, rkey) = unsafe { ((*self.qp).number, (*self.mr).rkey) };
        h[..16].copy_from_slice(&self.gid);
        h[16..20].copy_from_slice(&number.to_le_bytes());
        h[20..24].copy_from_slice(&self.mtu.to_le_bytes());
        h[24..32].copy_from_slice(&(self.words.as_ptr() as u64).to_le_bytes());
        h[32..36].copy_from_slice(&rkey.to_le_bytes());
        h[36..].copy_from_slice(&(std::mem::size_of_val(self.words) as u32).to_le_bytes());
        h
    }
    /// The peer's `hello`: its mailbox must have this one's size; the queue pair goes to RTS.
    pub fn ready(&mut self, peer: &[u8; HELLO]) -> Result<(), String> {
        let word = |r: Range<usize>| u32::from_le_bytes(peer[r].try_into().unwrap());
        if word(36..40) as usize != std::mem::size_of_val(self.words) { return Err("the peer's tensor split mailbox differs in size".into()); }
        let mtu = self.mtu.min(word(20..24) as i32);
        if !(1..=5).contains(&mtu) { return Err("invalid peer RoCE MTU".into()); }
        // From here `mtu` is the path MTU both ends use: the inline write sizes its packets with it.
        self.mtu = mtu;
        let mut attr = Attr { state: 2, mtu, destination: word(16..20), dest_atomic: 1, rnr_timer: 1, ..Default::default() };
        attr.ah = Ah { route: Route { gid: Gid(peer[..16].try_into().unwrap()), index: self.index, hops: 1, ..Default::default() }, global: 1, port: 1, ..Default::default() };
        // SAFETY: modify_qp on the queue pair created in open, with attribute masks for RTR / RTS.
        unsafe {
            code((self.verbs.modify_qp)(self.qp, &mut attr, (1 << 0) | (1 << 7) | (1 << 8) | (1 << 12) | (1 << 15) | (1 << 17) | (1 << 20)), "QP RTR")?;
            let mut attr = Attr { state: 3, timeout: 14, retries: 7, rnr_retries: 7, atomic: 1, ..Default::default() };
            code((self.verbs.modify_qp)(self.qp, &mut attr, (1 << 0) | (1 << 9) | (1 << 10) | (1 << 11) | (1 << 13) | (1 << 16)), "QP RTS")?;
        }
        self.peer = Some((u64::from_le_bytes(peer[24..32].try_into().unwrap()), word(32..36)));
        Ok(())
    }
}
impl TpDoorbell for TpRdma<'_> {
    /// Two chained RC writes: the payload (unsignaled), then the staged sequence word
    /// (signaled, completion id `id`). One RC queue pair executes them in order.
    /// The INLINE form: a payload of more than one word whose sequence word is its own last word (and lands as the
    /// last word) moves as ONE signaled write. Its packets land in order, so the sequence word lands last.
    fn post(&mut self, from: Range<usize>, to: usize, flag: usize, flag_to: usize, id: u64) -> Result<(), String> {
        let (address, key) = self.peer.ok_or("tensor split write wire not ready")?;
        let [payload, seq] = chain(self.words.len(), from, to, flag, flag_to)?;
        // SAFETY: mr lives until drop.
        let lkey = unsafe { (*self.mr).lkey };
        let sge = |p: &Piece| Sge { address: self.words[p.from.start..].as_ptr() as u64, length: (p.from.len() * 4) as u32, key: lkey };
        let (mut first, mut second) = (sge(&payload), sge(&seq));
        let remote = |p: &Piece| [address + (p.to * 4) as u64, key as u64, 0, 0];
        let mut last = Send { id, next: ptr::null_mut(), sg: &mut second, count: 1, opcode: 0, flags: 2, imm: 0, wr: remote(&seq), qp_type: 0, padding: 0, tail: [0; 6] };
        let mut wr = Send { id: 0, next: &mut last, sg: &mut first, count: 1, opcode: 0, flags: 0, imm: 0, wr: remote(&payload), qp_type: 0, padding: 0, tail: [0; 6] };
        if inline(&payload, &seq) {
            (wr.id, wr.flags, wr.next) = (id, 2, ptr::null_mut());
        }
        // SAFETY: both gather entries lie inside `words` (chain), registered under lkey for this
        // value's life; the remote ranges lie inside the peer's mailbox of the same size (ready);
        // ops[25] is verbs.h's post_send, which copies the work requests (two, or the one of an inline
        // message, whose `next` is null) before returning.
        unsafe {
            let mut bad = ptr::null_mut();
            let post: unsafe extern "C" fn(*mut Qp, *mut Send, *mut *mut Send) -> i32 = std::mem::transmute((*self.context).ops[25]);
            code(post(self.qp, &mut wr, &mut bad), "post tensor split write")
        }
    }
    fn reap(&mut self, done: &mut dyn FnMut(u64)) -> Result<(), String> {
        let mut wc = [Wc::default(); 8];
        // SAFETY: ops[11] is verbs.h's poll_cq on this value's CQ; `wc` holds the 8 entries it may fill.
        let n = unsafe {
            let poll: unsafe extern "C" fn(*mut c_void, i32, *mut Wc) -> i32 = std::mem::transmute((*self.context).ops[11]);
            poll(self.cq, 8, wc.as_mut_ptr())
        };
        if n < 0 { return Err("tensor split CQ poll failed".into()); }
        for w in &wc[..n as usize] {
            if w.status != 0 { return Err(format!("tensor split write completion status={} vendor={}", w.status, w.vendor)); }
            done(w.id);
        }
        Ok(())
    }
    /// Staging word `from` (holding 1) into the peer's word `to`, as payload and sequence word both.
    fn notify(&mut self, from: usize, to: usize) -> Result<(), String> { self.post(from..from + 1, to, from, to, u64::MAX) }
    /// Path MTU code m is 128 << m bytes (256 to 4,096): 32 << m words a packet. 0 until the queue pair is ready.
    fn packet_words(&self) -> usize { if self.peer.is_some() { 32usize << self.mtu } else { 0 } }
}
impl Drop for TpRdma<'_> {
    fn drop(&mut self) {
        // SAFETY: each handle was created in open (or is null) and is released once, the queue
        // pair before the registration it may still write into.
        unsafe {
            if !self.qp.is_null() { (self.verbs.destroy_qp)(self.qp); }
            if !self.mr.is_null() { (self.verbs.dereg_mr)(self.mr); }
            if !self.cq.is_null() { (self.verbs.destroy_cq)(self.cq); }
            if !self.pd.is_null() { (self.verbs.dealloc_pd)(self.pd); }
            if !self.context.is_null() { (self.verbs.close_device)(self.context); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// A message's two writes stay inside both mailboxes: the payload is never empty and the
    /// sequence word is one word; anything outside (or an overflowing offset) is refused.
    #[test]
    fn write_chain_stays_inside_the_mailbox() {
        assert_eq!(chain(100, 10..20, 60, 4, 8), Ok([Piece { from: 10..20, to: 60 }, Piece { from: 4..5, to: 8 }]));
        assert_eq!(chain(100, 90..100, 90, 99, 99).map(|c| c[1].to), Ok(99));
        for (from, to, flag, flag_to) in [(10..10, 60, 4, 8), (10..101, 0, 4, 8), (10..20, 91, 4, 8), (10..20, usize::MAX, 4, 8), (10..20, 60, 100, 8), (10..20, 60, 4, 100), (10..20, 60, usize::MAX, 8)] {
            assert!(chain(100, from.clone(), to, flag, flag_to).is_err(), "{from:?} {to} {flag} {flag_to}");
        }
    }
    /// One write only for a message that ends in its own sequence word at both ends. A header flag, the one-word
    /// error notice and any other shape keep two writes.
    #[test]
    fn only_a_run_that_ends_in_its_sequence_word_is_one_write() {
        let one = |from: Range<usize>, to, flag, flag_to| { let [payload, seq] = chain(100, from, to, flag, flag_to).unwrap(); inline(&payload, &seq) };
        assert!(one(10..20, 60, 19, 69));
        for (from, to, flag, flag_to) in [(10..20, 60, 4, 8), (5..6, 7, 5, 7), (10..20, 60, 19, 70), (10..20, 60, 18, 68), (10..20, 60, 19, 68), (10..20, 60, 20, 70)] {
            assert!(!one(from.clone(), to, flag, flag_to), "{from:?} {to} {flag} {flag_to}");
        }
    }
}
