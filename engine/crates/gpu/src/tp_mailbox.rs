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
#[path = "tp_reduce.rs"] mod tp_reduce;

/// A proof fixture sets this to time and compare the one-block put at every size. Production never sets it.
static PUT_NARROW:AtomicBool=AtomicBool::new(false);
pub fn set_put_narrow(narrow:bool) {PUT_NARROW.store(narrow,Ordering::Relaxed);}
/// The mapped mailbox of one rank (device-visible host memory), `slot_bytes` per slot.
/// `bell`: every queued put rings it, so a waiting write doorbell can park.
/// `inline`: the mailbox of the INLINE wire (tp_mailbox_host::TRAILER): every message ends at its slot's trailer and
/// moves with one write. Only the REDUCE launches (tp_reduce) use such a mailbox; the other launches refuse it.
pub struct TpMailbox<'a> { vector: DeviceVector<'a>, done: DeviceVector<'a>, host: *mut f32, layout: Channels, host_taken: AtomicBool, copy_group: AtomicBool, bell: std::sync::Arc<Bell>, inline: bool }
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
        (!self.host_taken.swap(true, Ordering::AcqRel)).then(|| { let host = TpMailboxHost::with_bell(self.words(), self.layout.slot_words, &self.bell); if self.inline { host.inline() } else { host } })
    }
    /// The doorbell's parks and parked ns, and the messages it served lingering and the ns it polled instead of parking.
    pub fn doorbell_stats(&self) -> [u64; 4] { let (a, b) = self.bell.stats(); let (c, d) = self.bell.linger_stats(); [a, b, c, d] }
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
    pub fn slot_bytes(&self) -> usize { (self.layout.slot_words - self.layout.frame - if self.inline { crate::tp_mailbox_host::TRAILER } else { 0 }) * 4 }
    pub fn is_inline(&self) -> bool { self.inline }
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
        // The wide put counts finished blocks here, in device memory. Zero between launches.
        let mut done = self.vector(1)?; self.zero_vector(&mut done)?;
        self.zero_vector(&mut vector)?;
        self.synchronize()?;
        let host = self.mapped_exchange_host(&vector)?;
        Ok(TpMailbox { vector, done, host, layout, host_taken: AtomicBool::new(false), copy_group: AtomicBool::new(false), bell: std::sync::Arc::new(Bell::default()), inline: false })
    }
    /// A zeroed pair mailbox of the INLINE wire whose slots hold messages of `slot_bytes` (a multiple of 16) and
    /// their trailer.
    pub fn tp_mailbox_inline(&self, slot_bytes: usize) -> Result<TpMailbox<'a>, String> {
        if slot_bytes == 0 || !slot_bytes.is_multiple_of(16) { return Err("inline tensor split mailbox slot size".into()); }
        let mut mailbox = self.tp_mailbox_routed(Channels::new(&Links::new(2, false)?, slot_bytes / 4 + crate::tp_mailbox_host::TRAILER)?)?;
        mailbox.inline = true;
        Ok(mailbox)
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
        self.tp_mailbox_launch("tp_mailbox_put", 1, 1, 1024, &mut args)?;
        self.resident_mailbox_queued(&m.bell,seq);
        Ok(())
    }
    /// Floating resident rows reject nonfinite payloads through the sticky wire
    /// error. The consumer checks that error after its completion boundary.
    pub fn tp_mailbox_put_finite(&self,m:&TpMailbox<'_>,source:&DeviceVector<'_>,rows:usize,width:usize,half:usize,
        own:usize,seq:u64)->Result<(),String> {
        self.check_mailbox_owner(m,source)?;check_mailbox_shape(m,source.len,rows,width,half,own)?;
        let slot=seq as usize%m.layout.slots;
        if crate::tp_put_wide::admitted(rows,half) && !PUT_NARROW.load(Ordering::Relaxed) {
            let most=match m.words()[TAKE_CTAS].load(Ordering::Relaxed) {0=>96,n=>n.min(96)};
            let mut args=[source.buffer.address+(own*4) as u64,half as u64,m.device(m.layout.send_slot(slot)+m.layout.frame),m.device(SEND_LEN+slot),m.device(SEND_SEQ+slot),
                (seq as u32) as u64|m.wide()<<32,m.device(ERROR),m.done.device_address()];
            self.tp_mailbox_launch("tp_mailbox_put_wide",crate::tp_put_wide::blocks(half,most),1,crate::tp_put_wide::THREADS,&mut args)?;
            crate::prompt_launches::record("tp_put_wide");self.resident_mailbox_queued(&m.bell,seq);return Ok(());
        }
        crate::prompt_launches::record("tp_put_narrow");
        let mut args=[source.buffer.address,rows as u64,width as u64,half as u64,own as u64,
            m.device(m.layout.send_slot(slot)+m.layout.frame),m.device(SEND_LEN+slot),m.device(SEND_SEQ+slot),seq as u32 as u64,m.device(ERROR)];
        self.tp_mailbox_launch("tp_mailbox_put_finite",1,1,1024,&mut args)?;self.resident_mailbox_queued(&m.bell,seq);Ok(())
    }
    /// One launch for one exchange message (tp_mailbox_put_join). The message is the `parts` in order; a part with
    /// `true` is BF16-rounded first. The message goes to the send slot exactly as tp_mailbox_put_finite sends the
    /// joined, rounded parts. Every word of `output` is written: each row [message offset, output offset, words] holds
    /// those message words, exactly as resident_scatter_copy places them, and every other word is zero, as the fresh
    /// fill leaves it. A nonfinite message word sets the sticky wire error and the message is not published.
    /// False: the shape is outside tp_put_join::admitted, nothing was queued, and the caller keeps its own launches.
    pub fn tp_mailbox_put_join(&self,m:&TpMailbox<'_>,parts:&[(&DeviceVector<'_>,bool)],output:&mut DeviceVector<'_>,rows:&[[usize;3]],seq:u64)->Result<bool,String> {
        self.check_vector(output,output.len())?;if output.is_segmented() || parts.iter().any(|(part,_)|part.is_segmented()) {return Ok(false);}
        let ends=|v:&DeviceVector<'_>|v.device_address().checked_add((v.len() as u64).checked_mul(4)?);
        let output_end=ends(output).ok_or("joined put output address overflow")?;let mut words=0usize;
        for (part,_) in parts {
            self.check_mailbox_owner(m,part)?;
            if part.device_address()<output_end && output.device_address()<ends(part).ok_or("joined put part address overflow")? {return Err("joined put part and output overlap".into());}
            words=words.checked_add(part.len()).ok_or("joined put message extent overflow")?;
        }
        check_mailbox_shape(m,words,1,words,words,0)?;
        if !crate::tp_put_join::admitted(parts.len(),rows.len(),words,output.len()) {return Ok(false);}
        let table=crate::tp_put_join::table(&parts.iter().map(|(part,round)|(part.len(),*round)).collect::<Vec<_>>(),output.len(),rows)?;
        let slot=seq as usize%m.layout.slots;
        let encoded=crate::tp_put_join::encode(&table,&parts.iter().map(|(part,_)|part.device_address()).collect::<Vec<_>>(),m.device(m.layout.send_slot(slot)+m.layout.frame),output.device_address(),m.wide()==1)?;
        let [output_blocks,threads]=crate::tp_put_join::grid(encoded.units);let (bytes,blocks)=(encoded.bytes,1+output_blocks);
        let mut args=[0,table.len() as u64|m.wide()<<32,encoded.units[0]|encoded.units[1]<<32,encoded.slot_rows as u64,m.device(SEND_LEN+slot),(words*4) as u64,m.device(SEND_SEQ+slot),seq as u32 as u64,m.device(ERROR)];
        crate::prompt_launches::record("tp_put_join");
        if self.record_resident_tabled(crate::resident_launch::Kind::PutJoin,[blocks,1,1,threads,0],&bytes,|address| {
            let mut c=crate::resident_launch::Command::new(crate::resident_launch::Kind::PutJoin);args[0]=address;
            for (i,&value) in args.iter().enumerate() {c.word(i,value);}c
        })? {self.resident_mailbox_queued(&m.bell,seq);self.resident_completion()?;return Ok(true);}
        let mut rows_vector=self.vector(bytes.len().div_ceil(4))?;self.write_vector_bytes(&mut rows_vector,&bytes)?;args[0]=rows_vector.device_address();
        let launched=self.tp_mailbox_launch("tp_mailbox_put_join",blocks,1,threads,&mut args);
        self.resident_mailbox_queued(&m.bell,seq);let completed=self.resident_completion();launched.and(completed)?;Ok(true)
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
        self.tp_mailbox_take_shared(m, target, rows, width, half, peer, channel, seq, limit_ns)
    }
    /// The same take for a plane that its owner shares only with later consumers on the ordered
    /// stream (an mHC pass plane behind an Rc): no host reference reads it before the stream drains.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn tp_mailbox_take_shared(&self, m: &TpMailbox<'_>, target: &DeviceVector<'_>, rows: usize, width: usize, half: usize,
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
        self.tp_mailbox_launch("tp_mailbox_take", (rows * half).div_ceil(256).clamp(1, most) as u32, 1, 256, &mut args)
    }
    /// Receive unchanged wire words directly into disjoint final job ranges.
    /// Same wait, timeout and sticky error as tp_mailbox_take: a failed take writes no target word.
    pub fn tp_mailbox_take_scatter(&self,m:&TpMailbox<'_>,target:&mut DeviceVector<'_>,rows:&[[usize;3]],channel:usize,seq:u64,limit_ns:u64)->Result<(),String> {
        self.check_mailbox_owner(m,target)?;
        if target.is_segmented() || !crate::scatter_copy::channel_inside(channel,m.layout.world) {return Err("scatter mailbox owner or channel differs".into());}
        let plan=crate::scatter_copy::Plan::checked(m.slot_bytes()/4,target.len(),rows)?;
        let slot=seq as usize%m.layout.slots;
        let bytes=plan.encode(m.device(m.layout.recv_slot(channel,slot)+m.layout.frame),target.device_address())?;
        let limit=limit_ns.min((1<<47)-1)|m.wide()<<47|u64::from(m.words()[TAKE_PAUSE].load(Ordering::Relaxed)&0xffff)<<48;
        let most=match m.words()[TAKE_CTAS].load(Ordering::Relaxed) {0=>96,n=>n.min(96) as usize};
        let grid=[crate::scatter_copy::take_blocks(plan.longest,plan.rows.len(),most),plan.rows.len() as u32,1,256,0];
        let mut args=[0,m.device(m.layout.recv_seq(channel,slot)),seq as u32 as u64,m.device(ERROR),limit];
        if self.record_resident_tabled(crate::resident_launch::Kind::TakeScatter,grid,&bytes,|address| {
            let mut c=crate::resident_launch::Command::new(crate::resident_launch::Kind::TakeScatter);args[0]=address;
            for (i,&value) in args.iter().enumerate() {c.word(i,value);}c
        })? {return self.resident_completion();}
        let mut table=self.vector(bytes.len()/4)?;self.write_vector_bytes(&mut table,&bytes)?;args[0]=table.device_address();
        let launched=self.tp_mailbox_launch("tp_mailbox_take_scatter",grid[0],grid[1],256,&mut args);
        let completed=self.resident_completion();launched.and(completed)
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
        self.tp_mailbox_launch("tp_mailbox_put_pair", 1, 1, 1024, &mut args)?;
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
        self.tp_mailbox_launch("tp_mailbox_take_pair", p.words().div_ceil(256).clamp(1, most) as u32, 1, 256, &mut args)
    }
    /// The mailbox and the rows' vector were allocated in this Backend's context: a launch on
    /// another context could outlive the memory it reads or writes.
    fn check_mailbox_owner(&self, m: &TpMailbox<'_>, rows: &DeviceVector<'_>) -> Result<(), String> {
        if let Some(error)=self.fatal_error() {return Err(error);}
        // These launches place a message at its slot's start; an inline mailbox ends it at the trailer.
        if m.inline {return Err("this tensor split launch does not take an inline mailbox".into());}
        self.check_vector(&m.vector, m.vector.len).and_then(|()| self.check_vector(rows, rows.len))
            .map_err(|_| "tensor split mailbox or rows belong to another backend".into())
    }
    /// The mailbox kernels take only u64 parameters (pointers, counts, the sequence word).
    fn tp_mailbox_launch(&self, name: &str, blocks: u32, rows: u32, threads: u32, args: &mut [u64]) -> Result<(), String> {
        let mut pointers: Vec<*mut core::ffi::c_void> = args.iter_mut().map(|a| (a as *mut u64).cast()).collect();
        // SAFETY: the named scalar or pair mailbox kernel takes these u64 parameters in this
        // order; every address and extent was checked against its vector or the mailbox, and
        // both outlive the ordered stream (the mailbox lives with the model). `rows` is 1 for every
        // kernel except tp_mailbox_take_scatter, whose grid-y indexes its table of exactly `rows`
        // checked HybridCopyRow entries; that table is retained through ordered completion. tp_reduce_take (tp_reduce.rs)
        // is one of these kernels: nine u64 parameters, checked by Backend::tp_reduce_take against the partial sum, the
        // output and the inline mailbox's receive slot.
        unsafe { self.launch_flash_rows(name, blocks, rows, threads, 0, &mut pointers) }
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
