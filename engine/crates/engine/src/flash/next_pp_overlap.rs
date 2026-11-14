//! Two verify groups in flight on a layer chain (three or more ranks). A tick's verify groups
//! (spark_serve::mtp_batch::verify_groups) hold different chats. The front runs the next group's
//! layers while a group is on the later ranks, and a middle rank sends a request on down the
//! chain without waiting for its reply, so it runs the next request while the later ranks run
//! this one. Every group runs the same kernels on the same rows in the same layer order as one
//! exchange at a time; only the time a group's work starts moves. The tensor split never posts.
use super::*;

/// Requests a link keeps on the wire: each side keeps two receive slots posted.
pub(in super::super) const IN_FLIGHT: usize = 2;
/// Rows of one in-flight verify group (the verify tile). A wide tick, one group of up to 32
/// rows, runs on a chain as groups of this size, two in flight.
pub(in super::super) const GROUP_ROWS: usize = 16;

/// Consecutive chat groups of at most `limit` rows, in order; a chat wider than `limit` stays
/// alone (as spark_serve::mtp_batch::verify_groups).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(in super::super) fn split_groups(widths: &[usize], limit: usize) -> Vec<std::ops::Range<usize>> {
    let (mut groups, mut start, mut total) = (Vec::new(), 0, 0);
    for (i, &n) in widths.iter().enumerate() {
        if i > start && total + n > limit {
            groups.push(start..i);
            (start, total) = (i, 0);
        }
        total += n;
    }
    if start < widths.len() { groups.push(start..widths.len()); }
    groups
}

/// The posted exchanges of one link.
#[derive(Default)]
pub(super) struct Window {
    /// Front: the next verify exchange posts and returns before its reply.
    pub(super) defer: bool,
    /// Front: the last verify exchange posted; its pass stops before the tail.
    pub(super) posted: bool,
    /// Exchange ids on the wire without a reply, oldest first, and the last sign of life.
    ids: VecDeque<u32>,
    heard: Option<Instant>,
    look: Option<Instant>,
}

impl Link {
    /// The exchange `meta`/`bytes` can post now: the wire carries both directions, the reply
    /// carries no kept rows, no queued tile waits, the split is off and a wire slot is free.
    pub(super) fn can_post(&self, chats: usize, bytes: usize) -> bool {
        !self.tp && !self.poisoned && self.owed == 0 && self.pending == 0 && self.window.ids.len() < IN_FLIGHT
            && self.relay_kept.as_ref().is_none_or(|rows| rows.is_empty())
            && self.fast.as_ref().is_some_and(|f| wire_fits(f.capacity(), REQUEST_HEAD + chats * 16, bytes, 0))
    }
    /// True while a posted exchange waits for its reply.
    pub(super) fn posted(&self) -> bool {
        !self.window.ids.is_empty()
    }
    /// Send one exchange on the wire; `collect` or `poll` takes its reply later, in order.
    pub(super) fn post_wire(&mut self, kind: u32, meta: &[[u32; 4]], residual: &[f32]) -> Result<()> {
        let bytes = std::mem::size_of_val(residual);
        if !residual.len().is_multiple_of(W) || meta.is_empty() || meta.len() > POOL_SIZE || !self.can_post(meta.len(), bytes) {
            return Err("pipeline post outside the wire window".into());
        }
        let id = self.sent.wrapping_add(1);
        let flagged = kind | if self.release_pool { RELEASE_POOLED } else { 0 } | self.released;
        let head = request_head(flagged, residual.len() / W, meta, id);
        self.fast.as_mut().ok_or("pipeline wire missing")?.send(&[&head, f32_bytes(residual)])?;
        self.sent = id;
        self.release_pool = false;
        self.released = 0;
        self.carried += 1;
        self.bytes += (head.len() + 2 * bytes) as u64;
        if self.window.ids.is_empty() { self.window.heard = Some(Instant::now()); }
        self.window.ids.push_back(id);
        Ok(())
    }
    /// Take the oldest posted reply into `residual` (its rows) if it arrived; false if not yet.
    /// Heartbeats ride TCP and are read every 50 ms; silence past `limit` is an error.
    pub(super) fn poll(&mut self, residual: &mut [f32]) -> Result<bool> {
        let id = *self.window.ids.front().ok_or("pipeline reply poll without a posted exchange")?;
        let mut fast = self.fast.take().ok_or("pipeline wire missing")?;
        let arrived = fast.try_recv(&mut self.message);
        self.fast = Some(fast);
        let mut heard = self.window.heard.unwrap_or_else(Instant::now);
        if !arrived? {
            if self.window.look.is_none_or(|t| t.elapsed() > Duration::from_millis(50)) {
                self.window.look = Some(Instant::now());
                self.wire_beats(id, &mut heard)?;
                self.window.heard = Some(heard);
            }
            if heard.elapsed() > self.limit {
                return Err(format!("pipeline back reply timed out after {} ms without a heartbeat", self.limit.as_millis()));
            }
            return Ok(false);
        }
        let message = std::mem::take(&mut self.message);
        let result = self.wire_reply(&message, id, residual);
        self.message = message;
        result?;
        self.window.ids.pop_front();
        self.window.heard = Some(Instant::now());
        Ok(true)
    }
    /// Wait for the oldest posted reply into `residual`.
    pub(super) fn collect(&mut self, residual: &mut [f32]) -> Result<()> {
        let started = Instant::now();
        while !self.poll(residual)? {
            if self.down.load(Ordering::Relaxed) { return Err("pipeline stopped while a reply was owed".into()); }
            std::hint::spin_loop();
        }
        self.account(started, 0, residual.len() / W);
        Ok(())
    }
}

/// A middle rank's requests that went on down the chain without their replies: (id from
/// upstream, rows). Their replies go up in this order.
pub(super) type Relayed = VecDeque<(u32, usize)>;

/// Middle rank: request `kind` (flags removed) may go down the chain without waiting. Only a
/// verify pass on the wire with no release or tensor-split bits, with a free downstream slot.
pub(super) fn relays(kind: u32, flags: u32, on_wire: bool, down: Option<&Link>, chats: usize, rows: usize) -> bool {
    kind == KIND_VERIFY && flags == 0 && on_wire && down.is_some_and(|l| l.can_post(chats, rows * W * 4))
}

/// Middle rank: send the oldest relayed reply up once it came back (`wait`: until it does).
/// Returns true when one went up.
pub(super) fn relay(down: &mut Link, up: Option<&mut Box<dyn PpWire>>, relayed: &mut Relayed, rows: &mut Vec<f32>,
    wait: bool, stop: &AtomicBool) -> Result<bool> {
    let &(id, count) = relayed.front().ok_or("pipeline relay without a relayed request")?;
    rows.resize(count * W, 0.0);
    loop {
        if down.poll(rows)? { break; }
        if !wait { return Ok(false); }
        if stop.load(Ordering::Relaxed) { return Err("pipeline back stopped by signal".into()); }
        std::hint::spin_loop();
    }
    let mut header = Vec::with_capacity(REPLY_HEAD);
    for v in [MAGIC_RESPONSE, 0, (rows.len() * 4) as u32, id] { put(&mut header, v); }
    up.ok_or("pipeline wire missing")?.send(&[&header, f32_bytes(rows)])?;
    relayed.pop_front();
    Ok(true)
}

#[cfg(target_os = "linux")]
impl<'a> NextModel<'a> {
    /// Front: the chain can keep two verify groups of up to `chats`/`rows` in flight now (no
    /// split, wire up, nothing owed or posted).
    pub(in super::super) fn pp_overlap_ready(&self, chats: usize, rows: usize) -> bool {
        self.pp.borrow().as_ref().is_some_and(|l| l.mtp && !l.posted() && l.can_post(chats, rows * W * 4))
    }
    /// Front: post (true) or exchange (false) the next verify passes.
    pub(in super::super) fn pp_defer(&self, on: bool) {
        if let Some(l) = self.pp.borrow_mut().as_mut() { l.window.defer = on; }
    }
    pub(in super::super) fn pp_deferring(&self) -> bool {
        self.pp.borrow().as_ref().is_some_and(|l| l.window.defer)
    }
    /// Front: the pass just posted its rows (and clears the mark when `take`).
    pub(in super::super) fn pp_posted(&self, take: bool) -> bool {
        self.pp.borrow_mut().as_mut().is_some_and(|l| if take { std::mem::take(&mut l.window.posted) } else { l.window.posted })
    }
    /// Front: copy `count` verify rows out and post them; the reply comes from pp_collect_verify.
    pub(in super::super) fn pp_post_verify(&self, b: &Backend<'a>, meta: &[[u32; 4]], vector: &DeviceVector<'a>, count: usize) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().filter(|l| !l.poisoned).ok_or("pipeline link failed earlier; the front is stopping")?;
        let mut buffer = std::mem::take(&mut link.buffer);
        buffer.resize(count * W, 0.0);
        let result = b.read_vector_range_into(vector, 0, &mut buffer).and_then(|()| link.post_wire(KIND_VERIFY, meta, &buffer));
        link.buffer = buffer;
        link.window.posted = result.is_ok();
        result.map_err(|e| link.fail(e))
    }
    /// Front: the oldest posted verify reply, into `vector`.
    pub(in super::super) fn pp_collect_verify(&self, b: &Backend<'a>, vector: &mut DeviceVector<'a>, count: usize) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().filter(|l| !l.poisoned).ok_or("pipeline link failed earlier; the front is stopping")?;
        let mut buffer = std::mem::take(&mut link.buffer);
        buffer.resize(count * W, 0.0);
        let result = link.collect(&mut buffer)
            .and_then(|()| b.write_vector_range(vector, 0, &buffer))
            .and_then(|()| b.synchronize());
        link.buffer = buffer;
        result.map_err(|e| link.fail(e))
    }
}

#[cfg(test)]
#[path = "next_pp_overlap_tests.rs"]
mod tests;
