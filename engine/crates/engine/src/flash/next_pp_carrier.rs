//! Which carrier a queued prompt tile takes on a chain of more than two ranks. Tiles cross on
//! TCP behind the sender thread, so a tile's transfer overlaps the next tile's work. A tile with
//! no ack unread (the first tile of a prompt, or a later one when the next rank kept up: that rank
//! is idle and waits for it) and the final tile (the later ranks wait for it) are on the critical
//! path, so those take the RDMA wire when it is there and they fit. The back takes requests in exchange-id order
//! from either carrier: a request that comes early on one carrier waits until the ids before it
//! came on the other (Early). The rows, kinds, ids and acks do not change.
use super::*;

/// Back: requests that came before their id. The wire holds one; TCP holds `tcp` and the requests
/// behind it. The prefetch reader stops after two parked requests (it has no spare rows left), so
/// it may not see a closed link: the link limit that a parked request counts stops the back.
pub(super) struct Early {
    pub(super) wire: Option<Vec<u8>>,
    pub(super) tcp: Option<super::prefetch::Request>,
    /// TCP requests behind `tcp`, in order.
    pub(super) later: std::collections::VecDeque<super::prefetch::Request>,
    /// The link limit, and when a request last started to wait for an earlier id.
    limit: Duration,
    parked: Option<Instant>,
}
impl Early {
    pub(super) fn new(limit: Duration) -> Self {
        Self { wire: None, tcp: None, later: Default::default(), limit, parked: None }
    }
    /// Some(true): the wire request `expect` is now in `message`; Some(false): the TCP request
    /// `expect` waits in `tcp`; None: neither carrier holds it yet. Err: the id cannot come, or
    /// it did not come within the link limit of a request that waits for it.
    pub(super) fn next(&mut self, expect: u32, message: &mut Vec<u8>) -> Result<Option<bool>> {
        if self.tcp.is_none() { self.tcp = self.later.pop_front(); }
        if self.wire.as_deref().is_some_and(|m| wire_id(m, expect) == expect) {
            *message = self.wire.take().unwrap_or_default();
            self.parked = None;
            return Ok(Some(true));
        }
        if self.tcp.as_ref().is_some_and(|r| r.id == expect) {
            self.parked = None;
            return Ok(Some(false));
        }
        if self.wire.is_some() && self.tcp.is_some() {
            return Err(format!("pipeline request out of order: neither carrier holds id {expect}"));
        }
        if self.wire.is_none() && self.tcp.is_none() {
            self.parked = None;
            return Ok(None);
        }
        if self.parked.get_or_insert_with(Instant::now).elapsed() > self.limit {
            return Err(format!("pipeline request out of order: id {expect} did not come within {} ms", self.limit.as_millis()));
        }
        Ok(None)
    }
    /// Poll the wire once, unless a wire request already waits: true when `message` now holds
    /// request `expect`; a later one waits in `wire`.
    pub(super) fn poll_wire(&mut self, fast: Option<&mut Box<dyn PpWire>>, expect: u32, message: &mut Vec<u8>) -> Result<bool> {
        let Some(f) = fast.filter(|_| self.wire.is_none()) else { return Ok(false) };
        if !f.try_recv(message)? { return Ok(false); }
        if wire_id(message, expect) == expect { return Ok(true); }
        self.wire = Some(std::mem::take(message));
        Ok(false)
    }
}

/// The exchange id of a wire request (a message too short for a header gives `expect`, so the
/// header check that follows rejects it).
pub(super) fn wire_id(message: &[u8], expect: u32) -> u32 {
    if message.len() < REQUEST_HEAD { expect } else { word(message, 4) }
}

impl Link {
    /// The queued tile `kind` (with `head` and `bytes` of rows) takes the wire: a tile with no
    /// ack unread or a prompt's final tile, on a chain of more than two ranks, a chain tile of the
    /// split or any tile of a plain chain, when the wire holds it.
    pub(super) fn tile_on_wire(&self, kind: u32, head: &[u8], bytes: usize) -> bool {
        let base = kind & !(crate::flash_tp_switch::MAIL | crate::flash_tp_switch::HEADS | crate::flash_tp_switch::HEAD);
        matches!(base, KIND_ROWS_ASYNC | KIND_ROWS_KEEP | KIND_ROWS_FINAL)
            && (base == KIND_ROWS_FINAL || self.pending == 0)
            && (self.chain_kept || self.relay_kept.is_some())
            && (self.tp_tile || !self.tp)
            && self.fast.as_ref().is_some_and(|f| head.len().checked_add(bytes).is_some_and(|n| n <= f.capacity()))
    }
    /// A tile is queued: its ack (or the final reply) is owed.
    pub(super) fn queued(&mut self) {
        self.pending += 1;
        self.posted += 1;
        self.release_pool = false;
        self.released = 0;
    }
}

#[cfg(test)]
#[path = "next_pp_carrier_tests.rs"]
mod tests;
