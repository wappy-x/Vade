use super::*;

/// One wake of the back's heartbeat thread: while a request is marked and `period` has
/// passed (on `now`) since its last heartbeat, queue one heartbeat for it.
pub(super) fn beat(working: &Mutex<Option<(u32, Instant)>>, period: Duration, reply_tx: &mpsc::SyncSender<ReplySend>, now: impl Fn() -> Instant) {
    let mut mark = working.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((id, last)) = mark.as_mut()
        && now().saturating_duration_since(*last) >= period {
            // Full means a row payload is queued; the front is reading that frame.
            let _ = reply_tx.try_send(ReplySend::Bytes(working_head(*id)));
            *last = now();
        }
}

/// Tests: the front's half of a lockstep with a test back that owns the manual clock.
/// Before each step the front says it waits (`ready`); the back then advances the clock,
/// writes that step's TCP headers and sends how many it wrote (`step`).
#[cfg(test)]
pub(super) struct Steps {
    pub(super) ready: mpsc::Sender<()>,
    pub(super) step: mpsc::Receiver<usize>,
    /// Headers the back wrote that the front has not read yet.
    pub(super) unread: usize,
}

#[cfg(test)]
impl Link {
    fn manual_clock(&self) -> bool { self.test_clock.is_some() }
    fn test_next_step(&mut self) -> Result<()> {
        let stopped = || "test back stopped stepping the clock".to_string();
        let steps = self.test_steps.as_mut().ok_or("a manual clock needs test steps")?;
        steps.ready.send(()).map_err(|_| stopped())?;
        steps.unread += steps.step.recv().map_err(|_| stopped())?;
        Ok(())
    }
    /// The manual-clock stand-in for the read timeout of one TCP header: wait steps until
    /// a header is unread; false once `left` has passed on the manual clock first.
    fn test_header_within(&mut self, left: Duration) -> Result<bool> {
        let started = self.clock_now();
        while self.test_steps.as_ref().is_some_and(|s| s.unread == 0) {
            self.test_next_step()?;
            if self.clock_elapsed(started) >= left { return Ok(false); }
        }
        let steps = self.test_steps.as_mut().ok_or("a manual clock needs test steps")?;
        steps.unread -= 1;
        Ok(true)
    }
    /// One pass of the wire wait on the manual clock: wait for the next step, then until
    /// every unread header is in the socket, so the link check sees what the back wrote.
    pub(super) fn test_wire_step(&mut self) -> Result<()> {
        if !self.manual_clock() { return Ok(()); }
        self.test_next_step()?;
        let want = self.test_steps.as_ref().map_or(0, |s| s.unread) * REPLY_HEAD;
        let mut probe = vec![0u8; want];
        loop {
            match self.stream.peek(&mut probe) {
                Ok(0) if want > 0 => return Err("pipeline back closed".into()),
                Ok(n) if n >= want => return Ok(()),
                Ok(_) => std::thread::yield_now(),
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => (),
                Err(e) => return Err(format!("pipeline link: {e}")),
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Set while an Unbounded guard lives on this thread: read_message has no real-time limit.
    static UNBOUNDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
pub(super) fn test_unbounded() -> bool { UNBOUNDED.with(|u| u.get()) }
/// Tests: while it lives, reads on this thread wait on the manual clock, not on real time.
#[cfg(test)]
pub(super) struct Unbounded(());
#[cfg(test)]
impl Drop for Unbounded {
    fn drop(&mut self) { UNBOUNDED.with(|u| u.set(false)); }
}
#[cfg(test)]
impl Link {
    /// The manual-clock stand-in for the real-time limit of a TCP reply's payload: wait steps
    /// until `bytes` are in the socket, failing once the limit passes on the manual clock;
    /// then lift the real-time limit for the payload read (the returned guard). None without
    /// a manual clock: the production real-time limit applies.
    pub(super) fn test_payload_within(&mut self, bytes: usize) -> Result<Option<Unbounded>> {
        if !self.manual_clock() { return Ok(None); }
        let started = self.clock_now();
        let mut probe = vec![0u8; bytes];
        loop {
            self.stream.set_nonblocking(true).map_err(err)?;
            let seen = self.stream.peek(&mut probe);
            self.stream.set_nonblocking(false).map_err(err)?;
            match seen {
                Ok(n) if n >= bytes => break,
                Ok(0) if bytes > 0 => return Err("pipeline peer closed mid-message (0 of the payload)".into()),
                Ok(_) => (),
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => (),
                Err(e) => return Err(format!("pipeline read: {e}")),
            }
            if self.clock_elapsed(started) > self.limit {
                return Err(format!("pipeline peer read timed out after {} ms (payload)", self.limit.as_millis()));
            }
            self.test_next_step()?;
        }
        UNBOUNDED.with(|u| u.set(true));
        Ok(Some(Unbounded(())))
    }
}

impl Link {
    pub(super) fn clock_now(&self) -> Instant {
        #[cfg(test)]
        if let Some((origin, clock)) = &self.test_clock {
            return *origin + Duration::from_millis(clock.load(Ordering::Relaxed));
        }
        Instant::now()
    }
    pub(super) fn clock_elapsed(&self, started: Instant) -> Duration {
        self.clock_now().saturating_duration_since(started)
    }
    /// Read one TCP header for exchange `expect`; it must arrive within `limit` of `heard`
    /// (the last sign of life). None for a heartbeat (the back is still running `expect`;
    /// `heard` moves) or a late one of an answered wire exchange (`heard` stays), else the
    /// reply's (status, len) (`heard` moves).
    pub(super) fn frame(&mut self, expect: u32, heard: &mut Instant) -> Result<Option<(u32, usize)>> {
        let limit = self.limit;
        let silent = || format!("pipeline back reply timed out after {} ms without a heartbeat", limit.as_millis());
        let left = self.limit.saturating_sub(self.clock_elapsed(*heard));
        if left.is_zero() { return Err(silent()); }
        // The manual clock stands in for the read timeout (test_header_within).
        #[cfg(test)]
        let left = match self.manual_clock() {
            true if !self.test_header_within(left)? => return Err(silent()),
            true => Duration::MAX,
            false => left,
        };
        let mut reply = [0u8; REPLY_HEAD];
        read_rest(&mut self.stream, &mut reply, None, left).map_err(|e| if e.contains("timed out") { silent() } else { e })?;
        if word(&reply, 0) == MAGIC_RESPONSE && word(&reply, 1) == STATUS_WORKING {
            let id = word(&reply, 3);
            if word(&reply, 2) == 0 && (id == expect || self.posted() && id.wrapping_sub(expect) <= self.sent.wrapping_sub(expect)) && self.replied != self.sent {
                self.beats += 1;
                (self.floor, *heard) = (id, self.clock_now());
                return Ok(None);
            }
            if word(&reply, 2) == 0 && id.wrapping_sub(self.floor) < self.replied.wrapping_add(1).wrapping_sub(self.floor) {
                self.late += 1;
                self.floor = id;
                return Ok(None);
            }
            return Err(format!("pipeline heartbeat out of order: id {id}, expected {expect}"));
        }
        let reply = self.reply_head(&reply, expect)?;
        (self.floor, *heard) = (expect.wrapping_add(1), self.clock_now());
        Ok(Some(reply))
    }
    /// The reply header for exchange `expect` (sent just now): heartbeats are skipped, and
    /// the back must show life within `limit` of the send and of each heartbeat.
    pub(super) fn await_reply(&mut self, expect: u32) -> Result<(u32, usize)> {
        let mut heard = self.clock_now();
        loop {
            if let Some(reply) = self.frame(expect, &mut heard)? { return Ok(reply); }
        }
    }
}
