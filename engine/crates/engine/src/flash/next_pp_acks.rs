//! The front reads queued-tile acks while it waits for its own GPU. A KEEP ack carries its
//! tile's rows; while it stays unread the back's pp-replier blocks, the back runs out of row
//! buffers and stops, and the front's sender then blocks on the next tile. Reading an ack
//! as soon as its header is on the socket keeps every byte and its order.
use super::*;

impl Link {
    /// True when a whole reply header is already on the socket; never waits. The sender
    /// thread shares the socket, and send_all retries a write that would block meanwhile.
    pub(super) fn header_ready(&mut self) -> Result<bool> {
        let mut probe = [0u8; REPLY_HEAD];
        self.stream.set_nonblocking(true).map_err(err)?;
        let peeked = self.stream.peek(&mut probe);
        self.stream.set_nonblocking(false).map_err(err)?;
        match peeked {
            Ok(0) => Err("pipeline back closed".into()),
            Ok(n) => Ok(n >= REPLY_HEAD),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => Ok(false),
            Err(e) => Err(format!("pipeline read: {e}")),
        }
    }
    /// Read every queued-tile header (a heartbeat, or an ack with the rows it carries)
    /// already on the socket, oldest first; never waits for one that has not come. `heard`
    /// moves with each header read (see `frame`).
    pub(super) fn read_ready(&mut self, heard: &mut Instant) -> Result<()> {
        while self.pending > 0 && self.header_ready()? {
            self.queued_frame(heard)?;
        }
        Ok(())
    }
    /// Run `wait` (this rank's host wait for its GPU) while a scoped thread reads the acks
    /// that arrive (read_ready every ACK_POLL), then once more after `wait` returns. Only
    /// that thread touches the link meanwhile; `wait` must not use it.
    pub(super) fn reading_acks<R>(&mut self, wait: impl FnOnce() -> Result<R>) -> Result<R> {
        if self.pending == 0 {
            return wait();
        }
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let link = &mut *self;
            let reader = scope.spawn(|| -> Result<()> {
                let mut heard = Instant::now();
                while !done.load(Ordering::Acquire) && link.pending > 0 {
                    link.read_ready(&mut heard)?;
                    std::thread::sleep(ACK_POLL);
                }
                link.read_ready(&mut heard)
            });
            let result = wait();
            done.store(true, Ordering::Release);
            let read = reader.join().map_err(|_| "pipeline ack reader panicked".to_string())?;
            read.and(result)
        })
    }
}
#[cfg(target_os = "linux")]
impl NextModel<'_> {
    /// Front: read the queued-tile acks already on the socket, so the rows they carry are back
    /// for observe_kept. It waits for no new header; the payload of a header that has arrived
    /// can still wait, within the read deadline (queued_frame).
    pub(in crate::flash) fn pp_read_acks(&self) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().filter(|l| !l.poisoned).ok_or("pipeline link failed earlier; the front is stopping")?;
        let mut heard = Instant::now();
        let read = link.read_ready(&mut heard);
        read.map_err(|e| link.fail(e))
    }
}
/// How often the front looks for an ack while it waits for its GPU.
const ACK_POLL: Duration = Duration::from_micros(500);
