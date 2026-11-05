//! Ordered resident work keeps every temporary allocation through its consumer.
use super::*;
/// One thread and one stream own the queue. A real readback still drains it.
/// Dropped contiguous buffers enter Context's deferred list until that drain.
/// This guard cannot leave capture active or release storage before completion.
pub struct ResidentQueue<'b,'a> {backend:&'b Backend<'a>}
impl<'a> Backend<'a> {
    pub fn resident_queue(&self)->Result<ResidentQueue<'_,'a>,String> {
        self.require_resident_stream()?;
        if self.context.resident_queued.get() {return Err("a resident queue is already active".into());}
        self.synchronize()?;self.context.resident_queued.set(true);Ok(ResidentQueue {backend:self})
    }
    /// Resident entry points may borrow because dropped owners stay retained.
    /// Ordinary callers keep the existing completion guarantee.
    pub(super) fn resident_completion(&self)->Result<(),String> {
        if self.context.resident_queued.get() {
            if let Some(error)=self.fatal_error() {return Err(error);}Ok(())
        } else {self.synchronize()}
    }
}
impl Drop for ResidentQueue<'_,'_> {
    fn drop(&mut self) {
        let b=self.backend;
        if let Err(error)=b.synchronize() {b.context.cleanup_errors.borrow_mut().push(error);}
        b.context.resident_queued.set(false);
    }
}
