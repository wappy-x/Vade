//! Ordered resident work keeps every temporary allocation through its consumer.
use super::*;
/// One thread and one stream own the queue. A real readback still drains it.
/// Dropped contiguous buffers enter Context's deferred list until that drain.
/// This guard cannot leave capture active or release storage before completion.
pub struct ResidentQueue<'b,'a> {backend:&'b Backend<'a>}
impl<'a> Backend<'a> {
    /// A vector's buffer. With fewer stream waits a step, more freed buffers stay charged until a
    /// drain, so the budget must not refuse what a drain would free. A refused try leaves no state.
    pub(super) fn funded_buffer(&self,bytes:usize,managed:bool)->Result<Buffer<'a>,String> {
        let context=self.context;
        crate::allocation_budget::drain_and_retry(
            ||if managed {Buffer::allocate(context,bytes,true)} else {Buffer::pooled(context,bytes)},
            ||context.has_deferred() && context.accounted_bytes().saturating_add(bytes)>context.budget.get(),
            ||{self.stats.borrow_mut().budget_drains+=1;self.synchronize()})
    }
    /// LOOKAHEAD: a fence, and the count of buffers dropped so far that wait for a drain. `wait_fence_frees` frees
    /// exactly those, so a caller that queues more work behind the fence holds no buffer longer than a drain here would.
    pub fn fence_frees(&self)->Result<(Fence<'a>,usize),String> {let fence=self.fence()?;Ok((fence,self.context.deferred_total.get()))}
    /// LOOKAHEAD: wait for a `fence_frees` mark, then free the buffers that were dropped before it. Work queued behind
    /// the fence keeps running, and buffers dropped after the mark wait for the next drain.
    pub fn wait_fence_frees(&self,fence:Fence<'_>,frees:usize)->Result<(),String> {self.wait_fence(fence)?;self.context.release_deferred_before(frees);Ok(())}
    pub fn resident_queue_active(&self)->bool {self.context.resident_queued.get()}
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
        if let Err(error)=b.disable_resident_launches() {b.context.cleanup_errors.borrow_mut().push(error);}
        if let Err(error)=b.synchronize() {b.context.cleanup_errors.borrow_mut().push(error);}
        b.context.resident_queued.set(false);
    }
}
