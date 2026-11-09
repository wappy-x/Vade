//! Flash-Next verify segments (VADE_MTP_GRAPH): captured without a stream drain, updated in place.
use super::*;
/// Upload staging a verify segment may use: the layer-descriptor allowance the Flash-Next
/// workspace reserves (next_execution.rs, (1 << 18) f32 words); one segment uploads far less.
const SEGMENT_UPLOADS: usize = (1 << 18) * 4;
impl<'a> Backend<'a> {
    /// Flash-Next verify segment (VADE_MTP_GRAPH=1). Without draining the stream, charges a
    /// whole-model allowance for `layers` x `rows` and starts capturing; false (no charge, no
    /// capture) when the ceiling cannot fund it, and the caller enqueues the segment eagerly.
    /// A staging drain cannot run inside a capture (reserve_upload would synchronize), so
    /// false as well when the upload staging has less than SEGMENT_UPLOADS bytes free.
    /// A buffer dropped inside the capture is freed after the next drain, not in its drop.
    pub fn begin_segment(&self, layers: usize, rows: usize) -> Result<bool, String> {
        if self.segment.get().is_some() { return Err("verify segment capture already open".into()); }
        if !self.context.cleanup_errors.borrow().is_empty() { return Ok(false); }
        if self.uploads.try_borrow().map_or(true, |s| s.capacity - s.used < SEGMENT_UPLOADS) { return Ok(false); }
        let maximum = crate::GraphAllowance::whole_model(layers, rows)?.maximum_bytes();
        self.context.make_room(maximum)?;
        let Some(storage)=crate::allocation_budget::graph_storage(self.allocated_bytes(),self.context.graph_storage.get(),self.context.pinned_staging.get(),maximum,self.context.graph_budget.get(),self.context.budget.get())? else {return Ok(false);};
        self.context.activate()?;
        self.context.graph_storage.set(storage);
        self.context.peak_graph_storage.set(self.context.peak_graph_storage.get().max(storage));
        self.context.peak_accounted.set(self.context.peak_accounted.get().max(self.accounted_bytes()));
        if let Err(error) = unsafe { check(cuStreamBeginCapture_v2(self.stream.raw, 1), "cuStreamBeginCapture") } {
            self.context.graph_storage.set(storage - maximum);
            return Err(error);
        }
        self.segment.set(Some((self.profiling.replace(false), layers, rows)));
        self.context.capturing.set(true);
        Ok(true)
    }
    /// Ends the open verify segment and launches it on the stream. The retained executable
    /// is updated in place when the capture keeps its topology (new arguments and copy
    /// addresses, same kernels in the same order), else replaced by a new instantiation.
    /// Safety: as replay_decode; every captured buffer stays live until the launch completes.
    /// Returns whether the retained executable was updated in place.
    pub unsafe fn end_segment(&self, slot: &mut Option<DecodeGraph<'a>>) -> Result<bool, String> {
        let (profiling, layers, rows) = self.segment.take().ok_or("no open verify segment capture")?;
        let allowance = crate::GraphAllowance::whole_model(layers, rows)?;
        let maximum = allowance.maximum_bytes();
        let mut graph = ptr::null_mut();
        let ended = unsafe { check(cuStreamEndCapture(self.stream.raw, &mut graph), "cuStreamEndCapture") };
        self.context.capturing.set(false);
        // Work queued before the capture may still be in flight: `pending` stays set.
        self.profiling.set(profiling);
        if let Err(error) = ended { self.discard_capture(graph, maximum); return Err(error); }
        let updated = slot.as_ref().is_some_and(|exec| {
            let mut info = GraphExecUpdateInfo { result: 0, error_node: ptr::null_mut(), error_from_node: ptr::null_mut() };
            unsafe { cuGraphExecUpdate_v2(exec.raw, graph, &mut info) == 0 }
        });
        if updated {
            let destroyed = unsafe { cuGraphDestroy(graph) };
            self.context.record_cleanup(destroyed, "cuGraphDestroy");
            if destroyed != 0 { return Err("CUDA captured graph destruction failed".into()); }
            self.context.graph_storage.set(self.context.graph_storage.get() - maximum);
        } else {
            drop(slot.take());
            let mut nodes = 0;
            let counted = unsafe { check(cuGraphGetNodes(graph, ptr::null_mut(), &mut nodes), "cuGraphGetNodes") }
                .and_then(|_| allowance.storage_bytes(nodes));
            let storage_bytes = match counted {
                Ok(bytes) => bytes,
                Err(error) => { self.discard_capture(graph, maximum); return Err(error); }
            };
            let mut raw = ptr::null_mut();
            let instantiated = unsafe { check(cuGraphInstantiate_v2(&mut raw, graph, ptr::null_mut(), ptr::null_mut(), 0), "cuGraphInstantiate") };
            let destroyed = unsafe { cuGraphDestroy(graph) };
            self.context.record_cleanup(destroyed, "cuGraphDestroy");
            if let Err(error) = instantiated {
                let released = if raw.is_null() { true } else {
                    let code = unsafe { cuGraphExecDestroy(raw) };
                    self.context.record_cleanup(code, "cuGraphExecDestroy");
                    code == 0
                };
                if destroyed == 0 && released { self.context.graph_storage.set(self.context.graph_storage.get() - maximum); }
                return Err(error);
            }
            if destroyed != 0 {
                let code = unsafe { cuGraphExecDestroy(raw) };
                self.context.record_cleanup(code, "cuGraphExecDestroy");
                return Err("CUDA captured graph destruction failed".into());
            }
            self.context.graph_storage.set(self.context.graph_storage.get() - (maximum - storage_bytes));
            self.stats.borrow_mut().graph_captures += 1;
            *slot = Some(DecodeGraph { raw, context: self.context, storage_bytes });
        }
        unsafe { self.replay_decode(slot.as_ref().ok_or("missing verify segment graph")?)? }
        Ok(updated)
    }
    /// Whether a verify segment capture is open (begin_segment returned true, not yet ended).
    pub fn in_segment(&self) -> bool { self.segment.get().is_some() }
    /// Buffers dropped while a verify segment was capturing (freed after the next drain).
    pub fn deferred_frees(&self) -> usize { self.context.deferred_total.get() }
    /// Closes an open verify segment after a failed enqueue; nothing it recorded runs.
    pub fn abort_segment(&self) {
        if let Some((profiling, layers, rows)) = self.segment.take() {
            let mut graph = ptr::null_mut();
            let _ = unsafe { cuStreamEndCapture(self.stream.raw, &mut graph) };
            self.context.capturing.set(false);
            self.profiling.set(profiling);
            let maximum = crate::GraphAllowance::whole_model(layers, rows).map_or(0, |a| a.maximum_bytes());
            self.discard_capture(graph, maximum);
        }
    }
}
