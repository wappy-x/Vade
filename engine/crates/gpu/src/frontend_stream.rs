//! A side stream whose events join the owning backend before row consumption.
use super::*;
// Stream drops first and drains outstanding gathers before operands are freed.
pub struct FrontendStream<'a> { stream: Stream<'a>, ready: Event<'a>, done: Event<'a> }
impl<'a> Backend<'a> {
    /// The next launch may start in the previous kernel's tail; that kernel must
    /// griddepcontrol.wait before reading the previous kernel's output.
    pub fn programmatic_next(&self) { self.programmatic.set(true) }
    /// Whether the loaded PTX exports `name` (newer entries, capability markers).
    pub fn has_flash(&self, name: &str) -> bool { self.module.function(name).is_ok() }
    pub fn pageable_host_page_tables(&self) -> Result<bool, String> {
        let mut value = 0; self.context.activate()?;
        unsafe { check(cuDeviceGetAttribute(&mut value, 100, self.context.device), "pageable host page tables (attribute 100)")?; }
        eprintln!("flashnext pageable_host_page_tables={value}"); Ok(value == 1)
    }
    pub fn frontend_stream(&self) -> Result<FrontendStream<'a>, String> {
        Ok(FrontendStream { stream: Stream::new(self.context)?, ready: Event::ready(&self.stream)?, done: Event::ready(&self.stream)? })
    }
    /// All raw kernel operands must outlive both streams; join before reuse.
    pub unsafe fn frontend_launch(&self, side: &FrontendStream<'a>, name: &str, args: &mut [*mut c_void]) -> Result<(), String> {
        if !ptr::eq(self.context, side.stream.context) { return Err("frontend stream context mismatch".into()); }
        self.context.activate()?; side.ready.record_again(&self.stream)?;
        check(cuStreamWaitEvent(side.stream.raw, side.ready.raw, 0), "frontend producer wait")?;
        let f = self.module.function(name)?;
        self.context.decode_enqueue(); self.context.pending.set(true);
        self.context.receipt_launch(f, 128, 0)?;
        check(cuLaunchKernel(f, 20, 1, 1, 128, 1, 1, 0, side.stream.raw, args.as_mut_ptr(), ptr::null_mut()), "frontend gather")?;
        side.done.record_again(&side.stream)
    }
    pub fn frontend_join(&self, side: &FrontendStream<'a>) -> Result<(), String> {
        self.context.activate()?;
        unsafe { check(cuStreamWaitEvent(self.stream.raw, side.done.raw, 0), "frontend consumer wait") }
    }
}

