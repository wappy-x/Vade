//! A side stream whose events join the owning backend before row consumption.
use super::*;
// Stream drops first and drains outstanding gathers before operands are freed.
pub struct FrontendStream<'a> { stream: Stream<'a>, ready: Event<'a>, done: Event<'a> }
impl<'a> Backend<'a> {
    /// The next launch may start in the previous kernel's tail; that kernel must
    /// griddepcontrol.wait before reading the previous kernel's output.
    pub fn programmatic_next(&self) { self.programmatic.set(true) }
    /// The entry of the next launch of a decode step: under decode_fuse a step of up to four
    /// rows runs `chain` in a programmatic launch, else `plain` runs. Call it in the launch
    /// expression. The chain twins hold for these rows; other rows keep main's launches.
    pub fn chain_entry<'n>(&self, tokens: usize, plain: &'n str, chain: &'n str) -> &'n str {
        if crate::decode_fuse::chain_rows(crate::decode_fuse::on(), tokens) { self.programmatic_next(); chain } else { plain }
    }
    /// The grouped decode entry (decode_fuse::pick_group), marked programmatic for a twin.
    pub fn chain_group(&self, tokens: usize, mx: bool, bf16: bool, total: usize, cols: usize) -> &'static str {
        let (name, chain) = crate::decode_fuse::pick_group(crate::decode_fuse::on(), tokens, mx, bf16, total, cols);
        if chain { self.programmatic_next(); }
        name
    }
    /// The solo MXFP8 decode entry (decode_fuse::pick_solo), marked programmatic for a twin.
    pub fn chain_solo(&self, tokens: usize, cols: usize, threads: u32) -> &'static str {
        let (name, chain) = crate::decode_fuse::pick_solo(crate::decode_fuse::on(), tokens, cols, threads);
        if chain { self.programmatic_next(); }
        name
    }
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
    /// # Safety
    /// All raw kernel operands must outlive both streams; join before reuse.
    pub unsafe fn frontend_launch(&self, side: &FrontendStream<'a>, name: &str, args: &mut [*mut c_void]) -> Result<(), String> { self.flush_resident_launches()?;
        if !ptr::eq(self.context, side.stream.context) { return Err("frontend stream context mismatch".into()); }
        self.context.activate()?; side.ready.record_again(&self.stream)?;
        check(cuStreamWaitEvent(side.stream.raw, side.ready.raw, 0), "frontend producer wait")?;
        let f = self.module.function(name)?;
        self.context.decode_enqueue(); self.context.pending.set(true);
        self.context.receipt_launch(f, 128, 0)?;
        check(cuLaunchKernel(f, 20, 1, 1, 128, 1, 1, 0, side.stream.raw, args.as_mut_ptr(), ptr::null_mut()), "frontend gather")?;
        side.done.record_again(&side.stream)
    }
    pub fn frontend_join(&self, side: &FrontendStream<'a>) -> Result<(), String> { self.flush_resident_launches()?;
        self.context.activate()?;
        unsafe { check(cuStreamWaitEvent(self.stream.raw, side.done.raw, 0), "frontend consumer wait") }
    }
}

