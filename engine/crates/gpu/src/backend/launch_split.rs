//! Check tools only (feature `kernel-bench`; no serving build compiles this): the wall time of each direct kernel launch and
//! zero fill, by entry name. While the split is on, the stream is drained before and after every launch, so each time is that
//! launch alone on the device. `resident-decode-check nv-prompt-split` uses it to split one prompt projection call.
use super::*;
thread_local! {static LOG:RefCell<Option<Vec<(String,u128)>>>=const {RefCell::new(None)};}
/// One timed launch: made before the launch is queued, dropped after it (the drop drains the stream and keeps the time).
pub(super) struct Timed<'b,'a> {backend:&'b Backend<'a>,name:String,start:Instant}
impl<'a> Backend<'a> {
    /// Starts (true) or stops (false) the split. Returns the (entry, nanoseconds) rows kept since the last call, in launch order.
    pub fn launch_split(&self,on:bool)->Vec<(String,u128)> {LOG.with(|log|log.replace(on.then(Vec::new)).unwrap_or_default())}
    pub(super) fn split_timed<'b>(&'b self,name:&str)->Result<Option<Timed<'b,'a>>,String> {
        if !LOG.with(|log|log.borrow().is_some()) {return Ok(None);}
        self.synchronize()?;Ok(Some(Timed {backend:self,name:name.into(),start:Instant::now()}))
    }
}
impl Drop for Timed<'_,'_> {
    fn drop(&mut self) {
        // A failed drain stays on the context and fails the tool's own next drain.
        let _=self.backend.synchronize();let ns=self.start.elapsed().as_nanos();
        LOG.with(|log|if let Some(log)=log.borrow_mut().as_mut() {log.push((std::mem::take(&mut self.name),ns));});
    }
}
