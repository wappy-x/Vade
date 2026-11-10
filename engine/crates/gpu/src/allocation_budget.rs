//! One inclusive byte sum for device, graph and pinned allocations.
#![forbid(unsafe_code)]
pub(crate) fn required(device:usize,graphs:usize,pinned:usize,extra:usize)->Result<usize,String> {
    device.checked_add(graphs).and_then(|n|n.checked_add(pinned)).and_then(|n|n.checked_add(extra)).ok_or_else(||"inclusive allocation accounting overflow".into())
}
/// A refused capture keeps the eager execution path and changes no counters.
pub(crate) fn graph_storage(device:usize,graphs:usize,pinned:usize,maximum:usize,graph_budget:usize,budget:usize)->Result<Option<usize>,String> {
    let storage=graphs.checked_add(maximum).ok_or("graph storage accounting overflow")?;
    if storage>graph_budget || required(device,storage,pinned,0)?>budget {return Ok(None);}
    Ok(Some(storage))
}
/// One more try after a drain, when the first failed and a drain can free charged bytes.
/// A deferred buffer stays charged until a real drain returns it to the pool, where the
/// next try's make_room can release it. Any other failure keeps its first error.
pub(crate) fn drain_and_retry<T>(mut attempt:impl FnMut()->Result<T,String>,can_drain:impl FnOnce()->bool,drain:impl FnOnce()->Result<(),String>)->Result<T,String> {
    match attempt() {
        Ok(value)=>Ok(value),
        Err(first)=>{if !can_drain() {return Err(first);}drain()?;attempt()},
    }
}
#[cfg(test)] mod tests {
    use super::*;
    /// A pool that charges deferred bytes until drained, as Context does.
    struct Charged {budget:usize,live:usize,deferred:usize,drains:usize}
    impl Charged {
        fn allocate(&mut self,bytes:usize)->Result<usize,String> {if self.live+self.deferred+bytes>self.budget {Err("over budget".into())} else {self.live+=bytes;Ok(bytes)}}
        fn run(&mut self,bytes:usize)->Result<usize,String> {
            let this=std::cell::RefCell::new(self);
            drain_and_retry(||this.borrow_mut().allocate(bytes),||{let s=this.borrow();s.live+s.deferred+bytes>s.budget && s.deferred>0},||{let mut s=this.borrow_mut();s.deferred=0;s.drains+=1;Ok(())})
        }
    }
    #[test] fn a_budget_failure_with_deferred_bytes_drains_once_and_retries() {
        let mut c=Charged {budget:100,live:40,deferred:50,drains:0};
        assert_eq!(c.run(10).unwrap(),10);assert_eq!(c.drains,0,"it fits: no drain");
        let mut c=Charged {budget:100,live:40,deferred:50,drains:0};
        assert_eq!(c.run(30).unwrap(),30);assert_eq!((c.drains,c.deferred,c.live),(1,0,70));
    }
    #[test] fn a_failure_a_drain_cannot_help_keeps_its_error() {
        // Nothing deferred: no drain.
        let mut c=Charged {budget:100,live:90,deferred:0,drains:0};assert_eq!(c.run(30).unwrap_err(),"over budget");assert_eq!(c.drains,0);
        // Still over after the drain: the retry's error, one drain only.
        let mut c=Charged {budget:100,live:90,deferred:5,drains:0};assert_eq!(c.run(30).unwrap_err(),"over budget");assert_eq!(c.drains,1);
        // A failed drain is the error.
        let mut tries=0;let result:Result<(),String>=drain_and_retry(||{tries+=1;Err("over budget".into())},||true,||Err("stream failed".into()));
        assert_eq!(result.unwrap_err(),"stream failed");assert_eq!(tries,1);
        // Another failure, with no budget pressure: no drain.
        let mut drained=false;let result:Result<(),String>=drain_and_retry(||Err("driver refused".into()),||false,||{drained=true;Ok(())});
        assert_eq!(result.unwrap_err(),"driver refused");assert!(!drained);
    }
    #[test] fn pinned_arena_and_graphs_share_the_device_ceiling() {
        let mib=1<<20;let ceiling=100*mib;
        assert_eq!(required(32*mib,4*mib,64*mib,0).unwrap(),ceiling);
        assert!(required(32*mib,4*mib,64*mib,1).unwrap()>ceiling);
        assert_eq!(required(0,0,64*mib,36*mib).unwrap(),ceiling);
        assert!(required(0,0,64*mib,36*mib+1).unwrap()>ceiling);
        for counts in [(usize::MAX,1,0,0),(0,usize::MAX,1,0),(0,0,usize::MAX,1)] {assert!(required(counts.0,counts.1,counts.2,counts.3).is_err());}
    }
    #[test] fn verify_capture_funds_pinned_arena_or_falls_back_before_allocation() {
        let mib=1<<20;
        assert_eq!(graph_storage(36*mib,4*mib,64*mib,8*mib,12*mib,112*mib).unwrap(),Some(12*mib));
        assert_eq!(graph_storage(36*mib,4*mib,64*mib,8*mib,12*mib,112*mib-1).unwrap(),None);
        assert_eq!(graph_storage(36*mib,4*mib,0,8*mib,12*mib,48*mib).unwrap(),Some(12*mib));
        assert_eq!(graph_storage(36*mib,4*mib,64*mib,8*mib,12*mib,48*mib).unwrap(),None);
        assert_eq!(graph_storage(0,4*mib,0,8*mib,12*mib-1,usize::MAX).unwrap(),None);
        assert!(graph_storage(0,usize::MAX,0,1,usize::MAX,usize::MAX).is_err());
    }
}
