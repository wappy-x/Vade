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
#[cfg(test)] mod tests {
    use super::*;
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
