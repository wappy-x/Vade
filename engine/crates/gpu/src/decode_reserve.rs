//! Spare device vectors of decode steps, kept apart from the vector pool.
//!
//! The vector pool is one list with a byte cap whose trims free the oldest returns first. A
//! prompt puts gigabytes of new buffers through it, so the buffers the decode steps of the
//! session before used are the first to go, and every session's first steps allocated them
//! again (about 380 MB of the same sizes at every context). A buffer that a decode step took
//! or allocated returns here, whenever it is freed. Prompt code never takes from here and no
//! trim frees from here. The reserve has its own byte cap: over it, the spare returned
//! longest ago leaves. It is spare storage: the allocation budget and the host pressure rule
//! release it before any allocation is refused.
use std::collections::{BTreeMap,VecDeque};

/// By exact byte length: (handle, return stamp), the newest at the back.
#[derive(Default)]
pub(crate) struct DecodeReserve {sizes:BTreeMap<usize,VecDeque<(u64,u64)>>,bytes:usize,cap:usize,stamp:u64,peak:usize}
impl DecodeReserve {
    /// `cap` 0 turns the reserve off: decode buffers then use the vector pool as before.
    pub(crate) fn configure(&mut self,cap:usize) {self.cap=cap;}
    pub(crate) fn configured(&self)->bool {self.cap!=0}
    /// Spare bytes now, the most ever held, and the cap.
    pub(crate) fn bound(&self)->(usize,usize,usize) {(self.bytes,self.peak,self.cap)}
    /// The spare of exactly `bytes` that returned last.
    pub(crate) fn take(&mut self,bytes:usize)->Option<u64> {
        let list=self.sizes.get_mut(&bytes)?;let (handle,_)=list.pop_back()?;
        if list.is_empty() {self.sizes.remove(&bytes);}
        self.bytes-=bytes;Some(handle)
    }
    /// Keep a returned buffer. The answer lists what must be freed now to stay under the cap:
    /// the spares returned longest ago, or the buffer itself when it alone is over the cap.
    pub(crate) fn keep(&mut self,handle:u64,bytes:usize)->Vec<(u64,usize)> {
        if bytes>self.cap {return vec![(handle,bytes)];}
        self.stamp+=1;self.sizes.entry(bytes).or_default().push_back((handle,self.stamp));self.bytes+=bytes;
        let mut leave=Vec::new();
        while self.bytes>self.cap {let Some(spare)=self.oldest() else {break;};leave.push(spare);}
        self.peak=self.peak.max(self.bytes);leave
    }
    /// Remove the spare that returned longest ago.
    pub(crate) fn oldest(&mut self)->Option<(u64,usize)> {
        let bytes=*self.sizes.iter().min_by_key(|(_,list)|list.front().map_or(u64::MAX,|spare|spare.1))?.0;
        let list=self.sizes.get_mut(&bytes)?;let (handle,_)=list.pop_front()?;
        if list.is_empty() {self.sizes.remove(&bytes);}
        self.bytes-=bytes;Some((handle,bytes))
    }
    /// Remove every spare.
    pub(crate) fn drain(&mut self)->Vec<(u64,usize)> {
        let all=std::mem::take(&mut self.sizes).into_iter().flat_map(|(bytes,list)|list.into_iter().map(move |(handle,_)|(handle,bytes))).collect();
        self.bytes=0;all
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_take_is_the_last_return_of_that_size_and_the_oldest_return_leaves_first() {
        let mut r=DecodeReserve::default();assert!(!r.configured());r.configure(100);assert!(r.configured());
        assert!(r.keep(1,40).is_empty());assert!(r.keep(2,40).is_empty());assert!(r.keep(3,10).is_empty());assert_eq!(r.bound(),(90,90,100));
        // Exact sizes only; the last return of a size comes back first.
        assert_eq!(r.take(41),None);assert_eq!(r.take(40),Some(2));assert_eq!(r.bound().0,50);
        assert!(r.keep(2,40).is_empty());
        // Over the cap the oldest return goes, whatever its size: 1 (40 B) before 3 (10 B).
        assert_eq!(r.keep(4,20),vec![(1,40)]);assert_eq!(r.bound().0,70);
        assert_eq!(r.keep(5,40),vec![(3,10)]);assert_eq!(r.bound().0,100);
        // A buffer larger than the cap is never kept and evicts nothing.
        assert_eq!(r.keep(6,101),vec![(6,101)]);assert_eq!(r.bound().0,100);
        assert_eq!(r.oldest(),Some((2,40)));assert_eq!(r.drain(),vec![(4,20),(5,40)]);assert_eq!(r.bound(),(0,100,100));assert_eq!(r.oldest(),None);
    }
    #[test]
    fn a_set_that_cycles_stays_while_sizes_no_step_asks_for_leave() {
        // A steady step takes and returns the same 30 buffers; 10 buffers of an old context are never asked for.
        let mut r=DecodeReserve::default();r.configure(40*8);let mut freed=Vec::new();
        for id in 0..10u64 {freed.extend(r.keep(1000+id,8));}
        for id in 0..30u64 {freed.extend(r.keep(id,8));}
        assert!(freed.is_empty());
        for _ in 0..5 {
            let taken=(0..30).map(|_|r.take(8).unwrap()).collect::<Vec<_>>();
            assert!(taken.iter().all(|&handle|handle<1000),"a step never needs the old buffers while its own are spare");
            for handle in taken {freed.extend(r.keep(handle,8));}
        }
        assert!(freed.is_empty());
        // A new size arrives: the old context's buffers leave, the cycling set stays whole.
        for id in 0..10u64 {freed.extend(r.keep(2000+id,8));}
        assert_eq!(freed.len(),10);assert!(freed.iter().all(|&(handle,_)|(1000..1010).contains(&handle)));
    }
    #[test]
    fn off_keeps_nothing() {
        let mut r=DecodeReserve::default();assert_eq!(r.keep(1,8),vec![(1,8)]);assert_eq!(r.take(8),None);assert_eq!(r.bound(),(0,0,0));
    }
}
