//! Print-only memo counters. No production reader depends on these records.
use std::collections::BTreeSet;
/// Most distinct evicted keys the history holds. A key evicted while it is full is not recorded and sets
/// `history_capped`: `reread` is then a lower bound. `evictions`, `refused` and `working_peak` stay exact.
pub const HISTORY_LIMIT:usize=4096;
pub(super) struct Proof<K,G> {pub evictions:u64,pub refused:u64,pub reread:u64,pub working_peak:usize,pub history_capped:bool,
    evicted:BTreeSet<K>,working:Vec<(G,u64,usize)>}
impl<K,G> Default for Proof<K,G> {fn default()->Self {Self {evictions:0,refused:0,reread:0,working_peak:0,history_capped:false,evicted:BTreeSet::new(),working:Vec::new()}}}
impl<K:Ord,G:PartialEq> Proof<K,G> {
    pub fn evict(&mut self,keys:impl IntoIterator<Item=K>) {self.evictions+=1;
        for key in keys {if self.evicted.len()<HISTORY_LIMIT {self.evicted.insert(key);} else if !self.evicted.contains(&key) {self.history_capped=true;}}
    }
    pub fn read(&mut self,key:&K,hit:bool) {if !hit && self.evicted.contains(key) {self.reread+=1;}}
    pub fn insert(&mut self,key:&K) {self.evicted.remove(key);}
    pub fn working(&mut self,group:G,source:u64,bytes:usize,chats:usize) {
        self.working.retain(|(g,s,_)|*g!=group || *s!=source);
        if self.working.iter().filter(|(g,_,_)|*g==group).count()>=chats {
            if let Some(at)=self.working.iter().position(|(g,_,_)|*g==group) {self.working.remove(at);}
        }
        self.working.push((group,source,bytes));self.working_peak=self.working_peak.max(self.working.iter().map(|(_,_,n)|n).sum());
    }
    /// The `MEMO PROOF` line. A report tool reads it: never reorder or rename a field; a new one goes at the end.
    /// `memo` is row, slab, peak row and peak slab bytes; `accounted` is the backend's accounted and peak bytes.
    pub fn receipt(&self,rank:u32,phase:&str,memo:[usize;4],limit:usize,accounted:[usize;2])->String {
        let [rows,slab,peak_rows,peak_slab]=memo;
        format!("MEMO PROOF rank={rank} phase={phase} row_bytes={rows} slab_bytes={slab} peak_row_bytes={peak_rows} peak_slab_bytes={peak_slab} working_peak_bytes={} evictions={} refused_rows={} read_after_eviction={} limit_bytes={limit} accounted_bytes={} peak_accounted_bytes={} history_capped={}",
            self.working_peak,self.evictions,self.refused,self.reread,accounted[0],accounted[1],self.history_capped)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    type P=Proof<u64,u8>;
    #[test]
    fn a_long_run_of_evictions_keeps_a_bounded_history_and_says_so() {
        let mut p=P::default();for key in 0..100_000u64 {p.evict([key]);}
        assert!(p.evicted.len()<=HISTORY_LIMIT,"{} keys kept",p.evicted.len());assert!(p.history_capped);assert_eq!(p.evictions,100_000);
        // A recorded key still counts; a dropped one cannot, so `reread` is a lower bound.
        p.read(&0,false);p.read(&99_999,false);assert_eq!(p.reread,1);
    }
    #[test]
    fn below_the_limit_nothing_is_dropped_and_rereads_count_as_before() {
        let mut p=P::default();p.evict(0..10);p.evict([3,20]);
        p.read(&3,false);p.read(&3,true);p.read(&50,false);p.insert(&4);p.read(&4,false);p.read(&20,false);
        assert_eq!((p.evictions,p.reread,p.history_capped,p.evicted.len()),(2,2,false,10));
        // Exactly full is not capped, and a key already recorded loses nothing; one more new key is dropped.
        let mut p=P::default();p.evict(0..HISTORY_LIMIT as u64);p.evict([7]);assert!(!p.history_capped);
        p.evict([HISTORY_LIMIT as u64]);assert!(p.history_capped);assert_eq!(p.evicted.len(),HISTORY_LIMIT);
    }
    #[test]
    fn the_receipt_keeps_its_fields_in_order_and_ends_with_the_cap() {
        let mut p=P {evictions:2,refused:3,reread:4,working_peak:5,..P::default()};
        assert_eq!(p.receipt(1,"work",[6,7,8,9],10,[11,12]),"MEMO PROOF rank=1 phase=work row_bytes=6 slab_bytes=7 peak_row_bytes=8 peak_slab_bytes=9 working_peak_bytes=5 evictions=2 refused_rows=3 read_after_eviction=4 limit_bytes=10 accounted_bytes=11 peak_accounted_bytes=12 history_capped=false");
        p.history_capped=true;assert!(p.receipt(0,"shutdown",[0;4],0,[0;2]).ends_with(" peak_accounted_bytes=0 history_capped=true"));
    }
}
