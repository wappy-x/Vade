//! Whole-batch removal from the projection memo. No device type enters, so a host test covers it.
use std::collections::BTreeMap;
pub(super) struct Batch<K> {pub keys:Vec<K>,pub bytes:usize,pub used:u64}
/// Remove these batches and every origin still owned by them. Returns the bytes freed. A key that a
/// later batch wrote again belongs to that batch and stays. A batch already evicted is skipped.
pub(super) fn discard<K:Ord,V>(origins:&mut BTreeMap<K,(V,u64)>,batches:&mut BTreeMap<u64,Batch<K>>,ids:&[u64])->usize {
    let mut freed=0;
    for id in ids {
        let Some(batch)=batches.remove(id) else {continue;};
        for key in &batch.keys {if origins.get(key).is_some_and(|entry|entry.1==*id) {origins.remove(key);}}
        freed+=batch.bytes;
    }freed
}
#[cfg(test)] mod tests {
    use super::*;
    type Maps=(BTreeMap<&'static str,(u32,u64)>,BTreeMap<u64,Batch<&'static str>>);
    fn add((origins,batches):&mut Maps,id:u64,keys:&[&'static str],value:u32) {
        for key in keys {origins.insert(*key,(value,id));}batches.insert(id,Batch {keys:keys.to_vec(),bytes:10*keys.len(),used:id});
    }
    /// A deferred NaN refusal: rows written under a finite source's lineage while the count was
    /// unverified must not answer the clean retry, and rows from before the window must survive.
    #[test] fn a_refused_window_leaves_no_entry_a_retry_can_read() {
        let mut m:Maps=Default::default();
        add(&mut m,1,&["old-a","old-b"],100);
        add(&mut m,2,&["row-5","row-6"],0);add(&mut m,3,&["row-7"],0);
        assert_eq!(m.0["row-5"].0,0,"the wrong row is readable until the refusal");
        assert_eq!(discard(&mut m.0,&mut m.1,&[2,3]),30);
        for key in ["row-5","row-6","row-7"] {assert!(!m.0.contains_key(key),"{key} survived the refusal");}
        assert_eq!((m.0["old-a"].0,m.0["old-b"].0),(100,100));assert_eq!(m.1.keys().copied().collect::<Vec<_>>(),vec![1]);
        // The clean retry misses, computes the real row, and that row stands.
        add(&mut m,4,&["row-5"],55);assert_eq!(m.0["row-5"].0,55);
        assert_eq!(discard(&mut m.0,&mut m.1,&[]),0);assert_eq!(m.0["row-5"].0,55);
    }
    #[test] fn a_key_written_again_later_and_an_evicted_batch_are_handled() {
        let mut m:Maps=Default::default();
        add(&mut m,1,&["x","y"],1);add(&mut m,2,&["x"],2);
        assert_eq!(discard(&mut m.0,&mut m.1,&[1]),20);assert_eq!(m.0["x"],(2,2));assert!(!m.0.contains_key("y"));
        assert_eq!(discard(&mut m.0,&mut m.1,&[1,99]),0);assert_eq!(m.0["x"],(2,2));
    }
}
