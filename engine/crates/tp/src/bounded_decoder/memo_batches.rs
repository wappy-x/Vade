//! Whole-batch removal from the projection memo. No device type enters, so a host test covers it.
use std::collections::BTreeMap;
pub(super) struct Batch<K> {pub keys:Vec<K>,pub bytes:usize,pub used:u64}
/// What batch removal needs from the memo's index: the batch that owns a key, and its removal.
pub(super) trait Index<K> {
    fn owner(&self,key:&K)->Option<u64>;
    fn forget(&mut self,key:&K);
}
impl<K:Ord,V> Index<K> for BTreeMap<K,(V,u64)> {
    fn owner(&self,key:&K)->Option<u64> {self.get(key).map(|entry|entry.1)}
    fn forget(&mut self,key:&K) {self.remove(key);}
}
/// The memo's index in two levels: the projection (its name, rounding and topology), then the
/// row. A selection looks up thousands of rows of one projection in a decode step; the first
/// level is found once for all of them, and a row lookup compares rows only and allocates
/// nothing. The keys, their owners and every answer equal those of one map of (group, row).
pub(super) struct Origins<G,R,V> {groups:Vec<(G,Rows<R,V>)>}
/// The rows of one projection, found by their identity alone. No code walks this map, so its
/// order decides nothing: every answer is that of an ordered map with the same entries.
pub(super) type Rows<R,V>=std::collections::HashMap<R,(V,u64),std::hash::BuildHasherDefault<Mix>>;
/// A short multiply-and-rotate mix for keys this process made itself (row identities: a
/// source number and word offsets). No outside input reaches it.
#[derive(Default)]
pub(super) struct Mix(u64);
impl std::hash::Hasher for Mix {
    fn finish(&self)->u64 {self.0}
    fn write(&mut self,bytes:&[u8]) {for chunk in bytes.chunks(8) {let mut word=[0u8;8];word[..chunk.len()].copy_from_slice(chunk);self.write_u64(u64::from_le_bytes(word));}}
    fn write_u64(&mut self,word:u64) {self.0=(self.0.rotate_left(5)^word).wrapping_mul(0x517c_c1b7_2722_0a95);}
    fn write_usize(&mut self,word:usize) {self.write_u64(word as u64);}
    fn write_u8(&mut self,word:u8) {self.write_u64(u64::from(word));}
    fn write_u32(&mut self,word:u32) {self.write_u64(u64::from(word));}
}
impl<G,R,V> Default for Origins<G,R,V> {fn default()->Self {Self {groups:Vec::new()}}}
impl<G,R:Eq+std::hash::Hash,V> Origins<G,R,V> {
    /// The rows of the group that `is` accepts. A decoder has a few groups (one a sparse layer).
    pub fn group(&self,is:impl Fn(&G)->bool)->Option<&Rows<R,V>> {self.groups.iter().find(|(group,_)|is(group)).map(|(_,rows)|rows)}
    pub fn insert(&mut self,group:&G,row:R,value:(V,u64)) where G:PartialEq+Clone {
        let at=match self.groups.iter().position(|(g,_)|g==group) {Some(at)=>at,None=>{self.groups.push((group.clone(),Rows::default()));self.groups.len()-1}};
        self.groups[at].1.insert(row,value);
    }
    pub fn clear(&mut self) {self.groups.clear();}
    #[cfg(test)] pub fn len(&self)->usize {self.groups.iter().map(|(_,rows)|rows.len()).sum()}
}
impl<G:PartialEq,R:Eq+std::hash::Hash,V> Index<(G,R)> for Origins<G,R,V> {
    fn owner(&self,key:&(G,R))->Option<u64> {self.group(|group|*group==key.0)?.get(&key.1).map(|entry|entry.1)}
    fn forget(&mut self,key:&(G,R)) {
        // An empty group stays listed: its projection is asked again at the next token.
        if let Some((_,rows))=self.groups.iter_mut().find(|(group,_)|*group==key.0) {rows.remove(&key.1);}
    }
}
/// Remove these batches and every origin still owned by them. Returns the bytes freed. A key that a
/// later batch wrote again belongs to that batch and stays. A batch already evicted is skipped.
pub(super) fn discard<K>(origins:&mut impl Index<K>,batches:&mut BTreeMap<u64,Batch<K>>,ids:&[u64])->usize {
    let mut freed=0;
    for id in ids {
        let Some(batch)=batches.remove(id) else {continue;};
        for key in &batch.keys {if origins.owner(key)==Some(*id) {origins.forget(key);}}
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
    /// The two-level index answers every lookup, owner and removal as one map of (group, row) does.
    #[test] fn the_two_level_index_equals_one_map() {
        type Key=((&'static str,bool),u32);
        let mut flat:BTreeMap<Key,(u32,u64)>=BTreeMap::new();let mut split:Origins<(&'static str,bool),u32,u32>=Origins::default();
        let mut batches_flat:BTreeMap<u64,Batch<Key>>=BTreeMap::new();let mut batches_split:BTreeMap<u64,Batch<Key>>=BTreeMap::new();
        let groups=[("a",true),("a",false),("b",true)];let mut state=7u32;
        let mut next=|n:u32|{state=state.wrapping_mul(1664525).wrapping_add(1013904223);(state>>8)%n};
        let mut removed=0;
        for id in 0..40u64 {
            let group=groups[next(3) as usize];let keys=(0..1+next(5)).map(|_|(group,next(24))).collect::<Vec<Key>>();
            for key in &keys {flat.insert(*key,(id as u32,id));split.insert(&key.0,key.1,(id as u32,id));}
            batches_flat.insert(id,Batch {keys:keys.clone(),bytes:10*keys.len(),used:id});batches_split.insert(id,Batch {keys,bytes:0,used:id});
            for batch in batches_split.values_mut() {batch.bytes=10*batch.keys.len();}
            // A refusal drops batches, one of them twice and one that never was.
            if id%7==6 {
                let ids=[id-5,id-2,id-5,id+100];let before=flat.len();
                assert_eq!(discard(&mut split,&mut batches_split,&ids),discard(&mut flat,&mut batches_flat,&ids));removed+=before-flat.len();
            }
            assert_eq!(split.len(),flat.len());assert_eq!(batches_split.keys().collect::<Vec<_>>(),batches_flat.keys().collect::<Vec<_>>());
            for group in groups {for row in 0..24 {
                let want=flat.get(&(group,row)).copied();
                assert_eq!(split.group(|g|*g==group).and_then(|rows|rows.get(&row)).copied(),want,"group {group:?} row {row} after batch {id}");
                assert_eq!(Index::owner(&split,&(group,row)),want.map(|entry|entry.1));
            }}
        }
        assert!(removed>0,"the refusals removed no row, so the test compared nothing");
        assert!(split.group(|g|*g==("c",true)).is_none());split.clear();assert_eq!(split.len(),0);
    }
}
