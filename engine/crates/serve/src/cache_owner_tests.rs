use super::*;
use crate::Session;
use std::{cell::Cell,rc::Rc};
const OWNER:usize=227_934_208;
#[derive(Clone)]
struct Held {id:usize,fail:Rc<Cell<bool>>,walks:Rc<Cell<usize>>}
impl Session for Held {
    fn step(&mut self,_:u32)->Result<Vec<f32>,String> {Err("metadata fixture does not decode".into())}
    fn can_fork(&self)->bool {true}
    fn fork(&self)->Result<Self,String> {if self.fail.get() {Err("resume refused".into())} else {Ok(self.clone())}}
    fn retained_bytes(&self)->Option<usize> {Some(OWNER)}
    fn retained_allocations(&self)->Option<Vec<(usize,usize)>> {self.walks.set(self.walks.get()+1);Some(vec![(self.id,OWNER)])}
    fn retention_is_immutable(&self)->bool {true}
}
fn source(id:usize)->CachedSession<Held> {CachedSession {checkpoint_lease:None,session:Held {id,fail:Rc::new(Cell::new(false)),walks:Rc::new(Cell::new(0))},
    tokens:vec![1;256],next_logits:vec![0.0],reserved_session_bytes:OWNER}}
fn cache()->SessionCache<Held> {SessionCache::new(CacheLimits {entries:8,retained_bytes:4*OWNER,tokens_per_entry:1024,logits_per_entry:1,idle_ttl:Duration::from_secs(10)}).unwrap()}
fn physical(cache:&SessionCache<Held>)->usize {cache.allocations.values().map(|v|v.0).sum()}
#[test]
fn shared_owner_counts_survive_every_cache_removal_path() {
    let now=Instant::now();let first=source(1);let second=source(2);let mut cache=cache();
    let keys=[SessionKey::generate().unwrap(),SessionKey::generate().unwrap(),SessionKey::generate().unwrap()];
    cache.snapshot(keys[0],&first,now);cache.snapshot(keys[1],&first,now);
    assert_eq!((cache.len(),physical(&cache),cache.allocations[&1].1),(2,OWNER,2));
    first.session.fail.set(true);let refused=cache.fork_prefix(&keys[0],&first.tokens,now);
    assert!(refused.state.is_none() && refused.failed.len()==1);assert_eq!(physical(&cache),OWNER);
    first.session.fail.set(false);let live=cache.fork_prefix(&keys[0],&first.tokens,now).state.unwrap();
    drop(live);assert_eq!((physical(&cache),cache.allocations[&1].1),(OWNER,2),"cancel leaves both cached holders");
    cache.snapshot(keys[2],&second,now);assert_eq!(physical(&cache),2*OWNER);
    let walks=first.session.walks.get()+second.session.walks.get();
    let private:usize=cache.entries.iter().map(|e|e.charge).sum();
    cache.rebalance_within(OWNER+private,None,BTreeSet::new(),now);
    assert_eq!(first.session.walks.get()+second.session.walks.get(),walks,"pressure does not walk or clone immutable owner lists");
    assert_eq!(physical(&cache),OWNER);assert_eq!(cache.counts.evicted,2);
    cache.expire(now+Duration::from_secs(11));assert_eq!((cache.len(),cache.retained_bytes(),physical(&cache)),(0,0,0));
    let checkpoint=cache.checkpoint(None,None,&first,now).unwrap();
    let live=cache.fork_checkpoint(&checkpoint.handle,None,&first.tokens,now).0.unwrap();
    drop(live);cache.release_checkpoint(&checkpoint.handle,None);assert_eq!((cache.len(),cache.retained_bytes()),(0,0));
    cache.insert_snapshot(keys[0],first,now).unwrap();let live=cache.fork_prefix(&keys[0],&[1;256],now).state.unwrap();
    drop(cache.remove(0));drop(live);assert_eq!((cache.len(),cache.retained_bytes()),(0,0),"consumed entry then completed resume");
    cache.insert_snapshot(keys[1],second,now).unwrap();assert_eq!(physical(&cache),OWNER);drop(cache);
}
