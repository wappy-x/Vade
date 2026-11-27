//! Device owner lifetimes through the actual cache and rank ledger.
use super::*;
use super::kept_charge_tests::Local;
use crate::bounded_decoder::{Host,Parked,park_state};
use spark_gpu_probe::driver::Backend;
use spark_serve::{Session as Serving,cache::{CachedSession,CacheLimits,SessionCache,SessionKey}};
use std::{rc::Rc,time::{Duration,Instant}};
struct Snapshot<'a,'f,C:Compute> {factory:&'a Factory<'f,Host,Local>,compute:&'a C,state:Parked<C::Value>,fail:Rc<Cell<u8>>}
impl<'a,'f,C:Compute> Snapshot<'a,'f,C> {
    fn new(factory:&'a Factory<'f,Host,Local>,compute:&'a C,state:Parked<C::Value>,fail:Rc<Cell<u8>>)->Result<Self> {
        if !factory.kept_fits(state.owners().ok_or("missing owners")?) {return Err("snapshot cap reached".into());}
        factory.keep_count(&state);Ok(Self {factory,compute,state,fail})
    }
}
impl<C:Compute> Drop for Snapshot<'_,'_,C> {fn drop(&mut self) {self.factory.release_count(&self.state);}}
impl<C:Compute> Serving for Snapshot<'_,'_,C> {
    fn step(&mut self,_:u32)->Result<Vec<f32>> {Err("allocation fixture does not decode".into())}
    fn can_fork(&self)->bool {true}
    fn fork(&self)->Result<Self> {
        if self.fail.get()==1 {return Err("resume refused before park".into());}
        let next=Self::new(self.factory,self.compute,park_state(self.compute,&self.state)?,self.fail.clone())?;
        if self.fail.get()==2 {return Err("resume refused after park".into());}Ok(next)
    }
    fn retained_bytes(&self)->Option<usize> {self.state.retained_bytes()}
    fn retained_allocations(&self)->Option<Vec<(usize,usize)>> {self.state.owners().map(|o|o.allocations().to_vec())}
    fn retained_private_bytes(&self)->usize {self.state.owners().map_or(0,|o|o.private_bytes())}
    fn retention_is_immutable(&self)->bool {true}
}
fn cached<'a,'f,C:Compute>(factory:&'a Factory<'f,Host,Local>,compute:&'a C,state:Parked<C::Value>,fail:Rc<Cell<u8>>)->Result<CachedSession<Snapshot<'a,'f,C>>> {
    let tokens=vec![1;state.position as usize];let bytes=state.retained_bytes().ok_or("missing owner bytes")?;
    Ok(CachedSession {checkpoint_lease:None,session:Snapshot::new(factory,compute,state,fail)?,tokens,next_logits:vec![1.0],reserved_session_bytes:bytes})
}
fn drain(backend:&Backend<'_>)->Result<()> {backend.release_group_spares()?;backend.synchronize()?;backend.release_group_spares()}
fn check<C:Compute>(backend:&Backend<'_>,factory:&Factory<'_,Host,Local>,cache:Option<&SessionCache<Snapshot<'_,'_,C>>>,base:usize,label:&str)->Result<()> {
    drain(backend)?;let real=backend.allocated_bytes().checked_sub(base).ok_or("allocation baseline changed")?;
    let charged=factory.kept_prompts().1;let owners=factory.kept_owner_count();let rounding_limit=owners*4096;
    let cache_bytes=cache.map_or(0,SessionCache::retained_bytes);
    println!("KEPT SHARED DEVICE MEASURED phase={label} owners={owners} real_bytes={real} rank_charge_bytes={charged} cache_bytes={cache_bytes} rounding_limit_bytes={rounding_limit}");
    assert_eq!(charged,real,"{label}: aggregate rank charge must equal physical pooled owners");
    if let Some(cache)=cache {assert!(cache.retained_bytes()>=charged && cache.retained_bytes()-charged<=rounding_limit,"cache metadata exceeds stated page allowance");}
    Ok(())
}
/// Builds are outside measured idle states. `longer` uses actual append storage.
#[allow(clippy::too_many_arguments)]
pub(super) fn exercise<C:Compute>(backend:&Backend<'_>,factory:&Factory<'_,Host,Local>,compute:&C,base:usize,
    first:Parked<C::Value>,mut build:impl FnMut()->Result<Parked<C::Value>>,mut longer:impl FnMut(&Parked<C::Value>)->Result<Parked<C::Value>>)->Result<()> {
    let initial=first.retained_bytes().ok_or("missing owners")?;factory.keep_within(factory.full_state_reservation_bytes()+initial);
    let limits=CacheLimits {entries:8,retained_bytes:initial*4,tokens_per_entry:16384,logits_per_entry:1,idle_ttl:Duration::from_secs(10)};
    let mut cache=SessionCache::new(limits)?;let now=Instant::now();let fail=Rc::new(Cell::new(0));
    let keys=[SessionKey::generate()?,SessionKey::generate()?,SessionKey::generate()?,SessionKey::generate()?];
    let source=cached(factory,compute,first,fail.clone())?;let prompt=source.tokens.clone();
    cache.snapshot(keys[0],&source,now);cache.snapshot(keys[1],&source,now);drop(source);
    assert_eq!(factory.kept_prompts(),(2,initial));check(backend,factory,Some(&cache),base,"two_same_prompt_holders")?;
    fail.set(2);let refused=cache.fork_prefix(&keys[0],&prompt,now);assert!(refused.state.is_none() && !refused.failed.is_empty());fail.set(0);
    check(backend,factory,Some(&cache),base,"failed_resume_after_park")?;
    let live=cache.fork_prefix(&keys[0],&prompt,now).state.ok_or("resume failed")?;drop(live);
    check(backend,factory,Some(&cache),base,"cancel_resumed_holder")?;
    factory.keep_within(factory.full_state_reservation_bytes()+initial*4);
    let live=cache.fork_prefix(&keys[0],&prompt,now).state.ok_or("resume failed")?;
    let next=longer(&live.session.state)?;drop(live);let source=cached(factory,compute,next,fail.clone())?;
    cache.snapshot(keys[2],&source,now);drop(source);
    check(backend,factory,Some(&cache),base,"resumed_longer_shared_prefix")?;
    cache.expire(now+Duration::from_secs(11));assert_eq!(cache.len(),0);check(backend,factory,Some(&cache),base,"expiry_last_shared_holder")?;
    let source=cached(factory,compute,build()?,fail.clone())?;let prompt=source.tokens.clone();cache.snapshot(keys[0],&source,now);cache.snapshot(keys[1],&source,now);drop(source);
    let source=cached(factory,compute,build()?,fail.clone())?;cache.snapshot(keys[2],&source,now);drop(source);
    check(backend,factory,Some(&cache),base,"two_unrelated_prompts")?;
    let before=cache.retained_bytes();cache.rebalance_within(initial+before.saturating_sub(2*initial),None,Default::default(),now);
    assert_eq!(cache.len(),1);check(backend,factory,Some(&cache),base,"eviction_shared_owner_last_reference")?;
    cache.rebalance_within(initial*4,None,Default::default(),now);
    let live=cache.fork_prefix(&keys[2],&prompt,now).state.ok_or("resume failed")?;
    cache.rebalance_within(0,None,Default::default(),now);assert_eq!(cache.len(),0);
    check::<C>(backend,factory,None,base,"consumed_entry_live_holder")?;drop(live);
    check(backend,factory,Some(&cache),base,"completed_consumed_resume")?;
    cache.rebalance_within(initial*4,None,Default::default(),now);
    let source=cached(factory,compute,build()?,fail.clone())?;fail.set(2);cache.snapshot(keys[3],&source,now);drop(source);fail.set(0);
    assert_eq!(cache.len(),0);check(backend,factory,Some(&cache),base,"failed_snapshot_rollback")?;
    let source=cached(factory,compute,build()?,fail)?;cache.snapshot(keys[0],&source,now);cache.snapshot(keys[1],&source,now);drop(source);
    check(backend,factory,Some(&cache),base,"before_shutdown")?;drop(cache);
    check::<C>(backend,factory,None,base,"shutdown_empty")?;assert_eq!(factory.kept_prompts(),(0,0));Ok(())
}
