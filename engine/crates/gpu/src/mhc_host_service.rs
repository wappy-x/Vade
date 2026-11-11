//! Ordered local mailbox work. Plane IDs remain live until a queued release.
use crate::{mhc_host::{HostBoundary,Planes,Pool},tp_mailbox_host::{TpMailboxHost,HEADER,RECV_SEQ,SEND_LEN,SEND_SEQ,TP_MAILBOX_SLOTS}};
use std::{collections::{HashMap,VecDeque},sync::{atomic::{AtomicBool,AtomicU64,AtomicUsize,Ordering},Arc,Mutex,OnceLock},thread::{self,Scope,ScopedJoinHandle,Thread},time::{Duration,Instant}};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Kind {Run {previous:Option<u64>},Residual {of:u64},Coefficients {of:u64},Release {of:u64}}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Ticket {pub put:Option<u64>,pub take:u64}
struct Job {ticket:Ticket,boundary:Option<Arc<HostBoundary>>,kind:Kind}
#[derive(Default)]
struct State {jobs:VecDeque<Job>,puts:u64,takes:u64,kept:HashMap<u64,Planes>,free:Vec<Planes>,error:Option<String>}
#[derive(Default)]
pub struct Queue {state:Mutex<State>,stop:AtomicBool,failed:AtomicBool,capacity:AtomicUsize,worker:OnceLock<Thread>,stats:[AtomicU64;8]}
impl Queue {
    pub fn new()->Arc<Self> {Arc::new(Self::default())}
    fn state(&self)->Result<std::sync::MutexGuard<'_,State>,String> {self.state.lock().map_err(|_|"host mHC queue lock is poisoned".into())}
    pub fn reserve_coefficients(&self,coefficients:usize)->Result<(),String> {
        if self.worker.get().is_some() {return Err("host mHC coefficient reservation follows service startup".into());}
        self.capacity.fetch_max(coefficients,Ordering::Release);Ok(())
    }
    pub fn push(&self,boundary:Arc<HostBoundary>,kind:Kind)->Result<Ticket,String> {
        let mut state=self.state()?;
        if self.failed.load(Ordering::Acquire) || self.stop.load(Ordering::Acquire) {return Err(state.error.clone().unwrap_or_else(||"host mHC service is stopped".into()));}
        if matches!(kind,Kind::Release {..}) {return Err("host mHC release jobs come from Queue::release".into());}
        let take=state.takes.checked_add(1).filter(|&sequence|sequence<=u32::MAX as u64).ok_or("host mHC receive sequence exhausted")?;
        let put=if matches!(kind,Kind::Run {..}) {Some(state.puts.checked_add(1).filter(|&sequence|sequence<=u32::MAX as u64).ok_or("host mHC send sequence exhausted")?)} else {None};
        self.capacity.fetch_max(boundary.shape().coefficients(),Ordering::Release);
        let ticket=Ticket {put,take};state.takes=take;if let Some(put)=put {state.puts=put;}
        state.jobs.push_back(Job {ticket,boundary:Some(boundary),kind});drop(state);self.wake();Ok(ticket)
    }
    pub fn release(&self,take:u64) {
        if take==0 {return;}
        match self.state() {
            Ok(mut state)=>{state.jobs.push_back(Job {ticket:Ticket {put:None,take:0},boundary:None,kind:Kind::Release {of:take}});drop(state);self.wake();},
            Err(_)=>{self.failed.store(true,Ordering::Release);self.stop();},
        }
    }
    fn wake(&self) {if let Some(worker)=self.worker.get() {worker.unpark();}}
    pub fn stop(&self) {self.stop.store(true,Ordering::Release);self.wake();}
    pub fn failed(&self)->bool {self.failed.load(Ordering::Acquire)}
    pub fn take_stats(&self)->[u64;8] {std::array::from_fn(|index|self.stats[index].swap(0,Ordering::Relaxed))}
    pub fn planes(&self,take:u64)->Option<Planes> {self.state.lock().ok().and_then(|state|state.kept.get(&take).cloned())}
    fn fail(&self,error:String) {self.failed.store(true,Ordering::Release);if let Ok(mut state)=self.state() {if state.error.is_none() {state.error=Some(error);}}self.stop();}
    fn pop(&self)->Result<Option<Job>,String> {Ok(self.state()?.jobs.pop_front())}
    fn two_ahead(&self)->Result<Option<Arc<HostBoundary>>,String> {
        Ok(self.state()?.jobs.iter().filter(|job|matches!(job.kind,Kind::Run {..})).nth(1).and_then(|job|job.boundary.clone()))
    }
}
struct Endpoint<'queue,'mailbox> {host:TpMailboxHost<'mailbox>,queue:&'queue Queue,clean:bool}
impl Drop for Endpoint<'_,'_> {
    fn drop(&mut self) {if !self.clean {self.host.fail();self.queue.fail("host mHC service did not stop cleanly".into());}}
}
fn layout_check(host:&TpMailboxHost<'_>)->Result<(),String> {
    let expected=host.slot_words().checked_mul(TP_MAILBOX_SLOTS).and_then(|words|words.checked_mul(2)).and_then(|words|words.checked_add(HEADER));
    if host.slot_words()==0 || expected!=Some(host.words().len()) {return Err("host mHC service mailbox layout differs".into());}Ok(())
}
fn wait_send(endpoint:&mut Endpoint<'_,'_>,sequence:u64)->Result<(),String> {
    let deadline=Instant::now()+Duration::from_secs(10);
    while endpoint.host.sent(sequence)!=sequence as u32 {
        if endpoint.host.error() {return Err("host mHC mailbox failed before input publication".into());}
        if endpoint.queue.stop.load(Ordering::Acquire) {return Err("host mHC service stopped before input publication".into());}
        if Instant::now()>=deadline {return Err("host mHC input publication timed out".into());}std::hint::spin_loop();
    }Ok(())
}
fn write(host:&TpMailboxHost<'_>,take:u64,values:&[f32])->Result<(),String> {
    if values.len()>host.slot_words() {return Err("host mHC answer exceeds its mailbox slot".into());}
    let base=host.recv_slot(take as usize%TP_MAILBOX_SLOTS);
    let destination=host.words().get(base..base+values.len()).ok_or("host mHC answer exceeds the mailbox")?;
    for (word,value) in destination.iter().zip(values) {word.store(value.to_bits(),Ordering::Relaxed);}Ok(())
}
fn publish(host:&TpMailboxHost<'_>,take:u64) {host.words()[RECV_SEQ+take as usize%TP_MAILBOX_SLOTS].store(take as u32,Ordering::Release);}
fn serve_jobs(endpoint:&mut Endpoint<'_,'_>,helpers:usize)->Result<(),String> {
    layout_check(&endpoint.host)?;
    #[cfg(target_os="linux")] crate::driver::prefer_fast_cores()?;
    let mut pool=None;let mut capacity=0usize;let mut source=Vec::new();let mut idle=Instant::now();let mut publications=0usize;
    loop {
        if endpoint.host.error() {return Err("host mHC mailbox failed".into());}
        let Some(job)=endpoint.queue.pop()? else {
            if endpoint.queue.stop.load(Ordering::Acquire) {return Ok(());}
            if idle.elapsed()>=Duration::from_millis(2) {thread::park_timeout(Duration::from_millis(2));}else {std::hint::spin_loop();}continue;
        };idle=Instant::now();
        if let Kind::Release {of}=job.kind {let mut state=endpoint.queue.state()?;if let Some(planes)=state.kept.remove(&of) {state.free.push(planes);}continue;}
        if endpoint.queue.stop.load(Ordering::Acquire) {return Err("host mHC service stopped with pending jobs".into());}
        let boundary=job.boundary.ok_or("host mHC job has no boundary")?;
        match job.kind {
            Kind::Run {previous}=>{
                let shape=boundary.shape();
                if pool.is_none() {
                    capacity=endpoint.queue.capacity.load(Ordering::Acquire).max(shape.coefficients());pool=Some(Pool::new(helpers,capacity));
                    #[cfg(target_os="linux")] crate::driver::prefer_fast_core(0)?;
                }
                if shape.coefficients()>capacity {return Err("host mHC coefficient extent exceeds its startup reservation".into());}
                let pool=pool.as_ref().ok_or("host mHC pool is absent")?;pool.prepare(&boundary)?;
                let waiting=Instant::now();let mut ns=[0u64;7];
                let put=job.ticket.put.ok_or("host mHC run job has no put")?;wait_send(endpoint,put)?;let slot=put as usize%TP_MAILBOX_SLOTS;
                ns[0]=waiting.elapsed().as_nanos() as u64;let clock=Instant::now();
                let expected=if previous.is_some() {shape.hidden as usize} else {shape.wide()};
                if expected>endpoint.host.slot_words() || endpoint.host.words()[SEND_LEN+slot].load(Ordering::Relaxed) as usize!=expected.checked_mul(4).ok_or("host mHC input byte count overflow")? {return Err("host mHC input plane length differs".into());}
                let base=endpoint.host.send_slot(slot);source.clear();source.extend(endpoint.host.words()[base..base+expected].iter().map(|word|f32::from_bits(word.load(Ordering::Relaxed))));
                if endpoint.host.words()[SEND_SEQ+slot].load(Ordering::Acquire)!=put as u32 {return Err("host mHC input slot was overwritten".into());}
                let (last,mut planes)={let mut state=endpoint.queue.state()?;
                    let last=match previous {Some(of)=>Some((of,state.kept.remove(&of).ok_or("host mHC previous planes were released")?)),None=>None};
                    if let Some((_,last))=&last {if last.residual.len()!=shape.wide() || last.coefficients.len()!=shape.coefficients() {return Err("host mHC previous plane geometry differs".into());}}
                    let planes=state.free.pop().filter(|planes|planes.residual.len()==shape.wide() && planes.coefficients.len()==shape.coefficients() && planes.output.len()==shape.hidden as usize);
                    (last,planes.unwrap_or_else(||boundary.planes()))};
                ns[1]=clock.elapsed().as_nanos() as u64;
                let ran=pool.run(&boundary,last.as_ref().map(|(_,last)|(&last.residual[..],&last.coefficients[..])),&source,&mut planes);
                ns[2..6].copy_from_slice(&pool.last_ns.get());let before=clock.elapsed().as_nanos() as u64;
                let answered=ran.and_then(|()|write(&endpoint.host,job.ticket.take,&planes.output));
                let mut state=endpoint.queue.state()?;if let Some((of,last))=last {state.kept.insert(of,last);}state.kept.insert(job.ticket.take,planes);drop(state);
                answered?;publish(&endpoint.host,job.ticket.take);publications=1;
                ns[6]=clock.elapsed().as_nanos() as u64-before;endpoint.queue.stats[0].fetch_add(1,Ordering::Relaxed);
                for (sum,part) in endpoint.queue.stats[1..].iter().zip(ns) {sum.fetch_add(part,Ordering::Relaxed);}
                if let Some(ahead)=endpoint.queue.two_ahead()? {pool.prepare(&ahead)?;}
            },
            Kind::Residual {of}|Kind::Coefficients {of}=>{
                if publications>=TP_MAILBOX_SLOTS {return Err("host mHC delivery would overwrite an unconsumed receive slot".into());}
                let state=endpoint.queue.state()?;let planes=state.kept.get(&of).ok_or("host mHC delivery planes were released")?;
                let values=if matches!(job.kind,Kind::Residual {..}) {&planes.residual} else {&planes.coefficients};
                write(&endpoint.host,job.ticket.take,values)?;publish(&endpoint.host,job.ticket.take);publications+=1;
            },Kind::Release {..}=>return Err("host mHC release dispatch differs".into()),
        }
    }
}
pub fn serve(host:TpMailboxHost<'_>,queue:&Queue,helpers:usize)->Result<(),String> {
    let mut endpoint=Endpoint {host,queue,clean:false};if queue.worker.set(thread::current()).is_err() {return Err("host mHC queue already has a service thread".into());}
    let result=serve_jobs(&mut endpoint,helpers);if let Err(error)=&result {endpoint.host.fail();queue.fail(error.clone());}endpoint.clean=result.is_ok();result
}
pub struct Worker<'scope> {queue:Arc<Queue>,handle:Option<ScopedJoinHandle<'scope,Result<(),String>>>}
impl Worker<'_> {
    pub fn finish(mut self)->Result<(),String> {self.queue.stop();self.handle.take().ok_or("host mHC worker is already joined")?.join().map_err(|_|"host mHC worker panicked".to_owned())?}
}
impl Drop for Worker<'_> {fn drop(&mut self) {self.queue.stop();if let Some(handle)=self.handle.take() {let _=handle.join();}}}
pub fn spawn<'scope,'environment>(scope:&'scope Scope<'scope,'environment>,host:TpMailboxHost<'environment>,queue:Arc<Queue>,helpers:usize)->Result<Worker<'scope>,String> {
    let owned=queue.clone();let handle=thread::Builder::new().name("mhc-host".into()).spawn_scoped(scope,move||serve(host,&owned,helpers)).map_err(|error|error.to_string())?;
    Ok(Worker {queue,handle:Some(handle)})
}
#[cfg(test)]
#[path="mhc_host_service_tests.rs"] mod tests;
