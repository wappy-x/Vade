//! Ordered local mailbox work. Plane IDs remain live until a queued release.
use crate::{mhc_host::{HostBoundary,Planes,Pool},tp_mailbox_host::{TpMailboxHost,HEADER,RECV_SEQ,SEND_LEN,SEND_SEQ,TP_MAILBOX_SLOTS}};
use std::{collections::{HashMap,VecDeque},sync::{atomic::{AtomicBool,AtomicU32,AtomicU64,AtomicUsize,Ordering},Arc,Mutex,OnceLock,RwLock},thread::{self,Scope,ScopedJoinHandle,Thread},time::{Duration,Instant}};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Kind {Run {previous:Option<u64>},Residual {of:u64},Coefficients {of:u64},Release {of:u64}}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Ticket {pub put:Option<u64>,pub take:u64}
/// The most rows of one job: one row a chat of a batched decode step.
pub const ROWS_MAX:usize=8;
/// The most workers of a row job: the service thread and four crew threads (fast cores 0 and 1..=4 of its mask).
const ROW_WORKERS:usize=5;
/// The workers of a row job when the service thread may use `cores` cores: one worker a core, five at most. Two workers
/// on one core stop each other: each spins while it waits, so the other runs only when the scheduler takes the core
/// away. Measured on a pair (2026-11-10): thread placement gave the host lane 3 cores, 5 workers ran on them, and a
/// step of 2 chats cost 630 ms against 105 ms.
fn row_workers(cores:usize)->usize {cores.clamp(1,ROW_WORKERS)}
/// The count of the CPUs in a list as the system prints it ("5-7,15").
fn listed_cpus(list:&str)->Option<usize> {
    list.trim().split(',').try_fold(0usize,|count,part| {
        let (first,last)=part.split_once('-').unwrap_or((part,part));
        let (first,last)=(first.trim().parse::<usize>().ok()?,last.trim().parse::<usize>().ok()?);
        (first<=last).then(||count+last-first+1)
    }).filter(|&count|count>0)
}
/// The cores the calling thread may use now. None where the system does not say (then a row job has ROW_WORKERS workers).
fn own_cores()->Option<usize> {
    let status=std::fs::read_to_string("/proc/thread-self/status").ok()?;
    listed_cpus(status.lines().find_map(|line|line.strip_prefix("Cpus_allowed_list:"))?)
}
struct Job {ticket:Ticket,boundary:Option<Arc<HostBoundary>>,kind:Kind,rows:usize}
/// `kept` holds one set of planes a row of the job that made them.
#[derive(Default)]
struct State {jobs:VecDeque<Job>,puts:u64,takes:u64,kept:HashMap<u64,Vec<Planes>>,free:Vec<Planes>,error:Option<String>}
#[derive(Default)]
pub struct Queue {state:Mutex<State>,stop:AtomicBool,failed:AtomicBool,capacity:AtomicUsize,worker:OnceLock<Thread>,stats:[AtomicU64;8]}
impl Queue {
    pub fn new()->Arc<Self> {Arc::new(Self::default())}
    fn state(&self)->Result<std::sync::MutexGuard<'_,State>,String> {self.state.lock().map_err(|_|"host mHC queue lock is poisoned".into())}
    pub fn reserve_coefficients(&self,coefficients:usize)->Result<(),String> {
        if self.worker.get().is_some() {return Err("host mHC coefficient reservation follows service startup".into());}
        self.capacity.fetch_max(coefficients,Ordering::Release);Ok(())
    }
    pub fn push(&self,boundary:Arc<HostBoundary>,kind:Kind)->Result<Ticket,String> {self.push_rows(boundary,kind,1)}
    /// The same job for `rows` rows of a batch (1..=ROWS_MAX): the put holds the rows one after another, and so does
    /// each answer. A row's planes equal the one-row job's planes: the rows do not see each other.
    pub fn push_rows(&self,boundary:Arc<HostBoundary>,kind:Kind,rows:usize)->Result<Ticket,String> {
        if rows==0 || rows>ROWS_MAX {return Err("host mHC row count is outside 1..=8".into());}
        let mut state=self.state()?;
        if self.failed.load(Ordering::Acquire) || self.stop.load(Ordering::Acquire) {return Err(state.error.clone().unwrap_or_else(||"host mHC service is stopped".into()));}
        if matches!(kind,Kind::Release {..}) {return Err("host mHC release jobs come from Queue::release".into());}
        let take=state.takes.checked_add(1).filter(|&sequence|sequence<=u32::MAX as u64).ok_or("host mHC receive sequence exhausted")?;
        let put=if matches!(kind,Kind::Run {..}) {Some(state.puts.checked_add(1).filter(|&sequence|sequence<=u32::MAX as u64).ok_or("host mHC send sequence exhausted")?)} else {None};
        self.capacity.fetch_max(boundary.shape().coefficients(),Ordering::Release);
        let ticket=Ticket {put,take};state.takes=take;if let Some(put)=put {state.puts=put;}
        state.jobs.push_back(Job {ticket,boundary:Some(boundary),kind,rows});drop(state);self.wake();Ok(ticket)
    }
    pub fn release(&self,take:u64) {
        if take==0 {return;}
        match self.state() {
            Ok(mut state)=>{state.jobs.push_back(Job {ticket:Ticket {put:None,take:0},boundary:None,kind:Kind::Release {of:take},rows:0});drop(state);self.wake();},
            Err(_)=>{self.failed.store(true,Ordering::Release);self.stop();},
        }
    }
    fn wake(&self) {if let Some(worker)=self.worker.get() {worker.unpark();}}
    pub fn stop(&self) {self.stop.store(true,Ordering::Release);self.wake();}
    pub fn failed(&self)->bool {self.failed.load(Ordering::Acquire)}
    pub fn take_stats(&self)->[u64;8] {std::array::from_fn(|index|self.stats[index].swap(0,Ordering::Relaxed))}
    pub fn planes(&self,take:u64)->Option<Planes> {self.state.lock().ok().and_then(|state|state.kept.get(&take).and_then(|rows|rows.first().cloned()))}
    /// The kept planes of every row of the job behind `take`.
    pub fn planes_rows(&self,take:u64)->Option<Vec<Planes>> {self.state.lock().ok().and_then(|state|state.kept.get(&take).cloned())}
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
fn write(host:&TpMailboxHost<'_>,take:u64,values:&[f32])->Result<(),String> {write_at(host,take,0,values)}
/// `values` at word `offset` of the receive slot of `take`.
fn write_at(host:&TpMailboxHost<'_>,take:u64,offset:usize,values:&[f32])->Result<(),String> {
    if offset.checked_add(values.len()).is_none_or(|end|end>host.slot_words()) {return Err("host mHC answer exceeds its mailbox slot".into());}
    let base=host.recv_slot(take as usize%TP_MAILBOX_SLOTS)+offset;
    let destination=host.words().get(base..base+values.len()).ok_or("host mHC answer exceeds the mailbox")?;
    for (word,value) in destination.iter().zip(values) {word.store(value.to_bits(),Ordering::Relaxed);}Ok(())
}
fn publish(host:&TpMailboxHost<'_>,take:u64) {host.words()[RECV_SEQ+take as usize%TP_MAILBOX_SLOTS].store(take as u32,Ordering::Release);}
/// One row of a row job: its last planes, its planes and its input words.
#[derive(Default)]
struct RowSlot {last:Option<Planes>,planes:Option<Planes>,source:Vec<f32>}
/// What the row workers read. `run` false: read the boundary's weights into the core's cache until the next
/// order (the input is not here yet). `run` true: the rows. `sequence` is the order this job belongs to; `expected`
/// is the words a row brings; `send` and `recv` are where the rows start in the mailbox.
struct RowJob {sequence:u32,run:bool,boundary:Arc<HostBoundary>,rows:usize,expected:usize,send:usize,recv:usize}
/// The row workers. Row r belongs to worker r % `workers` (worker 0 is the service thread). A worker reads its
/// row from the send slot, runs the whole boundary on its core (HostBoundary::run, the one-core twin of the kernels)
/// and writes its output row to the receive slot. Crew threads sleep until the first row job and after 2 ms without one.
struct Crew {workers:usize,sequence:AtomicU32,done:AtomicU32,stop:AtomicBool,gone:AtomicBool,job:RwLock<Option<RowJob>>,slots:Vec<Mutex<RowSlot>>,error:Mutex<Option<String>>,threads:Mutex<Vec<Thread>>}
impl Crew {
    fn new(workers:usize)->Self {Self {workers,sequence:AtomicU32::new(0),done:AtomicU32::new(0),stop:AtomicBool::new(false),gone:AtomicBool::new(false),job:RwLock::new(None),slots:(0..ROWS_MAX).map(|_|Mutex::new(RowSlot::default())).collect(),error:Mutex::new(None),threads:Mutex::new(Vec::new())}}
    fn fail(&self,error:String) {if let Ok(mut saved)=self.error.lock() {if saved.is_none() {*saved=Some(error);}}}
    fn check(&self)->Result<(),String> {match &*self.error.lock().map_err(|_|"host mHC crew error lock is poisoned")? {Some(error)=>Err(error.clone()),None=>Ok(())}}
    /// The rows of `worker` in the published run job.
    fn work(&self,memory:&[std::sync::atomic::AtomicU32],worker:usize)->Result<(),String> {
        let guard=self.job.read().map_err(|_|"host mHC row job lock is poisoned")?;let job=guard.as_ref().ok_or("host mHC crew has no row job")?;
        let hidden=job.boundary.shape().hidden as usize;
        for row in (worker..job.rows).step_by(self.workers) {
            let mut slot=self.slots[row].lock().map_err(|_|"host mHC row slot lock is poisoned")?;let slot=&mut *slot;
            let from=memory.get(job.send+row*job.expected..job.send+(row+1)*job.expected).ok_or("host mHC row input exceeds the mailbox")?;
            slot.source.clear();slot.source.extend(from.iter().map(|word|f32::from_bits(word.load(Ordering::Relaxed))));
            let planes=slot.planes.as_mut().ok_or("host mHC row has no planes")?;
            job.boundary.run(slot.last.as_ref().map(|last|(&last.residual[..],&last.coefficients[..])),&slot.source,planes);
            let to=memory.get(job.recv+row*hidden..job.recv+(row+1)*hidden).ok_or("host mHC row answer exceeds the mailbox")?;
            for (word,value) in to.iter().zip(&planes.output) {word.store(value.to_bits(),Ordering::Relaxed);}
        }Ok(())
    }
    fn member(&self,memory:&[std::sync::atomic::AtomicU32],worker:usize) {
        // Best effort. The crew has no more members than this mask has cores, so no two workers take one core.
        #[cfg(target_os="linux")] let _=crate::driver::prefer_fast_core(worker);
        struct Gone<'a>(&'a AtomicBool);
        impl Drop for Gone<'_> {fn drop(&mut self) {self.0.store(true,Ordering::Release);}}
        let _gone=Gone(&self.gone);let (mut seen,mut idle)=(0u32,Instant::now());
        loop {
            let now=self.sequence.load(Ordering::Acquire);
            if now==seen {
                if self.stop.load(Ordering::Acquire) {return;}
                if idle.elapsed()>=Duration::from_millis(2) {thread::park();idle=Instant::now();} else {std::hint::spin_loop();}
                continue;
            }
            seen=now;
            // A job of a later order is left to the next turn of this loop.
            let order=match self.job.read() {Ok(guard)=>guard.as_ref().filter(|job|job.sequence==now).map(|job|(job.run,job.rows,job.boundary.clone())),Err(_)=>None};
            match order {
                Some((true,..))=>{if let Err(error)=self.work(memory,worker) {self.fail(error);}self.done.fetch_add(1,Ordering::Release);},
                Some((false,rows,boundary)) if worker<rows=>{std::hint::black_box(boundary.touch_while(||self.sequence.load(Ordering::Acquire)==now));},
                _=>{},
            }idle=Instant::now();
        }
    }
    fn wake(&self) {if let Ok(threads)=self.threads.lock() {for thread in threads.iter() {thread.unpark();}}}
    /// Give the crew its next order. Only the service thread calls this.
    fn post(&self,mut job:RowJob)->Result<(),String> {
        let next=self.sequence.load(Ordering::Acquire).wrapping_add(1);job.sequence=next;if job.run {self.done.store(0,Ordering::Release);}
        *self.job.write().map_err(|_|"host mHC row job lock is poisoned")?=Some(job);self.sequence.store(next,Ordering::Release);self.wake();Ok(())
    }
    /// Run the rows on every worker; the caller is worker 0.
    fn run(&self,memory:&[std::sync::atomic::AtomicU32],job:RowJob)->Result<(),String> {
        self.check()?;self.post(job)?;let own=self.work(memory,0);
        while self.done.load(Ordering::Acquire)!=(self.workers-1) as u32 {if self.gone.load(Ordering::Acquire) {return Err("host mHC row worker stopped".into());}std::hint::spin_loop();}
        own?;self.check()
    }
}
fn serve_jobs(endpoint:&mut Endpoint<'_,'_>,helpers:usize,workers:Option<usize>)->Result<(),String> {
    layout_check(&endpoint.host)?;
    // Before any thread starts: the crew and the pool helpers each take their own fast core out of this mask.
    #[cfg(target_os="linux")] crate::driver::prefer_fast_cores()?;
    // One row worker a core of this thread's mask (after the line above: its fast cores). `workers`: a test's own count.
    let workers=workers.unwrap_or_else(||own_cores().map_or(ROW_WORKERS,row_workers));
    let (crew,memory)=(Crew::new(workers),endpoint.host.words());
    thread::scope(|scope| {
        let mut started=Ok(());
        for worker in 1..workers {
            let crew=&crew;
            match thread::Builder::new().name(format!("mhc-row-{worker}")).spawn_scoped(scope,move||crew.member(memory,worker)) {
                Ok(handle)=>{if let Ok(mut threads)=crew.threads.lock() {threads.push(handle.thread().clone());}},
                Err(error)=>{started=Err(error.to_string());break;},
            }
        }
        let result=started.and_then(|()|serve_loop(endpoint,helpers,&crew));
        crew.stop.store(true,Ordering::Release);crew.wake();result
    })
}
/// The helper pool, made at the first run job of either kind. The helpers take their cores first; then this thread
/// takes fast core 0. A row job does not use the helpers, and they sleep.
fn pool_for<'a>(pool:&'a mut Option<Pool>,capacity:&mut usize,queue:&Queue,coefficients:usize,helpers:usize)->Result<&'a Pool,String> {
    if pool.is_none() {
        *capacity=queue.capacity.load(Ordering::Acquire).max(coefficients);*pool=Some(Pool::new(helpers,*capacity));
        #[cfg(target_os="linux")] crate::driver::prefer_fast_core(0)?;
    }
    if coefficients>*capacity {return Err("host mHC coefficient extent exceeds its startup reservation".into());}
    pool.as_ref().ok_or_else(||"host mHC pool is absent".into())
}
fn serve_loop(endpoint:&mut Endpoint<'_,'_>,helpers:usize,crew:&Crew)->Result<(),String> {
    let memory=endpoint.host.words();
    let mut pool=None;let mut capacity=0usize;let mut source=Vec::new();let mut idle=Instant::now();let mut publications=0usize;
    loop {
        if endpoint.host.error() {return Err("host mHC mailbox failed".into());}
        let Some(job)=endpoint.queue.pop()? else {
            if endpoint.queue.stop.load(Ordering::Acquire) {return Ok(());}
            if idle.elapsed()>=Duration::from_millis(2) {thread::park_timeout(Duration::from_millis(2));}else {std::hint::spin_loop();}continue;
        };idle=Instant::now();
        if let Kind::Release {of}=job.kind {let mut state=endpoint.queue.state()?;if let Some(planes)=state.kept.remove(&of) {state.free.extend(planes);}continue;}
        if endpoint.queue.stop.load(Ordering::Acquire) {return Err("host mHC service stopped with pending jobs".into());}
        let boundary=job.boundary.ok_or("host mHC job has no boundary")?;
        match job.kind {
            Kind::Run {previous} if job.rows>1=>{
                // A batch: one row a chat. Each worker runs whole rows on its own core.
                let shape=boundary.shape();let (rows,hidden)=(job.rows,shape.hidden as usize);
                pool_for(&mut pool,&mut capacity,endpoint.queue,shape.coefficients(),helpers)?;
                let put=job.ticket.put.ok_or("host mHC run job has no put")?;let slot=put as usize%TP_MAILBOX_SLOTS;
                let expected=if previous.is_some() {hidden} else {shape.wide()};
                let (send,recv)=(endpoint.host.send_slot(slot),endpoint.host.recv_slot(job.ticket.take as usize%TP_MAILBOX_SLOTS));
                // Until the input is here (the device runs the stage before this boundary), every worker with a row
                // reads the weights into its cache.
                crew.post(RowJob {sequence:0,run:false,boundary:boundary.clone(),rows,expected,send,recv})?;
                std::hint::black_box(boundary.touch_while(||endpoint.host.words()[SEND_SEQ+slot].load(Ordering::Acquire)!=put as u32));
                let waiting=Instant::now();let mut ns=[0u64;7];wait_send(endpoint,put)?;
                ns[0]=waiting.elapsed().as_nanos() as u64;let clock=Instant::now();
                let words=rows.checked_mul(expected).filter(|&words|words<=endpoint.host.slot_words() && rows*hidden<=endpoint.host.slot_words()).ok_or("host mHC row job exceeds its mailbox slot")?;
                if endpoint.host.words()[SEND_LEN+slot].load(Ordering::Relaxed) as usize!=words.checked_mul(4).ok_or("host mHC input byte count overflow")? {return Err("host mHC row input length differs".into());}
                let last={let mut state=endpoint.queue.state()?;
                    let last=match previous {Some(of)=>Some((of,state.kept.remove(&of).ok_or("host mHC previous planes were released")?)),None=>None};
                    if let Some((_,last))=&last {if last.len()!=rows || last.iter().any(|last|last.residual.len()!=shape.wide() || last.coefficients.len()!=shape.coefficients()) {return Err("host mHC previous row planes differ".into());}}
                    let mut planes=Vec::with_capacity(rows);
                    for _ in 0..rows {planes.push(state.free.pop().filter(|planes|planes.residual.len()==shape.wide() && planes.coefficients.len()==shape.coefficients() && planes.output.len()==hidden).unwrap_or_else(||boundary.planes()));}
                    let (of,mut last)=match last {Some((of,last))=>(Some(of),last.into_iter().map(Some).collect::<Vec<_>>()),None=>(None,(0..rows).map(|_|None).collect())};
                    for (row,planes) in planes.into_iter().enumerate() {let mut slot=crew.slots[row].lock().map_err(|_|"host mHC row slot lock is poisoned")?;slot.last=last[row].take();slot.planes=Some(planes);}
                    of};
                ns[1]=clock.elapsed().as_nanos() as u64;
                let ran=crew.run(memory,RowJob {sequence:0,run:true,boundary:boundary.clone(),rows,expected,send,recv});ns[2]=clock.elapsed().as_nanos() as u64-ns[1];let before=clock.elapsed().as_nanos() as u64;
                let overwritten=endpoint.host.words()[SEND_SEQ+slot].load(Ordering::Acquire)!=put as u32;
                let (mut kept_last,mut kept)=(Vec::with_capacity(rows),Vec::with_capacity(rows));
                for row in 0..rows {let mut slot=crew.slots[row].lock().map_err(|_|"host mHC row slot lock is poisoned")?;if let Some(last)=slot.last.take() {kept_last.push(last);}kept.push(slot.planes.take().ok_or("host mHC row planes are absent")?);}
                let mut state=endpoint.queue.state()?;if let Some(of)=last {state.kept.insert(of,kept_last);}state.kept.insert(job.ticket.take,kept);drop(state);
                ran?;if overwritten {return Err("host mHC input slot was overwritten".into());}
                publish(&endpoint.host,job.ticket.take);publications=1;
                ns[6]=clock.elapsed().as_nanos() as u64-before;endpoint.queue.stats[0].fetch_add(1,Ordering::Relaxed);
                for (sum,part) in endpoint.queue.stats[1..].iter().zip(ns) {sum.fetch_add(part,Ordering::Relaxed);}
            },
            Kind::Run {previous}=>{
                let shape=boundary.shape();
                let pool=pool_for(&mut pool,&mut capacity,endpoint.queue,shape.coefficients(),helpers)?;pool.prepare(&boundary)?;
                let waiting=Instant::now();let mut ns=[0u64;7];
                let put=job.ticket.put.ok_or("host mHC run job has no put")?;wait_send(endpoint,put)?;let slot=put as usize%TP_MAILBOX_SLOTS;
                ns[0]=waiting.elapsed().as_nanos() as u64;let clock=Instant::now();
                let expected=if previous.is_some() {shape.hidden as usize} else {shape.wide()};
                if expected>endpoint.host.slot_words() || endpoint.host.words()[SEND_LEN+slot].load(Ordering::Relaxed) as usize!=expected.checked_mul(4).ok_or("host mHC input byte count overflow")? {return Err("host mHC input plane length differs".into());}
                let base=endpoint.host.send_slot(slot);source.clear();source.extend(endpoint.host.words()[base..base+expected].iter().map(|word|f32::from_bits(word.load(Ordering::Relaxed))));
                if endpoint.host.words()[SEND_SEQ+slot].load(Ordering::Acquire)!=put as u32 {return Err("host mHC input slot was overwritten".into());}
                let (last,mut planes)={let mut state=endpoint.queue.state()?;
                    let last=match previous {Some(of)=>{let mut rows=state.kept.remove(&of).ok_or("host mHC previous planes were released")?;
                        if rows.len()!=1 {return Err("host mHC previous planes hold another row count".into());}Some((of,rows.remove(0)))},None=>None};
                    if let Some((_,last))=&last {if last.residual.len()!=shape.wide() || last.coefficients.len()!=shape.coefficients() {return Err("host mHC previous plane geometry differs".into());}}
                    let planes=state.free.pop().filter(|planes|planes.residual.len()==shape.wide() && planes.coefficients.len()==shape.coefficients() && planes.output.len()==shape.hidden as usize);
                    (last,planes.unwrap_or_else(||boundary.planes()))};
                ns[1]=clock.elapsed().as_nanos() as u64;
                let ran=pool.run(&boundary,last.as_ref().map(|(_,last)|(&last.residual[..],&last.coefficients[..])),&source,&mut planes);
                ns[2..6].copy_from_slice(&pool.last_ns.get());let before=clock.elapsed().as_nanos() as u64;
                let answered=ran.and_then(|()|write(&endpoint.host,job.ticket.take,&planes.output));
                let mut state=endpoint.queue.state()?;if let Some((of,last))=last {state.kept.insert(of,vec![last]);}state.kept.insert(job.ticket.take,vec![planes]);drop(state);
                answered?;publish(&endpoint.host,job.ticket.take);publications=1;
                ns[6]=clock.elapsed().as_nanos() as u64-before;endpoint.queue.stats[0].fetch_add(1,Ordering::Relaxed);
                for (sum,part) in endpoint.queue.stats[1..].iter().zip(ns) {sum.fetch_add(part,Ordering::Relaxed);}
                if let Some(ahead)=endpoint.queue.two_ahead()? {pool.prepare(&ahead)?;}
            },
            Kind::Residual {of}|Kind::Coefficients {of}=>{
                if publications>=TP_MAILBOX_SLOTS {return Err("host mHC delivery would overwrite an unconsumed receive slot".into());}
                let state=endpoint.queue.state()?;let rows=state.kept.get(&of).ok_or("host mHC delivery planes were released")?;
                if rows.len()!=job.rows {return Err("host mHC delivery row count differs".into());}
                let mut offset=0;
                for planes in rows {let values=if matches!(job.kind,Kind::Residual {..}) {&planes.residual} else {&planes.coefficients};write_at(&endpoint.host,job.ticket.take,offset,values)?;offset+=values.len();}
                publish(&endpoint.host,job.ticket.take);publications+=1;
            },Kind::Release {..}=>return Err("host mHC release dispatch differs".into()),
        }
    }
}
pub fn serve(host:TpMailboxHost<'_>,queue:&Queue,helpers:usize)->Result<(),String> {serve_rows(host,queue,helpers,None)}
/// `workers`: None in service; a test names the row worker count.
fn serve_rows(host:TpMailboxHost<'_>,queue:&Queue,helpers:usize,workers:Option<usize>)->Result<(),String> {
    let mut endpoint=Endpoint {host,queue,clean:false};if queue.worker.set(thread::current()).is_err() {return Err("host mHC queue already has a service thread".into());}
    let result=serve_jobs(&mut endpoint,helpers,workers);if let Err(error)=&result {endpoint.host.fail();queue.fail(error.clone());}endpoint.clean=result.is_ok();result
}
pub struct Worker<'scope> {queue:Arc<Queue>,handle:Option<ScopedJoinHandle<'scope,Result<(),String>>>}
impl Worker<'_> {
    pub fn finish(mut self)->Result<(),String> {self.queue.stop();self.handle.take().ok_or("host mHC worker is already joined")?.join().map_err(|_|"host mHC worker panicked".to_owned())?}
}
impl Drop for Worker<'_> {fn drop(&mut self) {self.queue.stop();if let Some(handle)=self.handle.take() {let _=handle.join();}}}
pub fn spawn<'scope,'environment>(scope:&'scope Scope<'scope,'environment>,host:TpMailboxHost<'environment>,queue:Arc<Queue>,helpers:usize)->Result<Worker<'scope>,String> {
    spawn_rows(scope,host,queue,helpers,None)
}
fn spawn_rows<'scope,'environment>(scope:&'scope Scope<'scope,'environment>,host:TpMailboxHost<'environment>,queue:Arc<Queue>,helpers:usize,workers:Option<usize>)
    ->Result<Worker<'scope>,String> {
    let owned=queue.clone();
    let handle=thread::Builder::new().name("mhc-host".into()).spawn_scoped(scope,move||serve_rows(host,&owned,helpers,workers)).map_err(|error|error.to_string())?;
    Ok(Worker {queue,handle:Some(handle)})
}
#[cfg(test)]
#[path="mhc_host_service_tests.rs"] mod tests;
