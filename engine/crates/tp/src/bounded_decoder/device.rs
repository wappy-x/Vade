//! Device values retain logical empty shapes with one private unused lane.
use super::*;
use spark_gpu_probe::driver::{Backend,BackendStats,DeviceVector,MhcBoundaryPass,MhcHostLink,ResidentMhcBoundary};
use std::{cell::{Cell,RefCell},rc::Rc};
use super::lineage::Lineage;
pub struct Device<'b,'d> {pub backend:&'b Backend<'d>,weights:Rc<RefCell<crate::resident_weights::Weights<'b,'d>>>,
    #[cfg(feature="kernel-bench")] pub(super) probe:Rc<RefCell<Option<super::projection_probe::Probe>>>,
    #[cfg(feature="kernel-bench")] attention_calls:Rc<Cell<u64>>,
    parameters:Rc<RefCell<parameters::ParameterCache<DeviceVector<'d>>>>,
    norms:Rc<RefCell<BTreeMap<String,spark_gpu_probe::driver::ResidentNorm<'b,'d>>>>,
    blocks:Rc<RefCell<BTreeMap<(usize,usize),ResidentMhcBoundary<'b,'d>>>>,mhc_host:Option<MhcHostLink<'b,'d>>,
    stages:Rc<RefCell<[[u128;2];4]>>,cost:Rc<crate::resident_profile::Cost>,memo:RefCell<Memo<'d>>,copies:RefCell<Option<DeviceVector<'d>>>,selection_checks:RefCell<Vec<(Rc<DeviceVector<'d>>,f32)>>,selection_read:RefCell<Option<(std::rc::Weak<DeviceVector<'d>>,Vec<f32>)>>,selection_counts:Cell<[u64;2]>,selection_shortcut:Cell<bool>,screen_nan:Cell<Option<u32>>,screen_keep:Cell<bool>,source:Rc<Cell<u64>>,programs:RefCell<BTreeMap<(Vec<u32>,u64),spark_gpu_probe::driver::ResidentProgram<'b,'d>>>,
    row_workspace:RefCell<Option<DeviceVector<'d>>>,small_rows:Cell<bool>,
    serving:Option<RefCell<Option<(u64,std::time::Instant,BackendStats)>> >,state_copies:Cell<usize>}
/// A projection's rows in the memo: its name, its rounding, and the topology (rank, parts).
type Group=(String,bool,(u32,u32));
type OriginKey=(Group,Lineage);
/// The key as the memo charges it: one flat tuple of a name, a span list with its word count,
/// the rounding and the topology. The charge of a row follows this shape, not the shape of the
/// index or of a lineage in host memory, so the memo keeps the same rows when those change.
type ChargedKey=(String,(Vec<u8>,usize),bool,(u32,u32));
#[cfg(target_pointer_width="64")]
const _:()=assert!(std::mem::size_of::<ChargedKey>()==72);
/// The rows of one projection share one allocation, so they leave the memo together.
use super::memo_batches::{self,Batch,Index};
#[derive(Default)]
struct Memo<'d> {origins:memo_batches::Origins<Group,Lineage,Rc<HeldStorage<'d>>>,batches:BTreeMap<u64,Batch<OriginKey>>,unverified:Vec<u64>,next:u64,clock:u64,bytes:usize,limit:usize,hits:u64,misses:u64,slab:Slab<'d>}
/// Rows of one slab chunk: one allocation, 8 MiB at GLM's 131,072-byte expanded row.
pub const SLAB_ROWS:usize=64;
/// The memo's own storage for kept rows: chunks of SLAB_ROWS rows, one slot a row. Every
/// memo entry and every live value of a kept row holds its slot's handle (a range of such
/// a value clones it too), so a slot whose only handle is this list is read by nobody and
/// may be written again. All chunks together stay within the memo's charged limit.
#[derive(Default)]
struct Slab<'d> {slots:Vec<Rc<HeldStorage<'d>>>,next:usize,bytes:usize}
impl<'d> Slab<'d> {
    /// Up to `count` slots of `width` words that nobody reads. A new chunk is allocated only
    /// while every chunk fits `limit`; past it fewer slots come back and those rows are not kept.
    fn take(&mut self,backend:&Backend<'d>,width:usize,count:usize,limit:usize)->Result<Vec<Rc<HeldStorage<'d>>>> {
        let mut found=Vec::with_capacity(count);let total=self.slots.len();let mut looked=0;
        while found.len()<count && looked<total {
            let at=(self.next+looked)%total;looked+=1;let slot=&self.slots[at];
            if slot.capacity==width && Rc::strong_count(slot)==1 {found.push(slot.clone());}
        }
        if total>0 {self.next=(self.next+looked)%total;}
        // A chunk is allocated in whole allocator pages and charged at that size, so the
        // allocator never holds more than the slab counts against the limit.
        let bytes=super::held::slab_chunk_bytes(SLAB_ROWS,width).ok_or("projection slab size overflow")?;let words=bytes/4;
        while found.len()<count && width>0 && self.bytes.checked_add(bytes).is_some_and(|n|n<=limit) {
            let chunk=backend.vector(words)?.into_resident_shared()?;self.bytes+=bytes;
            for row in 0..SLAB_ROWS {
                let whole=chunk.resident_range(row*width..(row+1)*width)?;let view=Rc::new(whole.resident_range(0..width)?);
                let slot=Rc::new(HeldStorage {whole,capacity:width,views:Default::default(),appendable:false,row:Some(view.clone())});slot.views.add(&view,width);
                if found.len()<count {found.push(slot.clone());}self.slots.push(slot);
            }
        }
        Ok(found)
    }
}
impl Clone for Device<'_,'_> {
    fn clone(&self)->Self {Self {backend:self.backend,weights:self.weights.clone(),parameters:self.parameters.clone(),norms:self.norms.clone(),
        #[cfg(feature="kernel-bench")] probe:self.probe.clone(),
        #[cfg(feature="kernel-bench")] attention_calls:self.attention_calls.clone(),
        blocks:self.blocks.clone(),mhc_host:self.mhc_host.clone(),stages:self.stages.clone(),cost:self.cost.clone(),memo:RefCell::new(Memo::default()),copies:RefCell::new(None),selection_checks:RefCell::new(Vec::new()),selection_read:RefCell::new(None),selection_counts:Cell::new([0;2]),selection_shortcut:Cell::new(true),screen_nan:Cell::new(None),screen_keep:Cell::new(false),row_workspace:RefCell::new(None),small_rows:Cell::new(false),source:self.source.clone(),programs:RefCell::new(BTreeMap::new()),serving:self.serving.as_ref().map(|_|RefCell::new(None)),state_copies:Cell::new(0)}}
}
struct Pending<'b,'d> {post:Option<(MhcBoundaryPass<'b,'d>,Rc<DeviceVector<'d>>)>,error:Option<String>,state_prefix:Option<usize>}
/// Held rows of one chat and layer: one allocation with room. Every view of it, whatever its
/// start, is listed with the storage word where it ends, and is live as long as its device
/// vector has a handle. An append may write only past the last live end; else it copies.
struct HeldStorage<'d> {whole:DeviceVector<'d>,capacity:usize,views:super::held::Ends<DeviceVector<'d>>,
    /// Cache rows may grow in place; a memo slot is immutable while any handle lives.
    appendable:bool,
    /// A memo slot's one view of its whole row. Every hit of the row shares it: a selection
    /// reads thousands of kept rows in a decode step, and a view a hit would cost each one
    /// two allocations. Held cache rows have none; their views differ in extent.
    row:Option<Rc<DeviceVector<'d>>>}
pub struct Value<'b,'d> {vector:Rc<DeviceVector<'d>>,words:usize,lineage:Lineage,known:RefCell<Option<Vec<f32>>>,assumed:Cell<bool>,pending:Option<Rc<RefCell<Pending<'b,'d>>>>,
    /// For a view of held rows: the storage and the storage word where the view starts.
    held:Option<(Rc<HeldStorage<'d>>,usize)>}
impl Value<'_,'_> {
    /// The device address of the first word. A check tells a view of held rows from a copy by it.
    pub fn device_address(&self)->u64 {self.vector.device_address()}
}
impl<'b,'d> Device<'b,'d> {
    #[cfg(feature="kernel-bench")]
    pub fn fixture_attention_calls(&self)->u64 {self.attention_calls.get()}
    #[cfg(feature="kernel-bench")]
    pub fn fixture_reset(&self)->Result<()> {
        self.backend.synchronize()?;
        self.probe.borrow_mut().take();self.row_workspace.borrow_mut().take();self.small_rows.set(false);
        self.programs.borrow_mut().clear();*self.memo.borrow_mut()=Memo::default();self.copies.borrow_mut().take();self.selection_read.borrow_mut().take();
        self.norms.borrow_mut().clear();self.blocks.borrow_mut().clear();self.parameters.borrow_mut().clear();
        self.backend.fixture_reset_resident()
    }
    #[cfg(feature="kernel-bench")]
    pub fn fixture_vote(&self,ok:bool)->Result<bool> {self.weights.borrow_mut().fixture_vote(ok)}
    pub fn new(backend:&'b Backend<'d>)->Self {Self::with_ring(backend,None)}
    pub fn with_ring(backend:&'b Backend<'d>,ring:Option<crate::transport::Ring>)->Self {Self::with_ring_profiled(backend,ring,false)}
    /// Diagnostic runner counters. Normal construction keeps them disabled.
    pub fn with_ring_profiled(backend:&'b Backend<'d>,ring:Option<crate::transport::Ring>,cost:bool)->Self {
        Self {backend,weights:Rc::new(RefCell::new(crate::resident_weights::Weights::new(ring,cost))),parameters:Rc::new(RefCell::new(parameters::ParameterCache::default())),
            #[cfg(feature="kernel-bench")] probe:Rc::new(RefCell::new(None)),
            #[cfg(feature="kernel-bench")] attention_calls:Rc::new(Cell::new(0)),
            norms:Rc::new(RefCell::new(BTreeMap::new())),blocks:Rc::new(RefCell::new(BTreeMap::new())),mhc_host:None,stages:Rc::new(RefCell::new([[0;2];4])),cost:Rc::new(crate::resident_profile::Cost::new(cost)),memo:RefCell::new(Memo::default()),copies:RefCell::new(None),selection_checks:RefCell::new(Vec::new()),selection_read:RefCell::new(None),selection_counts:Cell::new([0;2]),selection_shortcut:Cell::new(true),screen_nan:Cell::new(None),screen_keep:Cell::new(false),row_workspace:RefCell::new(None),small_rows:Cell::new(false),source:Rc::new(Cell::new(0)),programs:RefCell::new(BTreeMap::new()),serving:None,state_copies:Cell::new(0)}
    }
    /// Wall counters only. This does not enable CUDA event profiling.
    pub fn with_serving_trace(mut self,enabled:bool)->Self {self.serving=enabled.then(||RefCell::new(None));self}
    pub fn with_mhc_host(mut self,link:MhcHostLink<'b,'d>)->Result<Self> {
        if !self.blocks.borrow().is_empty() || self.mhc_host.is_some() {return Err("host mHC link requires an unprepared boundary cache".into());}
        self.weights.borrow_mut().attach_mhc_host(link.clone())?;self.mhc_host=Some(link);Ok(self)
    }
    fn upload_boundary(&self,parameters:&crate::layer_operands::ResidualBlockOperands)->Result<ResidentMhcBoundary<'b,'d>> {
        match &self.mhc_host {Some(link)=>self.backend.resident_mhc_boundary_hosted(parameters.mhc().shape(),parameters.norm().shape(),parameters.mhc().weights(),parameters.mhc().base(),parameters.mhc().scale(),parameters.norm().weight(),link),None=>parameters.upload_boundary(self.backend)}
    }
    /// The caller keeps the attached wire and mailbox alive through all steps.
    pub fn with_mailbox(backend:&'b Backend<'d>,ring:crate::transport::Ring,cost:bool,mailbox:&'b spark_gpu_probe::driver::TpMailbox<'d>)->Result<Self> {
        let device=Self::with_ring_profiled(backend,Some(ring),cost);device.weights.borrow_mut().attach_mailbox(mailbox)?;Ok(device)
    }
    // Device consumers share the backend's ordered stream. Readback waits, and
    // Buffer::drop drains pending work before a source or destination is freed.
    // Keep owners here; a value wrapper needs no separate completion barrier.
    fn value(&self,vector:DeviceVector<'d>,words:usize)->Result<Value<'b,'d>> {self.shared_value(Rc::new(vector.into_resident_shared()?),words)}
    fn shared_value(&self,vector:Rc<DeviceVector<'d>>,words:usize)->Result<Value<'b,'d>> {let source=self.source.get().checked_add(1).ok_or("immutable source identity exhausted")?;self.source.set(source);
        Ok(Value {vector,words,lineage:Lineage::fresh(source,words),known:RefCell::new(None),assumed:Cell::new(false),pending:None,held:None})}
    /// A value of one kept row. A kept row that cannot be viewed is a fault on this rank, not
    /// a miss. The value holds its slot's handle, so the slab does not write the slot while it lives.
    fn kept_row(&self,slot:Rc<HeldStorage<'d>>)->Result<Value<'b,'d>> {
        let words=slot.capacity;
        let vector=match &slot.row {Some(row)=>row.clone(),None=>{let view=Rc::new(slot.whole.resident_range(0..words)?);slot.views.add(&view,words);view}};
        let mut value=self.shared_value(vector,words)?;value.held=Some((slot,0));Ok(value)
    }
    fn take_post(&self,value:&Value<'b,'d>)->Result<Option<(MhcBoundaryPass<'b,'d>,Rc<DeviceVector<'d>>)>> {
        let Some(pending)=&value.pending else {return Ok(None);};let mut pending=pending.borrow_mut();
        if let Some(error)=&pending.error {return Err(error.clone());}
        let post=pending.post.take();if post.is_some() {pending.error=Some("deferred mHC residual was consumed by another boundary".into());}Ok(post)
    }
    fn ready(&self,value:&Value<'b,'d>)->Result<()> {
        if let Some((pass,block))=self.take_post(value)? {
            let result=pass.materialize(&block,&value.vector);
            value.pending.as_ref().unwrap().borrow_mut().error=result.as_ref().err().cloned();result?;
        }Ok(())
    }
    /// The selection and its count are complete together: one wait reads both.
    fn read_pair(&self,first:&Value<'b,'d>,second:&Value<'b,'d>)->Result<(Vec<f32>,Vec<f32>)> {
        self.ready(first)?;self.ready(second)?;
        if first.assumed.get() && second.assumed.get() {
            self.weights.borrow().check_exchange()?;if let Some(error)=self.backend.fatal_error() {return Err(error);}
            // No wait and no bytes. Submit the producer without restarting the
            // recorder's ramp: only a real completion makes the device idle.
            self.backend.flush_resident_launches()?;let [waits,known]=self.selection_counts.get();self.selection_counts.set([waits,known+1]);
            let copy=|v:&Value<'b,'d>|v.known.borrow().clone().ok_or_else(||"known selection lost its row".to_string());return Ok((copy(first)?,copy(second)?));
        }
        let [waits,known]=self.selection_counts.get();
        if first.words==0 || second.words==0 || first.known.borrow().is_some() || second.known.borrow().is_some() || first.vector.len()!=first.words || second.vector.len()!=second.words {
            // The ids the last pair read took, with the count it left on its value: no wait, so none is counted.
            let held=second.known.borrow().is_some() && first.known.borrow().is_none() && self.selection_seen(first).is_some();
            if !held {self.selection_counts.set([waits+1,known]);}
            return Ok((Compute::read(self,first)?,Compute::read(self,second)?));}
        self.selection_counts.set([waits+1,known]);
        self.weights.borrow().check_exchange()?;if let Some(error)=self.backend.fatal_error() {return Err(error);}
        self.check_selection(None)?;
        let stamp=self.cost.start(self.backend);let (a,b)=self.backend.read_resident_pair(&first.vector,&second.vector)?;
        self.weights.borrow().check_exchange()?;*first.known.borrow_mut()=Some(a.clone());*second.known.borrow_mut()=Some(b.clone());
        *self.selection_read.borrow_mut()=Some((Rc::downgrade(&first.vector),a.clone()));
        self.cost.record(||"readback".into(),stamp,self.backend,0);Ok((a,b))
    }
    /// The ids of the last pair read, if `v` is the same immutable device vector with the same
    /// extent. A layer's Gather reads its selection through a whole-range view, and the
    /// expansion asks for the same ids through the value itself: both hold one vector, so the
    /// host's copy answers the second one with no wait. A live vector is never written again,
    /// and the weak handle keeps its identity from another vector without keeping its storage.
    fn selection_seen(&self,v:&Value<'b,'d>)->Option<Vec<f32>> {
        self.selection_read.borrow().as_ref().filter(|(seen,row)|seen.strong_count()>0 && std::ptr::eq(seen.as_ptr(),Rc::as_ptr(&v.vector)) && row.len()==v.words && v.vector.len()==v.words).map(|(_,row)|row.clone())
    }
    /// Compare each known selection's count word with the host's value at this real read, so a
    /// NaN score still refuses the request before a result leaves. One join and one wait, shared
    /// with `with` when it can be read in the same wait. Returns `with`'s words if they were read.
    fn check_selection(&self,with:Option<&Value<'b,'d>>)->Result<Option<Vec<f32>>> {
        let checks=std::mem::take(&mut *self.selection_checks.borrow_mut());if checks.is_empty() {return Ok(None);}
        let verified=(|| {
            let parts=checks.iter().map(|(count,_)|(&**count,1usize)).collect::<Vec<_>>();
            let joined=self.backend.resident_join(&parts,&mut self.copies.borrow_mut(),0)?;
            let (out,counts)=match with.filter(|v|v.words>0 && v.vector.len()==v.words) {
                Some(v)=>{let (a,b)=self.backend.read_resident_pair(&v.vector,&joined)?;(Some(a),b)},
                None=>(None,self.backend.read_vector_values(&joined)?),
            };
            need(counts.len()==checks.len(),"known selection check extent differs")?;
            for (got,(_,want)) in counts.iter().zip(&checks) {need(got.to_bits()==want.to_bits(),"selection score is not a number")?;}
            Ok(out)
        })();
        if verified.is_ok() {self.memo.borrow_mut().unverified.clear();} else {self.discard_unverified();}verified
    }
    /// A refused selection's Gather wrote zero rows under the source's lineage. The projections of
    /// those rows are wrong for a clean retry, so every batch kept since the last check goes.
    fn discard_unverified(&self) {
        let mut memo=self.memo.borrow_mut();let memo=&mut *memo;let ids=std::mem::take(&mut memo.unverified);
        if self.screen_keep.get() {return;}
        let freed=memo_batches::discard(&mut memo.origins,&mut memo.batches,&ids);memo.bytes-=freed;
    }
    /// Screens only: make the selection after `after` more selections see NaN scores.
    pub fn screen_nan_selection(&self,after:u32) {self.screen_nan.set(Some(after));}
    /// Screens only: keep unverified memo rows after a refusal, to prove the retry test can fail.
    pub fn screen_keep_unverified(&self,keep:bool) {self.screen_keep.set(keep);}
    /// Screens only: false runs the readback path, so one process can compare both.
    pub fn set_selection_shortcut(&self,on:bool) {self.selection_shortcut.set(on);}
    /// Diagnostic: selection reads that waited, and those known on the host without a wait.
    pub fn selection_counts(&self)->[u64;2] {self.selection_counts.get()}
    /// Diagnostic: kept projection bytes, their bound, and rows found and added since the last report.
    pub fn memo_counts(&self)->(usize,usize,u64,u64) {let memo=self.memo.borrow();(memo.bytes,memo.limit,memo.hits,memo.misses)}
    pub fn prompt_state_copies(&self)->usize {self.state_copies.get()}
    /// Fixture and diagnostic use: a memo byte limit without a checkpoint, at most the charged bound.
    pub fn set_projection_memo_limit(&self,limit:usize)->Result<()> {
        let mut memo=self.memo.borrow_mut();need(limit<=super::PROJECTION_MEMO_BYTES,"memo limit exceeds its charged bound")?;
        need(limit>=memo.slab.bytes && limit>=memo.bytes,"memo limit is below the bytes it already holds")?;memo.limit=limit;Ok(())}
    /// Diagnostic: bytes of the memo's own row chunks (never above its limit).
    pub fn projection_slab_bytes(&self)->usize {self.memo.borrow().slab.bytes}
    pub fn clear_projection_memo(&self)->Result<()> {self.backend.synchronize()?;let mut memo=self.memo.borrow_mut();memo.origins.clear();memo.batches.clear();memo.bytes=0;memo.limit=0;memo.slab=Slab::default();Ok(())}
    fn record(&self,stage:usize,started:std::time::Instant) {let mut stages=self.stages.borrow_mut();stages[stage][0]+=1;stages[stage][1]+=started.elapsed().as_nanos();}
    pub(crate) fn report(&self) {self.weights.borrow_mut().report();self.cost.report();
        let route=spark_gpu_probe::device_route::counts();eprintln!("RESIDENT DEVICE ROUTE gate_up={} down={} host_fallback={}",route[0],route[1],route[2]);
        for (name,timing) in ["residual_begin","residual_finish","standalone_norm","causal_operations"].into_iter().zip(self.stages.borrow_mut().iter_mut()) {
            eprintln!("RESIDENT STAGE name={name} calls={} milliseconds={:.6}",timing[0],timing[1] as f64/1e6);*timing=[0;2];
        }
        let mut memo=self.memo.borrow_mut();eprintln!("RESIDENT MEMO hits={} misses={} bytes={} limit={}",memo.hits,memo.misses,memo.bytes,memo.limit);memo.hits=0;memo.misses=0;
    }
}
impl<'b,'d> Compute for Device<'b,'d> {
    fn work<T>(&self,run:impl FnOnce()->Result<T>)->Result<T> {
        let queue=self.backend.resident_queue()?;self.backend.enable_resident_launches()?;
        let result=run();
        // A scope may end with no real read. Its known selections are still checked.
        let checked=if result.is_ok() {self.check_selection(None).map(|_|())} else {self.selection_checks.borrow_mut().clear();self.discard_unverified();Ok(())};
        drop(queue);
        if let Some(error)=self.backend.fatal_error() {return Err(error);}
        checked?;self.weights.borrow().check_exchange()?;result
    }
    fn resident_weights_ready(&self)->bool {self.weights.borrow().prepared()}
    fn fatal_error(&self)->Option<String> {self.backend.fatal_error().or_else(||self.weights.borrow().check_exchange().err())}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {Some((self.backend.accounted_bytes(),self.backend.peak_accounted_bytes(),self.backend.pinned_host_bytes()))}
    fn vector_allocations(&self)->Option<(usize,usize)> {let s=self.backend.vector_pool_stats();Some((s.allocated_count,s.allocated_bytes))}
    fn vector_pool_counts(&self)->Option<(usize,usize,usize)> {let s=self.backend.vector_pool_stats();Some((s.allocated_count,s.allocated_bytes,s.reused_count))}
    fn serving_trace(&self)->bool {self.serving.is_some()}
    fn profile_tick_begin(&self,tick:u64) {if let Some(trace)=&self.serving {*trace.borrow_mut()=Some((tick,std::time::Instant::now(),self.backend.stats()));}}
    fn profile_tick_end(&self) {if let Some(trace)=&self.serving&& let Some((tick,started,before))=trace.borrow_mut().take() {
        let elapsed=started.elapsed().as_nanos();let s=self.backend.stats().since(before);
        eprintln!("SERVED TICK tick={tick} wall_ns={elapsed} allocation_ns={} wait_ns={} wait_calls={} launch_ns={} launch_calls={} download_ns={} upload_ns={} pack_ns={} graph_calls={}",
            s.allocation_ns,s.stream_wait_ns,s.stream_wait_calls,s.host_launch_ns,s.queued_calls,s.output_download_ns,s.weight_read_upload_ns,s.host_pack_ns,s.graph_launches);
    }}
    fn plan_cost(&self,ns:u128) {self.cost.host("planning",ns);}
    fn trace_start(&self) {if self.cost.enabled() {self.backend.set_profiling(true);}}
    fn cost_report(&self,phase:&str) {if self.cost.enabled() {eprintln!("SERVED COST phase={phase}");self.report();}}
    type Value=Value<'b,'d>;type Pass=MhcBoundaryPass<'b,'d>;
    fn projects_complete_rows(&self)->bool {true}
    fn sparse_topology(&self)->Topology {self.weights.borrow().sparse_topology()}
    fn owned_expansion(&self,name:&str,input:&Value<'b,'d>,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Option<Value<'b,'d>>> {
        self.ready(input)?;
        let output=self.weights.borrow_mut().apply_owned(name,self.backend,&input.vector,tokens,layout)?;
        let words=output.len();Ok(Some(self.value(output,words)?))
    }
    fn gather_attended(&self,input:&Value<'b,'d>,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Value<'b,'d>> {
        self.ready(input)?;
        if layout.local_heads()==layout.total_heads() {return self.range(input,0,tokens*layout.total_heads()*layout.value_dim());}
        let stamp=self.cost.start(self.backend);
        let output=self.weights.borrow_mut().gather_attended(self.backend,&input.vector,tokens,layout)?;
        let words=output.len();let value=self.value(output,words)?;
        self.cost.record(||"attended_gather".into(),stamp,self.backend,0);Ok(value)
    }
    fn owned_expansion_rows(&self,name:&str,input:&Value<'b,'d>,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Option<Value<'b,'d>>> {
        self.ready(input)?;
        let output=self.weights.borrow_mut().apply_owned_rows(name,self.backend,&input.vector,tokens,layout)?;
        let words=output.len();Ok(Some(self.value(output,words)?))
    }
    fn prepare(&self,c:&CheckpointAdmission)->Result<()> {
        // A service can retain this owner while replacing a decoder session.
        // Pure memo entries are shared across live sessions; preparation clears them.
        self.row_workspace.borrow_mut().take();self.small_rows.set(false);
        *self.memo.borrow_mut()=Memo::default();self.weights.borrow_mut().prepare(c,self.backend)?;
        // Reuse completed private vectors under the backend's existing pressure
        // and size limits. A service's earlier memory admission owns its margin.
        if spark_gpu_probe::driver::vector_pool_from_env()? && !self.backend.vector_pool_configured() {self.backend.configure_vector_pool(1<<30);}
        let local=self.backend.available_memory_bytes()?.saturating_sub(1<<30).min(super::PROJECTION_MEMO_BYTES);
        self.memo.borrow_mut().limit=self.weights.borrow_mut().shared_budget(local)?;if self.backend.resident_queue_active() {self.backend.enable_resident_launches()?;}Ok(())}
    fn prepare_names(&self,c:&CheckpointAdmission,names:&[String])->Result<()> {self.weights.borrow_mut().prepare_names(c,self.backend,names)}
    fn projection_origin(&self,name:&str,x:&Value<'b,'d>,round:bool)->Result<Option<Value<'b,'d>>> {
        if self.memo.borrow().limit==0 || x.pending.is_some() {return Ok(None);}
        let topology=self.sparse_topology();let topology=(topology.rank,topology.parts);
        let Some((slot,batch))=self.memo.borrow().origins.group(|group|group.0==name && group.1==round && group.2==topology).and_then(|rows|rows.get(&x.lineage)).cloned() else {return Ok(None);};
        let value=self.kept_row(slot)?;
        let mut memo=self.memo.borrow_mut();memo.hits+=1;let now=memo.clock;if let Some(batch)=memo.batches.get_mut(&batch) {batch.used=now;}Ok(Some(value))
    }
    fn projection_origins(&self,name:&str,rows:&Value<'b,'d>,width:usize,slots:&[usize],round:bool)->Result<Vec<Option<Value<'b,'d>>>> {
        let mut found=Vec::with_capacity(slots.len());if slots.is_empty() {return Ok(found);}
        // What `range` checks for each row, and what it completes before it makes a view.
        for &slot in slots {need(slot.checked_mul(width).and_then(|start|start.checked_add(width)).is_some_and(|end|end<=rows.words),"bounded device range differs")?;}
        let whole=width==rows.words;if !whole {self.ready(rows)?;}
        if self.memo.borrow().limit==0 || (whole && rows.pending.is_some()) {found.resize_with(slots.len(),||None);return Ok(found);}
        let topology=self.sparse_topology();let topology=(topology.rank,topology.parts);
        let mut memo=self.memo.borrow_mut();let memo=&mut *memo;let now=memo.clock;
        // The projection's rows are found once. Each row is then one search by its identity,
        // written into one probe: no row view, no key and no allocation for a row.
        let Some(group)=memo.origins.group(|group|group.0==name && group.1==round && group.2==topology) else {found.resize_with(slots.len(),||None);return Ok(found);};
        let mut probe=Lineage::default();let mut touched=None;
        for &slot in slots {
            probe.clear();rows.lineage.range_into(slot*width,width,&mut probe)?;
            let Some((kept,batch))=group.get(&probe) else {found.push(None);continue;};
            found.push(Some(self.kept_row(kept.clone())?));memo.hits+=1;
            if touched!=Some(*batch) {if let Some(batch)=memo.batches.get_mut(batch) {batch.used=now;}touched=Some(*batch);}
        }
        Ok(found)
    }
    fn remember_projections(&self,name:&str,round:bool,kept:&[(&Value<'b,'d>,&Value<'b,'d>)])->Result<()> {
        let mut memo=self.memo.borrow_mut();if memo.limit==0 || kept.is_empty() {return Ok(());}
        // Each row is charged its input and output words and its key. All rows of
        // one call enter together, so a kept row never holds an uncounted batch.
        let row_bytes=|x:&Value<'b,'d>,output:&Value<'b,'d>|->Result<usize> {
            let key=x.lineage.bytes().checked_add(2*name.len()).and_then(|n|n.checked_add(2*std::mem::size_of::<ChargedKey>()+std::mem::size_of::<Rc<DeviceVector<'d>>>())).ok_or("projection origin byte count overflow")?;
            Ok(output.words.checked_add(x.words).and_then(|n|n.checked_mul(4)).and_then(|n|n.checked_add(key)).ok_or("projection memo byte count overflow")?)};
        let rows=kept.len();let mut total=0usize;
        for (x,output) in kept {total=total.checked_add(row_bytes(x,output)?).ok_or("projection memo byte count overflow")?;}
        if total>memo.limit {return Ok(());}
        // Make room by dropping whole batches, the one used longest ago first.
        // Rows used by the present token stay; a dropped row is expanded again.
        memo.clock+=1;
        while memo.bytes.checked_add(total).is_none_or(|n|n>memo.limit) {
            let Some(oldest)=memo.batches.iter().min_by_key(|(id,batch)|(batch.used,**id)).map(|(id,_)|*id) else {break;};
            let batch=memo.batches.remove(&oldest).ok_or("projection memo batch disappeared")?;
            for key in &batch.keys {if memo.origins.owner(key)==Some(oldest) {memo.origins.forget(key);}}
            memo.bytes-=batch.bytes;
        }
        // Kept rows are copied, raw words, into the memo's own slots with one ordered launch;
        // the projection output is then scratch again. Rows without a free funded slot
        // are not kept (expanded again when next selected), never kept outside the limit.
        let width=kept[0].1.words;if kept.iter().any(|(_,output)|output.words!=width) {return Ok(());}
        let limit=memo.limit;let slots=memo.slab.take(self.backend,width,kept.len(),limit)?;
        let kept=&kept[..slots.len()];if kept.is_empty() {return Ok(());}
        let total=if kept.len()==rows {total} else {kept.iter().map(|(x,output)|row_bytes(x,output)).sum::<Result<usize>>()?};
        for (_,output) in kept {self.ready(output)?;}
        let sources=kept.iter().map(|(_,output)|(&*output.vector,0)).collect::<Vec<_>>();
        let mut targets=slots.iter().map(|slot|slot.whole.resident_range(0..width)).collect::<std::result::Result<Vec<_>,_>>()?;
        self.backend.resident_scatter(&sources,&mut targets,&mut self.copies.borrow_mut())?;drop(targets);
        let id=memo.next;memo.next+=1;let now=memo.clock;let mut keys=Vec::with_capacity(kept.len());
        let topology=self.sparse_topology();let group:Group=(name.to_owned(),round,(topology.rank,topology.parts));
        for ((x,_),slot) in kept.iter().zip(slots) {memo.origins.insert(&group,x.lineage.clone(),(slot,id));keys.push((group.clone(),x.lineage.clone()));}
        // A known selection's count is compared later. Until then, these rows may come from a refused selection.
        if !self.selection_checks.borrow().is_empty() {memo.unverified.push(id);}
        memo.batches.insert(id,Batch {keys,bytes:total,used:now});memo.bytes+=total;memo.misses+=kept.len() as u64;Ok(())
    }
    fn planar_kda(&self,shape:Shape)->bool {matches!(shape.op,Op::StepPlanar|Op::ConvParts) && spark_gpu_probe::causal_ops::fast_admitted(shape,spark_gpu_probe::causal_ops::FAST_THREADS)}
    fn indirect_attention(&self,shape:Shape,parts:&[&Value<'b,'d>],query:&Value<'b,'d>)->Result<Option<Value<'b,'d>>> {
        if !spark_gpu_probe::causal_ops::indirect_attention_proven(shape) {return Ok(None);}
        self.ready(query)?;for value in parts {self.ready(value)?;}
        // ExpandKv stores every row packed (see pack_expanded); the packed entry widens each half in-kernel.
        let stamp=self.cost.start(self.backend);let rows=parts.iter().map(|value|&*value.vector).collect::<Vec<_>>();
        let [_,_,output]=self.backend.resident_packed_attention(shape,&rows,&query.vector)?;
        #[cfg(feature="kernel-bench")] self.attention_calls.set(self.attention_calls.get()+1);
        let words=output.len();let value=self.value(output,words)?;self.cost.record(||"indirect_attention".into(),stamp,self.backend,0);Ok(Some(value))
    }
    fn attention(&self,heads:u32,key_dim:u32,value_dim:u32,slots:usize,capacity:usize,x:[&Value<'b,'d>;4])->Result<Option<Value<'b,'d>>> {
        for value in x {self.ready(value)?;}
        if !self.backend.resident_queue_active() || self.backend.resident_launches_active() {return Ok(None);}
        need(slots>0 && slots<=capacity && capacity<=u32::MAX as usize,"attention capacity differs")?;
        let expected=[heads as usize*key_dim as usize,slots*heads as usize*key_dim as usize,slots*heads as usize*value_dim as usize,1];
        for (v,len) in x.iter().zip(expected) {need(v.words==len,"attention piece input extent differs")?;}
        let count=self.read(x[3])?[0];need(count.is_finite() && count>=0.0 && count.fract()==0.0 && count<=slots as f32,"attention live count exceeds slots")?;
        let key=(vec![u32::MAX,heads,key_dim,value_dim,capacity as u32],0);let mut programs=self.programs.borrow_mut();
        if !programs.contains_key(&key) {
            use spark_gpu_probe::resident_program::{Program,Node};
            let scores=Shape {op:Op::ScoresSlots,heads,dim:key_dim,taps:0,rows:capacity as u32,cols:0,epsilon:0.0,bound:0.0};
            let softmax=Shape {op:Op::Softmax,..scores};let values=Shape {op:Op::ValuesChannels,dim:value_dim,..scores};
            let plan=Program {inputs:vec![heads as usize*key_dim as usize,capacity*heads as usize*key_dim as usize,capacity*heads as usize*value_dim as usize,1],
                nodes:vec![Node {shape:scores,inputs:[Some(0),Some(1),Some(3),None]},Node {shape:softmax,inputs:[Some(4),Some(3),None,None]},Node {shape:values,inputs:[Some(5),Some(2),Some(3),None]}]};
            programs.insert(key.clone(),self.backend.resident_program(&plan,&[None,None,None])?);
        }
        let stamp=self.cost.start(self.backend);let result=programs.get_mut(&key).unwrap().run_padded(&x.map(|v|&*v.vector))?;
        let words=result.len();let output=self.value(result,words)?;self.cost.record(||"attention_piece".into(),stamp,self.backend,0);Ok(Some(output))
    }
    fn embedding(&self,token:u32,streams:usize)->Result<Option<Value<'b,'d>>> {
        let stamp=self.cost.start(self.backend);let out=self.weights.borrow().embedding(self.backend,token,streams)?;let words=out.len();let value=self.value(out,words)?;self.cost.record(||"embedding".into(),stamp,self.backend,0);Ok(Some(value))
    }
    fn projection(&self,_:&CheckpointAdmission,name:&str,x:&Value<'b,'d>,round:bool)->Result<Option<Value<'b,'d>>> {
        self.ready(x)?;
        let out=self.weights.borrow_mut().apply(name,self.backend,&x.vector,round)?;
        #[cfg(feature="kernel-bench")] self.observe_projection(name,&out)?;
        let words=out.len();Ok(Some(self.value(out,words)?))
    }
    fn projection_rows(&self,_:&CheckpointAdmission,name:&str,input:&[&Value<'b,'d>],round:bool)->Result<Option<Vec<Value<'b,'d>>>> {
        for value in input {self.ready(value)?;}let rows=input.iter().map(|value|&*value.vector).collect::<Vec<_>>();
        let values=self.weights.borrow_mut().apply_rows(name,self.backend,&rows,round)?;
        values.into_iter().map(|value|{let words=value.len();self.value(value,words)}).collect::<Result<Vec<_>>>().map(Some)
    }
    fn projection_batch(&self,_:&CheckpointAdmission,input:&[(&str,&Value<'b,'d>)],round:bool)->Result<Option<Vec<Value<'b,'d>>>> {
        for (_,value) in input {self.ready(value)?;}let rows=input.iter().map(|(name,value)|(*name,&*value.vector)).collect::<Vec<_>>();
        let Some(values)=self.weights.borrow_mut().apply_batch(self.backend,&rows,round)? else {return Ok(None);};
        values.into_iter().map(|value|{let words=value.len();self.value(value,words)}).collect::<Result<Vec<_>>>().map(Some)
    }
    fn routed_experts(&self,steps:&[engine_model::plan::layer::Step],ids:&Value<'b,'d>,input:&Value<'b,'d>)->Result<Option<Vec<Value<'b,'d>>>> {
        self.ready(ids)?;self.ready(input)?;
        let Some(projected)=self.weights.borrow_mut().apply_routed(self.backend,steps,&ids.vector,&input.vector)? else {return Ok(None);};
        let picks=ids.words;need(projected.len()==steps.len()*picks,"device routed projection count differs")?;
        let values=projected.into_iter().map(|value|{let words=value.len();self.value(value,words)}).collect::<Result<Vec<_>>>()?;
        values.chunks(picks).map(|stage|self.join(&stage.iter().collect::<Vec<_>>())).collect::<Result<Vec<_>>>().map(Some)
    }
    fn projection_stage(&self,_:&CheckpointAdmission,jobs:&[crate::projection_stage::Job<'_,Value<'b,'d>>])->Result<Option<Vec<Value<'b,'d>>>> {
        use crate::projection_stage::Job;
        let mut mapped=Vec::new();
        for job in jobs {
            mapped.push(match job {
                Job::Project {input,round}=>{for (_,value) in input {self.ready(value)?;}Job::Project {input:input.iter().map(|(name,value)|(*name,&*value.vector)).collect(),round:*round}},
                Job::Routed {steps,ids,input}=>{self.ready(ids)?;self.ready(input)?;Job::Routed {steps,ids:&*ids.vector,input:&*input.vector}},
                Job::Gated {name,gate,up,shape}=>{self.ready(gate)?;self.ready(up)?;Job::Gated {name,gate:&*gate.vector,up:&*up.vector,shape:*shape}},
            });
        }
        let Some(output)=self.weights.borrow_mut().apply_stage(self.backend,&mapped)? else {return Ok(None);};
        output.into_iter().map(|parts| {
            let mut values=parts.into_iter().map(|part|{let words=part.len();self.value(part,words)}).collect::<Result<Vec<_>>>()?;
            if values.len()==1 {Ok(values.remove(0))} else {self.join(&values.iter().collect::<Vec<_>>())}
        }).collect::<Result<Vec<_>>>().map(Some)
    }
    fn upload(&self,v:&[f32])->Result<Value<'b,'d>> {if v.is_empty() {return self.zeros(0);}let stamp=self.cost.start(self.backend);let out=self.value(self.backend.upload_vector(v)?,v.len())?;self.cost.record(||"upload".into(),stamp,self.backend,0);Ok(out)}
    fn read(&self,v:&Value<'b,'d>)->Result<Vec<f32>> {
        self.ready(v)?;
        self.weights.borrow().check_exchange()?;if let Some(error)=self.backend.fatal_error() {return Err(error);}
        if let Some(out)=v.known.borrow().as_ref() {return Ok(out.clone());}
        if let Some(out)=self.selection_seen(v) {*v.known.borrow_mut()=Some(out.clone());return Ok(out);}
        let stamp=self.cost.start(self.backend);let checked=self.check_selection(Some(v))?;let out=if v.words==0 {Vec::new()} else if let Some(out)=checked {out} else {self.backend.read_vector_values(&v.vector)?};
        self.weights.borrow().check_exchange()?;*v.known.borrow_mut()=Some(out.clone());self.cost.record(||"readback".into(),stamp,self.backend,0);Ok(out)
    }
    fn read_pair(&self,first:&Value<'b,'d>,second:&Value<'b,'d>)->Result<(Vec<f32>,Vec<f32>)> {Device::read_pair(self,first,second)}
    fn select_known(&self,ids:&Value<'b,'d>,count:&Value<'b,'d>,row:&[f32])->Result<()> {
        if !self.selection_shortcut.get() {return Ok(());}
        need(count.words==1 && ids.words+1==row.len(),"known selection extent differs")?;
        *ids.known.borrow_mut()=Some(row[..ids.words].to_vec());*count.known.borrow_mut()=Some(vec![row[ids.words]]);
        ids.assumed.set(true);count.assumed.set(true);self.selection_checks.borrow_mut().push((count.vector.clone(),row[ids.words]));Ok(())
    }
    fn range(&self,v:&Value<'b,'d>,start:usize,len:usize)->Result<Value<'b,'d>> {
        let stamp=self.cost.start(self.backend);need(start.checked_add(len).is_some_and(|n|n<=v.words),"bounded device range differs")?;
        if start==0 && len==v.words {return Ok(Value {vector:v.vector.clone(),words:len,lineage:v.lineage.clone(),known:RefCell::new(v.known.borrow().clone()),assumed:v.assumed.clone(),pending:v.pending.clone(),held:v.held.clone()});}self.ready(v)?;
if start==0 && v.pending.as_ref().and_then(|p|p.borrow().state_prefix)==Some(len) {
    // A recurrent state must not keep all token outputs in its parent.
    // Copy raw words; the ordered stream and deferred owners preserve them.
    let source=v.vector.resident_range(0..len)?;let mut out=self.backend.vector(len)?;
    self.backend.copy_prefix(&source,&mut out,len)?;self.state_copies.set(self.state_copies.get().checked_add(1).ok_or("state copy counter overflow")?);let value=self.value(out,len)?;self.cost.record(||"state_prefix_copy".into(),stamp,self.backend,0);return Ok(value);
}
        let out=if len==0 {self.backend.vector(1)?} else {v.vector.resident_range(start..start+len)?};let mut value=self.value(out,len)?;value.lineage=v.lineage.range(start,len)?;
        // A view of a selection known on the host is known too. Other read values keep their read.
        if v.assumed.get() {value.assumed.set(true);*value.known.borrow_mut()=v.known.borrow().as_ref().map(|row|row[start..start+len].to_vec());}
        // A view of held rows, with any start, is listed in its storage: no append writes the words it reads.
        if let Some((storage,first))=v.held.as_ref().filter(|_|len>0) {storage.views.add(&value.vector,first+start+len);value.held=Some((storage.clone(),first+start));}
        self.cost.record(||"range_view".into(),stamp,self.backend,0);Ok(value)
    }
    fn join(&self,parts:&[&Value<'b,'d>])->Result<Value<'b,'d>> {
        for value in parts {self.ready(value)?;}
        let stamp=self.cost.start(self.backend);let len=parts.iter().try_fold(0usize,|n,v|n.checked_add(v.words).ok_or("bounded join overflow"))?;
        let sources=parts.iter().map(|v|(&*v.vector,v.words)).collect::<Vec<_>>();let out=self.backend.resident_join(&sources,&mut self.copies.borrow_mut(),0)?;let mut value=self.value(out,len)?;value.lineage=Lineage::join(parts.iter().map(|v|&v.lineage))?;self.cost.record(||"join_copy".into(),stamp,self.backend,0);Ok(value)
    }
    fn cache_append(&self,old:&Value<'b,'d>,new:&Value<'b,'d>,rows:usize,width:usize,room:usize)->Result<Value<'b,'d>> {
        if room==0 {return self.join(&[old,new]);}self.ready(old)?;self.ready(new)?;
        need(rows.checked_mul(width)==Some(old.words) && new.words==width,"latent cache append extent differs")?;
        // Only a tail beyond every live view can be written. A retained future
        // view or an exhausted arena copies to a new owner through rows_extend.
        let stamp=self.cost.start(self.backend);
        let mut value=self.rows_extend((rows>0).then_some(old),rows,new,width,room)?;
        // Reusing physical storage must not change immutable row identities:
        // projection memo keys keep the old prefix and the new row's origin.
        value.lineage=Lineage::join([&old.lineage,&new.lineage])?;
        self.cost.record(||"cache_append".into(),stamp,self.backend,0);Ok(value)
    }
    fn join_reserved(&self,parts:&[&Value<'b,'d>],reserve:usize)->Result<Value<'b,'d>> {
        let len=parts.iter().try_fold(0usize,|n,v|n.checked_add(v.words).ok_or("bounded join overflow"))?;
        if len==0 || reserve<=len {return self.join(parts);}
        for value in parts {self.ready(value)?;}
        // The joined words are the published value. The rest of the storage is never read.
        let stamp=self.cost.start(self.backend);let sources=parts.iter().map(|v|(&*v.vector,v.words)).collect::<Vec<_>>();
        let out=self.backend.resident_join(&sources,&mut self.copies.borrow_mut(),reserve)?.into_resident_shared()?;
        let mut value=self.value(out.resident_range(0..len)?,len)?;value.lineage=Lineage::join(parts.iter().map(|v|&v.lineage))?;self.cost.record(||"join_copy".into(),stamp,self.backend,0);Ok(value)
    }
    fn zeros(&self,len:usize)->Result<Value<'b,'d>> {let stamp=self.cost.start(self.backend);let mut v=self.backend.vector(len.max(1))?;self.backend.zero_vector(&mut v)?;let value=self.value(v,len)?;self.cost.record(||"zero".into(),stamp,self.backend,0);Ok(value)}
    fn operation(&self,s:Shape,x:[Option<&Value<'b,'d>>;4],w:&[f32])->Result<Value<'b,'d>> {self.operation_bound(s,x,w,None)}
    fn operation_prepared(&self,s:Shape,x:[Option<&Value<'b,'d>>;4],w:&Rc<StaticParameters>)->Result<Value<'b,'d>> {self.operation_bound(s,x,w.values(),Some(w))}
    fn norm(&self,name:&str,s:NormShape,x:&Value<'b,'d>,w:&[f32])->Result<Value<'b,'d>> {
        self.ready(x)?;
        let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);
        if !self.norms.borrow().contains_key(name) {self.norms.borrow_mut().insert(name.into(),self.backend.resident_norm(s,w)?);}
        let input=self.backend.round_bf16_lanes(&x.vector)?;let owners=self.norms.borrow();
        need(owners[name].shape()==s,"resident norm shape changed")?;let norm=owners[name].apply(&input)?;
        let mut out=self.backend.vector(x.words)?;self.backend.copy_prefix(norm.vector(),&mut out,x.words)?;
        let value=self.value(out,x.words)?;self.cost.record(||"norm".into(),stamp,self.backend,w.len()*4);self.record(2,started);Ok(value)
    }
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Value<'b,'d>)->Result<(Self::Pass,Value<'b,'d>)> {self.begin_next(key,p,x,&[])}
    fn begin_next(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Value<'b,'d>,next:&[&str])->Result<(Self::Pass,Value<'b,'d>)> {
        let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);
        if !self.blocks.borrow().contains_key(&key) {self.blocks.borrow_mut().insert(key,self.upload_boundary(p)?);}
        let post=self.take_post(x)?;
        // CHAIN-SHADOW: the next stage's resident weights, in the order it reads them, pulled toward
        // L2 by SHADOW_DECODE_BLOCKS tail blocks of the rms, dots and finish launches; each launch
        // takes at most its SHADOW_DECODE_STAGE_BYTES of one matrix, then the walk moves on. A
        // matrix that is not resident yet is skipped. The arithmetic launches and every value are unchanged.
        let weights=self.weights.borrow();
        let mut spans=[None,None,None];let (mut target,mut offset)=(0usize,0usize);
        for (span,budget) in spans.iter_mut().zip(spark_gpu_probe::mhc_params::SHADOW_DECODE_STAGE_BYTES) {
            if budget==0 {continue;}
            while let Some(name)=next.get(target) {
                match weights.shadow_range(self.backend,name,offset,budget)? {
                    Some((range,total))=>{offset+=range.bytes() as usize;if offset>=total {target+=1;offset=0;}*span=Some(range);break;},
                    None=>{target+=1;offset=0;},
                }
            }
        }
        let shadow=spans.iter().any(Option::is_some).then(||spark_gpu_probe::driver::Shadow {rms:spans[0],dots:spans[1],finish:spans[2],blocks:spark_gpu_probe::mhc_params::SHADOW_DECODE_BLOCKS,evict_last:false});
        let result=if let Some((pass,block))=post {self.blocks.borrow()[&key].begin_after_shadowed(pass,block,x.vector.clone(),shadow.as_ref())}
            else {self.blocks.borrow()[&key].begin_shadowed(x.vector.clone(),shadow.as_ref())};
        drop(weights);
        let pass=match result {Ok(pass)=>pass,Err(error)=>{if let Some(pending)=&x.pending {pending.borrow_mut().error=Some(error.clone());}return Err(error);}};
        if let Some(pending)=&x.pending {pending.borrow_mut().error=None;}
        let vector=pass.normalized();let words=vector.len();let value=self.shared_value(vector,words)?;
        self.cost.record(||"residual_begin".into(),stamp,self.backend,(p.mhc().weights().len()+p.mhc().base().len()+p.mhc().scale().len()+p.norm().weight().len())*4);self.record(0,started);Ok((pass,value))
    }
    fn finish(&self,p:Self::Pass,x:&Value<'b,'d>)->Result<Value<'b,'d>> {self.ready(x)?;let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);let shape=p.shape();
        need(x.words==shape.rows as usize*shape.hidden as usize,"deferred mHC block extent differs")?;let words=shape.rows as usize*shape.wide();
        let mut value=self.value(self.backend.vector(words)?,words)?;value.pending=Some(Rc::new(RefCell::new(Pending {post:Some((p,x.vector.clone())),error:None,state_prefix:None})));
        self.cost.record(||"residual_finish".into(),stamp,self.backend,0);self.record(1,started);Ok(value)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&Value<'b,'d>)->Result<Value<'b,'d>> {
        self.ready(x)?;
        let out=self.weights.borrow_mut().apply(p.component_names()[0],self.backend,&x.vector,true)?;let words=out.len();self.value(out,words)
    }
    fn greedy_projection(&self,_:&CheckpointAdmission,name:&str,x:&Value<'b,'d>)->Result<u32> {
        self.ready(x)?;self.weights.borrow().check_exchange()?;if let Some(error)=self.backend.fatal_error() {return Err(error);}
        self.check_selection(None)?;let best=self.weights.borrow_mut().apply_greedy(name,self.backend,&x.vector)?;
        self.weights.borrow().check_exchange()?;Ok(best)
    }
    fn round(&self,x:&Value<'b,'d>)->Result<Value<'b,'d>> {self.ready(x)?;let stamp=self.cost.start(self.backend);let rounded=self.backend.round_bf16_lanes(&x.vector)?;let mut out=self.backend.vector(x.words)?;
        self.backend.copy_prefix(rounded.vector(),&mut out,x.words)?;let value=self.value(out,x.words)?;self.cost.record(||"round".into(),stamp,self.backend,0);Ok(value)}
    fn mean(&self,x:&Value<'b,'d>,streams:u32,hidden:u32)->Result<Value<'b,'d>> {let mut s=shape(Op::StreamMean);s.heads=streams;s.dim=hidden;self.operation(s,[Some(x),None,None,None],&[])}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Value<'b,'d>>;5],w:&[f32])->Result<Value<'b,'d>> {
        let mut x=x;let poisoned;
        if s.op==spark_gpu_probe::prompt_ops::Op::SelectRows&& let Some(left)=self.screen_nan.get() {
            if left>0 {self.screen_nan.set(Some(left-1));} else if let Some(words)=x[0].map(|v|v.words).filter(|&n|n>0) {
                self.screen_nan.set(None);poisoned=self.upload(&vec![f32::NAN;words])?;x[0]=Some(&poisoned);
            }
        }
        for value in x.iter().flatten() {self.ready(value)?;}
        let stamp=self.cost.start(self.backend);let (lengths,_,units,words)=s.lengths().ok_or("invalid prompt row shape")?;
        for (v,count) in x.iter().zip(lengths) {need(v.map_or(0,|v|v.words)==count,"prompt row input extent differs")?;}
        let weight=self.parameters.borrow_mut().resolve(w,None,|values|self.backend.upload_vector(values))?;
        let input=x.map(|v|v.filter(|v|v.words>0).map(|v|&*v.vector));
        let mut value=self.value(self.backend.prompt_operation(s,input,weight.as_deref())?,units*words)?;
        if s.op==spark_gpu_probe::prompt_ops::Op::StepRows && s.tokens>256 {let prefix=(s.heads as usize).checked_mul(s.dim as usize).and_then(|n|n.checked_mul(s.dim as usize)).ok_or("state prefix overflow")?;value.pending=Some(Rc::new(RefCell::new(Pending {post:None,error:None,state_prefix:Some(prefix)})));}self.cost.settle(self.backend)?;
        self.cost.record(||if s.op==spark_gpu_probe::prompt_ops::Op::Wrapped {format!("rows_{:?}",s.inner.op)} else {format!("rows_{:?}",s.op)},stamp,self.backend,w.len()*4);Ok(value)
    }
    fn rows_embedding(&self,tokens:&[u32],streams:usize)->Result<Option<Value<'b,'d>>> {
        self.small_rows.set(tokens.len()<=super::VERIFY_ROWS);
        if !self.small_rows.get() {self.row_workspace.borrow_mut().take();}
        // A small group is a verify pass or a prompt tail. Like a decode token it leaves 1 GiB
        // under the vector pool's cap first: its selected-row buffers change size with each
        // pass, and a full pool refuses them (measured at 1,040 tokens: 1.87 GB of new buffers a
        // pass without this, 0.2 GB after a step had trimmed). Large groups clear the pool at their end.
        if tokens.len()<super::PROMPT_SHARED {self.backend.trim_vector_spares(1<<30);}
        let stamp=self.cost.start(self.backend);let out=self.weights.borrow().embedding_rows(self.backend,tokens,streams)?;let words=out.len();let value=self.value(out,words)?;self.cost.record(||"rows_embedding".into(),stamp,self.backend,0);Ok(Some(value))
    }
    fn rows_project(&self,_:&CheckpointAdmission,name:&str,x:&Value<'b,'d>,m:usize,round:bool)->Result<Option<Value<'b,'d>>> {
        self.ready(x)?;
        let out=self.weights.borrow_mut().apply_matrix(name,self.backend,&x.vector,m,round)?;let words=out.len();Ok(Some(self.value(out,words)?))
    }
    fn rows_experts(&self,first:&[(&str,usize)],second:Option<(&[(&str,usize)],f32)>,x:&Value<'b,'d>,places:&[f32])->Result<Option<Value<'b,'d>>> {
        self.ready(x)?;
        let Some(out)=self.weights.borrow_mut().apply_experts(self.backend,first,second,&x.vector,places)? else {return Ok(None);};let words=out.len();Ok(Some(self.value(out,words)?))
    }
    fn rows_release(&self)->Result<()> {self.backend.release_group_spares()}
    fn rows_keep(&self)->Result<()> {self.backend.keep_group_spares(super::POOL_ROOM)}
    fn rows_peak_begin(&self)->usize {self.backend.begin_group_peak()}
    fn rows_peak_end(&self,whole:usize)->usize {self.backend.end_group_peak(whole)}
    fn rows_reserve(&self,words:usize)->Result<Value<'b,'d>> {
        let vector=if self.small_rows.get() {self.backend.resident_workspace(words.max(1),&mut self.row_workspace.borrow_mut())?}
            else {self.backend.vector(words.max(1))?};self.value(vector,words)
    }
    fn rows_fill(&self,target:&mut Value<'b,'d>,offset:usize,part:&Value<'b,'d>,expected:usize)->Result<()> {
        // A dropped chunk stays on the deferred list until a drain. This one makes "chunk
        // by chunk" true for a matrix that takes no gather, on any topology.
        self.backend.synchronize()?;self.ready(part)?;
        need(part.words==expected && offset.checked_add(expected).is_some_and(|end|end<=target.words),"prompt fill extent differs")?;
        let vector=Rc::get_mut(&mut target.vector).ok_or("prompt fill target is shared")?;
        self.backend.copy_range(&part.vector,vector,0,offset,part.words)
    }
    fn rows_append(&self,held:Option<&Value<'b,'d>>,rows:usize,new:&Value<'b,'d>,width:usize)->Result<Value<'b,'d>> {
        self.ready(new)?;let have=rows.checked_mul(width).ok_or("prompt expanded rows overflow")?;let needed=have.checked_add(new.words).ok_or("prompt expanded rows overflow")?;
        // Exactly the rows held, so old plus new stays within twice the prompt's charge
        // while the state that holds the old rows is still live.
        let mut out=self.backend.vector(needed)?;
        if let Some(old)=held {need(old.words>=have,"prompt expanded rows differ")?;self.backend.copy_range(&old.vector,&mut out,0,0,have)?;}
        self.backend.copy_range(&new.vector,&mut out,0,have,new.words)?;
        self.value(out,needed)
    }
    fn rows_extend(&self,held:Option<&Value<'b,'d>>,rows:usize,new:&Value<'b,'d>,width:usize,room:usize)->Result<Value<'b,'d>> {
        if room==0 {return self.rows_append(held,rows,new,width);}
        self.ready(new)?;let have=rows.checked_mul(width).ok_or("prompt expanded rows overflow")?;let needed=have.checked_add(new.words).ok_or("prompt expanded rows overflow")?;
        let publish=|storage:&Rc<HeldStorage<'d>>|->Result<Value<'b,'d>> {
            let mut value=self.shared_value(Rc::new(storage.whole.resident_range(0..needed)?),needed)?;storage.views.add(&value.vector,needed);value.held=Some((storage.clone(),0));Ok(value)};
        if let Some((v,(storage,start)))=held.and_then(|v|v.held.as_ref().map(|h|(v,h))) {
            // Appendable cache storage whose held rows start it and past which no live view ends:
            // nobody reads the words after them. A memo slot is never appendable. In every other
            // case the rows are copied below.
            if super::held::append_in_place(storage.appendable,*start,v.words,have,needed,storage.capacity,storage.views.last()) {
                let mut tail=storage.whole.resident_range(have..needed)?;self.backend.copy_range(&new.vector,&mut tail,0,0,new.words)?;return publish(storage);}
        }
        let capacity=needed.checked_add(room).ok_or("prompt expanded rows overflow")?;let mut out=self.backend.vector(capacity)?;
        if let Some(old)=held {need(old.words>=have,"prompt expanded rows differ")?;self.backend.copy_range(&old.vector,&mut out,0,0,have)?;}
        self.backend.copy_range(&new.vector,&mut out,0,have,new.words)?;
        publish(&Rc::new(HeldStorage {whole:out.into_resident_shared()?,capacity,views:Default::default(),appendable:true,row:None}))
    }
    fn rows_norm(&self,name:&str,s:NormShape,x:&Value<'b,'d>,m:usize,w:&[f32])->Result<Value<'b,'d>> {
        self.ready(x)?;
        let stamp=self.cost.start(self.backend);
        if !self.norms.borrow().contains_key(name) {self.norms.borrow_mut().insert(name.into(),self.backend.resident_norm(s,w)?);}
        let input=self.backend.round_bf16_lanes(&x.vector)?;let owners=self.norms.borrow();
        need(owners[name].shape()==s,"resident norm shape changed")?;let out=owners[name].apply_rows(&input,m)?;self.cost.settle(self.backend)?;
        let value=self.value(out,x.words)?;self.cost.record(||"rows_norm".into(),stamp,self.backend,w.len()*4);Ok(value)
    }
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Value<'b,'d>,m:usize)->Result<(Self::Pass,Value<'b,'d>)> {
        let stamp=self.cost.start(self.backend);
        if !self.blocks.borrow().contains_key(&key) {self.blocks.borrow_mut().insert(key,self.upload_boundary(p)?);}
        let result=if let Some((pass,block))=self.take_post(x)? {self.blocks.borrow()[&key].begin_after_rows(pass,block,x.vector.clone(),m)}
            else {self.blocks.borrow()[&key].begin_rows(x.vector.clone(),m)};
        let pass=match result {Ok(pass)=>pass,Err(error)=>{if let Some(pending)=&x.pending {pending.borrow_mut().error=Some(error.clone());}return Err(error);}};
        if let Some(pending)=&x.pending {pending.borrow_mut().error=None;}
        let vector=pass.normalized();let words=vector.len();let value=self.shared_value(vector,words)?;
        self.cost.record(||"rows_begin".into(),stamp,self.backend,(p.mhc().weights().len()+p.mhc().base().len()+p.mhc().scale().len()+p.norm().weight().len())*4);Ok((pass,value))
    }
    fn rows_finish(&self,p:Self::Pass,x:&Value<'b,'d>,m:usize)->Result<Value<'b,'d>> {
        need(p.shape().rows as usize==m,"deferred mHC row count differs")?;self.finish(p,x)
    }
}

impl<'b,'d> Device<'b,'d> {
    pub fn parameter_counts(&self)->ParameterCounts {self.parameters.borrow().counts()}
    fn operation_bound(&self,s:Shape,x:[Option<&Value<'b,'d>>;4],w:&[f32],prepared:Option<&Rc<StaticParameters>>)->Result<Value<'b,'d>> {
        for value in x.iter().flatten() {self.ready(value)?;}
        let s=match s.op {Op::Scores=>Shape {op:Op::ScoresSlots,..s},Op::Values=>Shape {op:Op::ValuesChannels,..s},_=>s};
        let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);
        let (lengths,_,units,words)=s.lengths().ok_or("invalid bounded device operation")?;
        for (v,count) in x.iter().zip(lengths) {need(v.map_or(0,|v|v.words)==count,"bounded device operation input extent differs")?;}
        let weight=self.parameters.borrow_mut().resolve(w,prepared,|values|self.backend.upload_vector(values)?.into_resident_shared())?;
        let input=x.map(|v|v.filter(|v|v.words>0).map(|v|&*v.vector));
        let graphable=self.backend.resident_queue_active() && !self.backend.resident_launches_active() && matches!(s.op,Op::Conv|Op::ConvParts|Op::StepPlanar|Op::Prepare|Op::Step|Op::GatedNorm|Op::SwiGlu|Op::KeyNorm|Op::Route|Op::ExpertFold|Op::StreamMean|Op::Mxfp8Quantize);
        let v=if graphable {
            let key=(vec![s.op as u32,s.heads,s.dim,s.taps,s.rows,s.cols,s.epsilon.to_bits(),s.bound.to_bits()],weight.as_deref().map_or(0,DeviceVector::device_address));
            let mut programs=self.programs.borrow_mut();
            if !programs.contains_key(&key) {
                let plan=spark_gpu_probe::resident_program::Program::operation(s)?;
                let weights=plan.nodes.iter().map(|n|if n.shape.lengths().unwrap().1>0 {weight.as_deref()} else {None}).collect::<Vec<_>>();
                programs.insert(key.clone(),self.backend.resident_program(&plan,&weights)?);
            }
            programs.get_mut(&key).unwrap().run(&input.into_iter().flatten().collect::<Vec<_>>())?
        } else if s.op==Op::Step && !spark_gpu_probe::causal_ops::fast_admitted(s,spark_gpu_probe::causal_ops::FAST_THREADS) {
            let columns=self.backend.causal_operation(Shape {op:Op::StepColumns,..s},input,None)?;
            self.backend.causal_operation(Shape {op:Op::StepLayout,..s},[Some(&columns),None,None,None],None)?
        } else if s.op==Op::Pool {
            let channels=self.backend.causal_operation(Shape {op:Op::PoolChannels,..s},input,weight.as_deref())?;
            self.backend.causal_operation(Shape {op:Op::PoolLayout,..s},[Some(&channels),None,None,None],None)?
        } else if s.op==Op::Conv && !spark_gpu_probe::causal_ops::fast_admitted(s,spark_gpu_probe::causal_ops::FAST_THREADS) {
            let channels=self.backend.causal_operation(Shape {op:Op::ConvChannels,..s},input,weight.as_deref())?;
            self.backend.causal_operation(Shape {op:Op::ConvLayout,..s},[Some(&channels),None,None,None],None)?
        } else {self.backend.causal_operation(s,input,weight.as_deref())?};
        let mut value=self.value(v,units*words)?;
        if s.op==Op::Gather {
            let source=x[0].ok_or("gather source is absent")?;
            let (selected,count)=self.read_pair(x[1].ok_or("gather selection is absent")?,x[2].ok_or("gather count is absent")?)?;let count=count[0] as usize;
            if count>s.cols as usize {return Err("gather count exceeds live slots".into());}
            let ids=selected[..count].iter().map(|v|*v as usize).collect::<Vec<_>>();
            value.lineage=source.lineage.gather(s.dim as usize,&ids,s.cols as usize,self.source.get())?;
        }
        self.cost.record(||format!("operation_{:?}",s.op),stamp,self.backend,w.len()*4);self.record(3,started);Ok(value)
    }
}
