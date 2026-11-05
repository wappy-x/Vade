//! Device values retain logical empty shapes with one private unused lane.
use super::*;
use spark_gpu_probe::driver::{Backend,BackendStats,DeviceVector,MhcBoundaryPass,ResidentMhcBoundary};
use std::{cell::{Cell,RefCell},rc::Rc};
use super::lineage::Lineage;
pub struct Device<'b,'d> {pub backend:&'b Backend<'d>,weights:Rc<RefCell<crate::resident_weights::Weights<'b,'d>>>,
    parameters:Rc<RefCell<BTreeMap<Vec<u32>,DeviceVector<'d>>>>,
    norms:Rc<RefCell<BTreeMap<String,spark_gpu_probe::driver::ResidentNorm<'b,'d>>>>,
    blocks:Rc<RefCell<BTreeMap<(usize,usize),ResidentMhcBoundary<'b,'d>>>>,
    stages:Rc<RefCell<[[u128;2];4]>>,cost:Rc<crate::resident_profile::Cost>,memo:RefCell<Memo<'d>>,copies:RefCell<Option<DeviceVector<'d>>>,source:Rc<Cell<u64>>,programs:RefCell<BTreeMap<(Vec<u32>,u64),spark_gpu_probe::driver::ResidentProgram<'b,'d>>>,
    serving:Option<RefCell<Option<(u64,std::time::Instant,BackendStats)>> >}
type BitsKey=(String,Vec<u32>,bool);
type OriginKey=(String,Lineage,bool);
#[derive(Default)]
struct Memo<'d> {values:BTreeMap<BitsKey,Rc<DeviceVector<'d>>>,origins:BTreeMap<OriginKey,Rc<DeviceVector<'d>>>,bytes:usize,limit:usize,hits:u64,misses:u64}
impl Clone for Device<'_,'_> {
    fn clone(&self)->Self {Self {backend:self.backend,weights:self.weights.clone(),parameters:self.parameters.clone(),norms:self.norms.clone(),
        blocks:self.blocks.clone(),stages:self.stages.clone(),cost:self.cost.clone(),memo:RefCell::new(Memo::default()),copies:RefCell::new(None),source:self.source.clone(),programs:RefCell::new(BTreeMap::new()),serving:self.serving.as_ref().map(|_|RefCell::new(None))}}
}
struct Pending<'b,'d> {post:Option<(MhcBoundaryPass<'b,'d>,Rc<DeviceVector<'d>>)>,error:Option<String>}
pub struct Value<'b,'d> {vector:Rc<DeviceVector<'d>>,words:usize,lineage:Lineage,known:RefCell<Option<Vec<f32>>>,pending:Option<Rc<RefCell<Pending<'b,'d>>>>}
impl<'b,'d> Device<'b,'d> {
    pub fn new(backend:&'b Backend<'d>)->Self {Self::with_ring(backend,None)}
    pub fn with_ring(backend:&'b Backend<'d>,ring:Option<crate::transport::Ring>)->Self {Self::with_ring_profiled(backend,ring,false)}
    /// Diagnostic runner counters. Normal construction keeps them disabled.
    pub fn with_ring_profiled(backend:&'b Backend<'d>,ring:Option<crate::transport::Ring>,cost:bool)->Self {
        Self {backend,weights:Rc::new(RefCell::new(crate::resident_weights::Weights::new(ring,cost))),parameters:Rc::new(RefCell::new(BTreeMap::new())),
            norms:Rc::new(RefCell::new(BTreeMap::new())),blocks:Rc::new(RefCell::new(BTreeMap::new())),stages:Rc::new(RefCell::new([[0;2];4])),cost:Rc::new(crate::resident_profile::Cost::new(cost)),memo:RefCell::new(Memo::default()),copies:RefCell::new(None),source:Rc::new(Cell::new(0)),programs:RefCell::new(BTreeMap::new()),serving:None}
    }
    /// Wall counters only. This does not enable CUDA event profiling.
    pub fn with_serving_trace(mut self,enabled:bool)->Self {self.serving=enabled.then(||RefCell::new(None));self}
    /// The caller keeps the attached wire and mailbox alive through all steps.
    pub fn with_mailbox(backend:&'b Backend<'d>,ring:crate::transport::Ring,cost:bool,mailbox:&'b spark_gpu_probe::driver::TpMailbox<'d>)->Result<Self> {
        let device=Self::with_ring_profiled(backend,Some(ring),cost);device.weights.borrow_mut().attach_mailbox(mailbox)?;Ok(device)
    }
    // Device consumers share the backend's ordered stream. Readback waits, and
    // Buffer::drop drains pending work before a source or destination is freed.
    // Keep owners here; a value wrapper needs no separate completion barrier.
    fn value(&self,vector:DeviceVector<'d>,words:usize)->Result<Value<'b,'d>> {self.shared_value(Rc::new(vector.into_resident_shared()?),words)}
    fn shared_value(&self,vector:Rc<DeviceVector<'d>>,words:usize)->Result<Value<'b,'d>> {let source=self.source.get().checked_add(1).ok_or("immutable source identity exhausted")?;self.source.set(source);
        Ok(Value {vector,words,lineage:Lineage::fresh(source,words),known:RefCell::new(None),pending:None})}
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
    fn record(&self,stage:usize,started:std::time::Instant) {let mut stages=self.stages.borrow_mut();stages[stage][0]+=1;stages[stage][1]+=started.elapsed().as_nanos();}
    pub(crate) fn report(&self) {self.weights.borrow_mut().report();self.cost.report();
        for (name,timing) in ["residual_begin","residual_finish","standalone_norm","causal_operations"].into_iter().zip(self.stages.borrow_mut().iter_mut()) {
            eprintln!("RESIDENT STAGE name={name} calls={} milliseconds={:.6}",timing[0],timing[1] as f64/1e6);*timing=[0;2];
        }
        let mut memo=self.memo.borrow_mut();eprintln!("RESIDENT MEMO hits={} misses={} bytes={} limit={}",memo.hits,memo.misses,memo.bytes,memo.limit);memo.hits=0;memo.misses=0;
    }
}
impl<'b,'d> Compute for Device<'b,'d> {
    fn work<T>(&self,run:impl FnOnce()->Result<T>)->Result<T> {
        let queue=self.backend.resident_queue()?;self.backend.enable_resident_launches()?;
        let result=run();drop(queue);
        if let Some(error)=self.backend.fatal_error() {return Err(error);}
        self.weights.borrow().check_exchange()?;result
    }
    fn resident_weights_ready(&self)->bool {self.weights.borrow().prepared()}
    fn fatal_error(&self)->Option<String> {self.backend.fatal_error().or_else(||self.weights.borrow().check_exchange().err())}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {Some((self.backend.accounted_bytes(),self.backend.peak_accounted_bytes(),self.backend.pinned_host_bytes()))}
    fn serving_trace(&self)->bool {self.serving.is_some()}
    fn profile_tick_begin(&self,tick:u64) {if let Some(trace)=&self.serving {*trace.borrow_mut()=Some((tick,std::time::Instant::now(),self.backend.stats()));}}
    fn profile_tick_end(&self) {if let Some(trace)=&self.serving {if let Some((tick,started,before))=trace.borrow_mut().take() {
        let elapsed=started.elapsed().as_nanos();let s=self.backend.stats().since(before);
        eprintln!("SERVED TICK tick={tick} wall_ns={elapsed} allocation_ns={} wait_ns={} wait_calls={} launch_ns={} launch_calls={} download_ns={} upload_ns={} pack_ns={} graph_calls={}",
            s.allocation_ns,s.stream_wait_ns,s.stream_wait_calls,s.host_launch_ns,s.queued_calls,s.output_download_ns,s.weight_read_upload_ns,s.host_pack_ns,s.graph_launches);
    }}}
    fn plan_cost(&self,ns:u128) {self.cost.host("planning",ns);}
    fn trace_start(&self) {if self.cost.enabled() {self.backend.set_profiling(true);}}
    fn cost_report(&self,phase:&str) {if self.cost.enabled() {eprintln!("SERVED COST phase={phase}");self.report();}}
    type Value=Value<'b,'d>;type Pass=MhcBoundaryPass<'b,'d>;
    fn projects_complete_rows(&self)->bool {true}
    fn prepare(&self,c:&CheckpointAdmission)->Result<()> {
        // A service can retain this owner while replacing a decoder session.
        // Pure memo entries are shared across live sessions; preparation clears them.
        *self.memo.borrow_mut()=Memo::default();self.weights.borrow_mut().prepare(c,self.backend)?;
        // Reuse completed private vectors under the backend's existing pressure
        // and size limits. A service's earlier memory admission owns its margin.
        if spark_gpu_probe::driver::vector_pool_from_env()? && !self.backend.vector_pool_configured() {self.backend.configure_vector_pool(1<<30);}
        let local=self.backend.available_memory_bytes()?.saturating_sub(1<<30).min(1<<30);
        self.memo.borrow_mut().limit=self.weights.borrow_mut().shared_budget(local)?;if self.backend.resident_queue_active() {self.backend.enable_resident_launches()?;}Ok(())}
    fn repeated_projection(&self,c:&CheckpointAdmission,name:&str,x:&Value<'b,'d>,round:bool)->Result<Option<Value<'b,'d>>> {
        self.ready(x)?;
        if self.memo.borrow().limit==0 {return self.projection(c,name,x,round);}
        // Unchanged immutable rows retain their original identity through view,
        // append and gather. A new origin still uses the full original bit key.
        let origin=(name.to_owned(),x.lineage.clone(),round);let stamp=self.cost.start(self.backend);
        let found=self.memo.borrow().origins.get(&origin).cloned();
        if let Some(output)=found {self.memo.borrow_mut().hits+=1;let words=output.len();let value=self.value(output.resident_range(0..words)?,words)?;
            self.cost.record(||"memo_origin".into(),stamp,self.backend,0);return Ok(Some(value));}
        let bits=self.read(x)?.iter().map(|v|v.to_bits()).collect::<Vec<_>>();let key=(name.to_owned(),bits,round);
        self.cost.record(||"memo_lookup".into(),stamp,self.backend,0);
        let prior=self.memo.borrow().values.get(&key).cloned();
        let output=if let Some(output)=prior {self.memo.borrow_mut().hits+=1;output} else {
            self.memo.borrow_mut().misses+=1;Rc::new(self.weights.borrow_mut().apply(name,self.backend,&x.vector,round)?.into_resident_shared()?)
        };
        let words=output.len();let value=self.value(output.resident_range(0..words)?,words)?;
        let mut memo=self.memo.borrow_mut();
        let bytes=origin.1.bytes().checked_add(name.len()).and_then(|n|n.checked_add(std::mem::size_of::<OriginKey>()+std::mem::size_of::<Rc<DeviceVector<'d>>>())).ok_or("projection origin byte count overflow")?;
        let bytes=words.checked_add(x.words).and_then(|n|n.checked_mul(4)).and_then(|n|n.checked_add(name.len()+bytes+std::mem::size_of::<BitsKey>())).ok_or("projection memo byte count overflow")?;
        if bytes<=memo.limit {
            if memo.bytes.checked_add(bytes).is_none_or(|n|n>memo.limit) {memo.values.clear();memo.origins.clear();memo.bytes=0;}
            memo.bytes+=bytes;memo.values.entry(key).or_insert_with(||output.clone());memo.origins.insert(origin,output);
        }
        Ok(Some(value))
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
        let out=self.weights.borrow_mut().apply(name,self.backend,&x.vector,round)?;let words=out.len();Ok(Some(self.value(out,words)?))
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
    fn upload(&self,v:&[f32])->Result<Value<'b,'d>> {if v.is_empty() {return self.zeros(0);}let stamp=self.cost.start(self.backend);let out=self.value(self.backend.upload_vector(v)?,v.len())?;self.cost.record(||"upload".into(),stamp,self.backend,0);Ok(out)}
    fn read(&self,v:&Value<'b,'d>)->Result<Vec<f32>> {
        self.ready(v)?;
        self.weights.borrow().check_exchange()?;if let Some(error)=self.backend.fatal_error() {return Err(error);}
        if let Some(out)=v.known.borrow().as_ref() {return Ok(out.clone());}
        let stamp=self.cost.start(self.backend);let out=if v.words==0 {Vec::new()} else {self.backend.read_vector_values(&v.vector)?};
        self.weights.borrow().check_exchange()?;*v.known.borrow_mut()=Some(out.clone());self.cost.record(||"readback".into(),stamp,self.backend,0);Ok(out)
    }
    fn range(&self,v:&Value<'b,'d>,start:usize,len:usize)->Result<Value<'b,'d>> {
        let stamp=self.cost.start(self.backend);need(start.checked_add(len).is_some_and(|n|n<=v.words),"bounded device range differs")?;
        if start==0 && len==v.words {return Ok(Value {vector:v.vector.clone(),words:len,lineage:v.lineage.clone(),known:RefCell::new(v.known.borrow().clone()),pending:v.pending.clone()});}self.ready(v)?;
        let out=if len==0 {self.backend.vector(1)?} else {v.vector.resident_range(start..start+len)?};let mut value=self.value(out,len)?;value.lineage=v.lineage.range(start,len)?;self.cost.record(||"range_view".into(),stamp,self.backend,0);Ok(value)
    }
    fn join(&self,parts:&[&Value<'b,'d>])->Result<Value<'b,'d>> {
        for value in parts {self.ready(value)?;}
        let stamp=self.cost.start(self.backend);let len=parts.iter().try_fold(0usize,|n,v|n.checked_add(v.words).ok_or("bounded join overflow"))?;
        let sources=parts.iter().map(|v|(&*v.vector,v.words)).collect::<Vec<_>>();let out=self.backend.resident_join(&sources,&mut self.copies.borrow_mut())?;let mut value=self.value(out,len)?;value.lineage=Lineage::join(parts.iter().map(|v|&v.lineage))?;self.cost.record(||"join_copy".into(),stamp,self.backend,0);Ok(value)
    }
    fn zeros(&self,len:usize)->Result<Value<'b,'d>> {let stamp=self.cost.start(self.backend);let mut v=self.backend.vector(len.max(1))?;self.backend.zero_vector(&mut v)?;let value=self.value(v,len)?;self.cost.record(||"zero".into(),stamp,self.backend,0);Ok(value)}
    fn operation(&self,s:Shape,x:[Option<&Value<'b,'d>>;4],w:&[f32])->Result<Value<'b,'d>> {
        for value in x.iter().flatten() {self.ready(value)?;}
        let s=match s.op {Op::Scores=>Shape {op:Op::ScoresSlots,..s},Op::Values=>Shape {op:Op::ValuesChannels,..s},_=>s};
        let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);
        let (lengths,_,units,words)=s.lengths().ok_or("invalid bounded device operation")?;
        for (v,count) in x.iter().zip(lengths) {need(v.map_or(0,|v|v.words)==count,"bounded device operation input extent differs")?;}
        let key=w.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        if !w.is_empty() && !self.parameters.borrow().contains_key(&key) {self.parameters.borrow_mut().insert(key.clone(),self.backend.upload_vector(w)?.into_resident_shared()?);}
        let owners=self.parameters.borrow();
        let input=x.map(|v|v.filter(|v|v.words>0).map(|v|&*v.vector));
        let graphable=self.backend.resident_queue_active() && !self.backend.resident_launches_active() && matches!(s.op,Op::Conv|Op::Prepare|Op::Step|Op::GatedNorm|Op::SwiGlu|Op::KeyNorm|Op::Route|Op::ExpertFold|Op::StreamMean|Op::Mxfp8Quantize);
        let v=if graphable {
            let key=(vec![s.op as u32,s.heads,s.dim,s.taps,s.rows,s.cols,s.epsilon.to_bits(),s.bound.to_bits()],owners.get(&key).map_or(0,DeviceVector::device_address));
            let mut programs=self.programs.borrow_mut();
            if !programs.contains_key(&key) {
                let plan=spark_gpu_probe::resident_program::Program::operation(s)?;
                let weights=plan.nodes.iter().map(|n|if n.shape.lengths().unwrap().1>0 {owners.get(&w.iter().map(|v|v.to_bits()).collect::<Vec<_>>())} else {None}).collect::<Vec<_>>();
                programs.insert(key.clone(),self.backend.resident_program(&plan,&weights)?);
            }
            programs.get_mut(&key).unwrap().run(&input.into_iter().flatten().collect::<Vec<_>>())?
        } else if s.op==Op::Step {
            let columns=self.backend.causal_operation(Shape {op:Op::StepColumns,..s},input,None)?;
            self.backend.causal_operation(Shape {op:Op::StepLayout,..s},[Some(&columns),None,None,None],None)?
        } else if s.op==Op::Pool {
            let channels=self.backend.causal_operation(Shape {op:Op::PoolChannels,..s},input,owners.get(&key))?;
            self.backend.causal_operation(Shape {op:Op::PoolLayout,..s},[Some(&channels),None,None,None],None)?
        } else if s.op==Op::Conv {
            let channels=self.backend.causal_operation(Shape {op:Op::ConvChannels,..s},input,owners.get(&key))?;
            self.backend.causal_operation(Shape {op:Op::ConvLayout,..s},[Some(&channels),None,None,None],None)?
        } else {self.backend.causal_operation(s,input,owners.get(&key))?};
        let mut value=self.value(v,units*words)?;
        if s.op==Op::Gather {
            let source=x[0].ok_or("gather source is absent")?;let selected=self.read(x[1].ok_or("gather selection is absent")?)?;
            let count=self.read(x[2].ok_or("gather count is absent")?)?[0] as usize;
            if count>s.cols as usize {return Err("gather count exceeds live slots".into());}
            let ids=selected[..count].iter().map(|v|*v as usize).collect::<Vec<_>>();
            value.lineage=source.lineage.gather(s.dim as usize,&ids,s.cols as usize,self.source.get())?;
        }
        self.cost.record(||format!("operation_{:?}",s.op),stamp,self.backend,w.len()*4);self.record(3,started);Ok(value)
    }
    fn norm(&self,name:&str,s:NormShape,x:&Value<'b,'d>,w:&[f32])->Result<Value<'b,'d>> {
        self.ready(x)?;
        let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);
        if !self.norms.borrow().contains_key(name) {self.norms.borrow_mut().insert(name.into(),self.backend.resident_norm(s,w)?);}
        let input=self.backend.round_bf16_lanes(&x.vector)?;let owners=self.norms.borrow();
        need(owners[name].shape()==s,"resident norm shape changed")?;let norm=owners[name].apply(&input)?;
        let mut out=self.backend.vector(x.words)?;self.backend.copy_prefix(norm.vector(),&mut out,x.words)?;
        let value=self.value(out,x.words)?;self.cost.record(||"norm".into(),stamp,self.backend,w.len()*4);self.record(2,started);Ok(value)
    }
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Value<'b,'d>)->Result<(Self::Pass,Value<'b,'d>)> {
        let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);
        if !self.blocks.borrow().contains_key(&key) {self.blocks.borrow_mut().insert(key,p.upload_boundary(self.backend)?);}
        let result=if let Some((pass,block))=self.take_post(x)? {self.blocks.borrow()[&key].begin_after(pass,block,x.vector.clone())}
            else {self.blocks.borrow()[&key].begin(x.vector.clone())};
        let pass=match result {Ok(pass)=>pass,Err(error)=>{if let Some(pending)=&x.pending {pending.borrow_mut().error=Some(error.clone());}return Err(error);}};
        if let Some(pending)=&x.pending {pending.borrow_mut().error=None;}
        let vector=pass.normalized();let words=vector.len();let value=self.shared_value(vector,words)?;
        self.cost.record(||"residual_begin".into(),stamp,self.backend,(p.mhc().weights().len()+p.mhc().base().len()+p.mhc().scale().len()+p.norm().weight().len())*4);self.record(0,started);Ok((pass,value))
    }
    fn finish(&self,p:Self::Pass,x:&Value<'b,'d>)->Result<Value<'b,'d>> {self.ready(x)?;let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);let shape=p.shape();
        need(x.words==shape.rows as usize*shape.hidden as usize,"deferred mHC block extent differs")?;let words=shape.rows as usize*shape.wide();
        let mut value=self.value(self.backend.vector(words)?,words)?;value.pending=Some(Rc::new(RefCell::new(Pending {post:Some((p,x.vector.clone())),error:None})));
        self.cost.record(||"residual_finish".into(),stamp,self.backend,0);self.record(1,started);Ok(value)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&Value<'b,'d>)->Result<Value<'b,'d>> {
        self.ready(x)?;
        let out=self.weights.borrow_mut().apply(p.component_names()[0],self.backend,&x.vector,true)?;let words=out.len();self.value(out,words)
    }
    fn round(&self,x:&Value<'b,'d>)->Result<Value<'b,'d>> {self.ready(x)?;let stamp=self.cost.start(self.backend);let rounded=self.backend.round_bf16_lanes(&x.vector)?;let mut out=self.backend.vector(x.words)?;
        self.backend.copy_prefix(rounded.vector(),&mut out,x.words)?;let value=self.value(out,x.words)?;self.cost.record(||"round".into(),stamp,self.backend,0);Ok(value)}
    fn mean(&self,x:&Value<'b,'d>,streams:u32,hidden:u32)->Result<Value<'b,'d>> {let mut s=shape(Op::StreamMean);s.heads=streams;s.dim=hidden;self.operation(s,[Some(x),None,None,None],&[])}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Value<'b,'d>>;5],w:&[f32])->Result<Value<'b,'d>> {
        for value in x.iter().flatten() {self.ready(value)?;}
        let stamp=self.cost.start(self.backend);let (lengths,_,units,words)=s.lengths().ok_or("invalid prompt row shape")?;
        for (v,count) in x.iter().zip(lengths) {need(v.map_or(0,|v|v.words)==count,"prompt row input extent differs")?;}
        let key=w.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        if !w.is_empty() && !self.parameters.borrow().contains_key(&key) {self.parameters.borrow_mut().insert(key.clone(),self.backend.upload_vector(w)?);}
        let owners=self.parameters.borrow();let input=x.map(|v|v.filter(|v|v.words>0).map(|v|&*v.vector));
        let value=self.value(self.backend.prompt_operation(s,input,owners.get(&key))?,units*words)?;self.cost.settle(self.backend)?;
        self.cost.record(||if s.op==spark_gpu_probe::prompt_ops::Op::Wrapped {format!("rows_{:?}",s.inner.op)} else {format!("rows_{:?}",s.op)},stamp,self.backend,w.len()*4);Ok(value)
    }
    fn rows_embedding(&self,tokens:&[u32],streams:usize)->Result<Option<Value<'b,'d>>> {
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
        if !self.blocks.borrow().contains_key(&key) {self.blocks.borrow_mut().insert(key,p.upload_boundary(self.backend)?);}
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
