//! Device values retain logical empty shapes with one private unused lane.
use super::*;
use spark_gpu_probe::driver::{Backend,DeviceVector};
use crate::layer_operands::device::ResidualBlockPass;
use std::{cell::RefCell,rc::Rc};
pub struct Device<'b,'d> {pub backend:&'b Backend<'d>,weights:Rc<RefCell<crate::resident_weights::Weights<'b,'d>>>,
    parameters:Rc<RefCell<BTreeMap<Vec<u32>,DeviceVector<'d>>>>,
    norms:Rc<RefCell<BTreeMap<String,spark_gpu_probe::driver::ResidentNorm<'b,'d>>>>,
    blocks:Rc<RefCell<BTreeMap<(usize,usize),crate::layer_operands::device::ResidentResidualBlock<'b,'d>>>>,
    stages:Rc<RefCell<[[u128;2];4]>>,cost:Rc<crate::resident_profile::Cost>,memo:RefCell<Memo<'d>>,copies:RefCell<Option<DeviceVector<'d>>>}
#[derive(Default)]
struct Memo<'d> {values:BTreeMap<(String,Vec<u32>,bool),DeviceVector<'d>>,bytes:usize,limit:usize,hits:u64,misses:u64}
impl Clone for Device<'_,'_> {
    fn clone(&self)->Self {Self {backend:self.backend,weights:self.weights.clone(),parameters:self.parameters.clone(),norms:self.norms.clone(),
        blocks:self.blocks.clone(),stages:self.stages.clone(),cost:self.cost.clone(),memo:RefCell::new(Memo::default()),copies:RefCell::new(None)}}
}
pub struct Value<'d> {vector:DeviceVector<'d>,words:usize}
impl<'b,'d> Device<'b,'d> {
    pub fn new(backend:&'b Backend<'d>)->Self {Self::with_ring(backend,None)}
    pub fn with_ring(backend:&'b Backend<'d>,ring:Option<crate::transport::Ring>)->Self {Self::with_ring_profiled(backend,ring,false)}
    /// Diagnostic runner counters. Normal construction keeps them disabled.
    pub fn with_ring_profiled(backend:&'b Backend<'d>,ring:Option<crate::transport::Ring>,cost:bool)->Self {
        Self {backend,weights:Rc::new(RefCell::new(crate::resident_weights::Weights::new(ring,cost))),parameters:Rc::new(RefCell::new(BTreeMap::new())),
            norms:Rc::new(RefCell::new(BTreeMap::new())),blocks:Rc::new(RefCell::new(BTreeMap::new())),stages:Rc::new(RefCell::new([[0;2];4])),cost:Rc::new(crate::resident_profile::Cost::new(cost)),memo:RefCell::new(Memo::default()),copies:RefCell::new(None)}
    }
    /// The caller keeps the attached wire and mailbox alive through all steps.
    pub fn with_mailbox(backend:&'b Backend<'d>,ring:crate::transport::Ring,cost:bool,mailbox:&'b spark_gpu_probe::driver::TpMailbox<'d>)->Result<Self> {
        let device=Self::with_ring_profiled(backend,Some(ring),cost);device.weights.borrow_mut().attach_mailbox(mailbox)?;Ok(device)
    }
    // Device consumers share the backend's ordered stream. Readback waits, and
    // Buffer::drop drains pending work before a source or destination is freed.
    // Keep owners here; a value wrapper needs no separate completion barrier.
    fn value(&self,vector:DeviceVector<'d>,words:usize)->Result<Value<'d>> {Ok(Value {vector:vector.into_resident_shared()?,words})}
    fn record(&self,stage:usize,started:std::time::Instant) {let mut stages=self.stages.borrow_mut();stages[stage][0]+=1;stages[stage][1]+=started.elapsed().as_nanos();}
    pub(crate) fn report(&self) {self.weights.borrow_mut().report();self.cost.report();
        for (name,timing) in ["residual_begin","residual_finish","standalone_norm","causal_operations"].into_iter().zip(self.stages.borrow_mut().iter_mut()) {
            eprintln!("RESIDENT STAGE name={name} calls={} milliseconds={:.6}",timing[0],timing[1] as f64/1e6);*timing=[0;2];
        }
        let mut memo=self.memo.borrow_mut();eprintln!("RESIDENT MEMO hits={} misses={} bytes={} limit={}",memo.hits,memo.misses,memo.bytes,memo.limit);memo.hits=0;memo.misses=0;
    }
}
impl<'b,'d> Compute for Device<'b,'d> {
    fn plan_cost(&self,ns:u128) {self.cost.host("planning",ns);}
    fn trace_start(&self) {if self.cost.enabled() {self.backend.set_profiling(true);}}
    fn cost_report(&self,phase:&str) {if self.cost.enabled() {eprintln!("SERVED COST phase={phase}");self.report();}}
    type Value=Value<'d>;type Pass=ResidualBlockPass<'b,'d>;
    fn projects_complete_rows(&self)->bool {true}
    fn prepare(&self,c:&CheckpointAdmission)->Result<()> {
        // A service can retain this owner while replacing a decoder session.
        // Previous steps synchronize before publishing; no old cache may cross sessions.
        *self.memo.borrow_mut()=Memo::default();self.weights.borrow_mut().prepare(c,self.backend)?;
        // Reuse completed private vectors under the backend's existing pressure
        // and size limits. A service's earlier memory admission owns its margin.
        if spark_gpu_probe::driver::vector_pool_from_env()? && !self.backend.vector_pool_configured() {self.backend.configure_vector_pool(1<<30);}
        let local=self.backend.available_memory_bytes()?.saturating_sub(1<<30).min(1<<30);
        self.memo.borrow_mut().limit=self.weights.borrow_mut().shared_budget(local)?;Ok(())}
    fn repeated_projection(&self,c:&CheckpointAdmission,name:&str,x:&Value<'d>,round:bool)->Result<Option<Value<'d>>> {
        if self.memo.borrow().limit==0 {return self.projection(c,name,x,round);}
        // The complete input bits and immutable weight name identify a pure
        // projection. Signed zeros remain distinct. This cache belongs only to
        // this decoder session. Failure or session drop cannot publish state.
        let stamp=self.cost.start(self.backend);let bits=self.backend.read_vector_values(&x.vector)?.iter().map(|v|v.to_bits()).collect::<Vec<_>>();let key=(name.to_owned(),bits,round);
        self.cost.record(||"memo_lookup".into(),stamp,self.backend,0);let stamp=self.cost.start(self.backend);let mut memo=self.memo.borrow_mut();
        if let Some(value)=memo.values.get(&key) {let words=value.len();let mut copy=self.backend.vector(words)?;self.backend.copy_prefix(value,&mut copy,words)?;
            let value=self.value(copy,words)?;self.cost.record(||"memo_copy".into(),stamp,self.backend,0);memo.hits+=1;return Ok(Some(value));}
        let output=self.weights.borrow_mut().apply(name,self.backend,&x.vector,round)?;let words=output.len();memo.misses+=1;
        let bytes=words.checked_add(x.words).and_then(|n|n.checked_mul(4)).and_then(|n|n.checked_add(name.len())).ok_or("projection memo byte count overflow")?;
        if bytes>memo.limit {return Ok(Some(self.value(output,words)?));}
        if memo.bytes.checked_add(bytes).is_none_or(|n|n>memo.limit) {memo.values.clear();memo.bytes=0;}
        let mut copy=self.backend.vector(words)?;self.backend.copy_prefix(&output,&mut copy,words)?;let value=self.value(copy,words)?;
        memo.bytes+=bytes;memo.values.insert(key,output);Ok(Some(value))
    }
    fn embedding(&self,token:u32,streams:usize)->Result<Option<Value<'d>>> {
        let stamp=self.cost.start(self.backend);let out=self.weights.borrow().embedding(self.backend,token,streams)?;let words=out.len();let value=self.value(out,words)?;self.cost.record(||"embedding".into(),stamp,self.backend,0);Ok(Some(value))
    }
    fn projection(&self,_:&CheckpointAdmission,name:&str,x:&Value<'d>,round:bool)->Result<Option<Value<'d>>> {
        let out=self.weights.borrow_mut().apply(name,self.backend,&x.vector,round)?;let words=out.len();Ok(Some(self.value(out,words)?))
    }
    fn projection_rows(&self,_:&CheckpointAdmission,name:&str,input:&[&Value<'d>],round:bool)->Result<Option<Vec<Value<'d>>>> {
        let rows=input.iter().map(|value|&value.vector).collect::<Vec<_>>();
        let values=self.weights.borrow_mut().apply_rows(name,self.backend,&rows,round)?;
        values.into_iter().map(|value|{let words=value.len();self.value(value,words)}).collect::<Result<Vec<_>>>().map(Some)
    }
    fn projection_batch(&self,_:&CheckpointAdmission,input:&[(&str,&Value<'d>)],round:bool)->Result<Option<Vec<Value<'d>>>> {
        let rows=input.iter().map(|(name,value)|(*name,&value.vector)).collect::<Vec<_>>();
        let Some(values)=self.weights.borrow_mut().apply_batch(self.backend,&rows,round)? else {return Ok(None);};
        values.into_iter().map(|value|{let words=value.len();self.value(value,words)}).collect::<Result<Vec<_>>>().map(Some)
    }
    fn upload(&self,v:&[f32])->Result<Value<'d>> {if v.is_empty() {return self.zeros(0);}let stamp=self.cost.start(self.backend);let out=self.value(self.backend.upload_vector(v)?,v.len())?;self.cost.record(||"upload".into(),stamp,self.backend,0);Ok(out)}
    fn read(&self,v:&Value<'d>)->Result<Vec<f32>> {let stamp=self.cost.start(self.backend);let out=if v.words==0 {Vec::new()} else {self.backend.read_vector_values(&v.vector)?};self.weights.borrow().check_exchange()?;self.cost.record(||"readback".into(),stamp,self.backend,0);Ok(out)}
    fn range(&self,v:&Value<'d>,start:usize,len:usize)->Result<Value<'d>> {
        let stamp=self.cost.start(self.backend);need(start.checked_add(len).is_some_and(|n|n<=v.words),"bounded device range differs")?;
        let out=if len==0 {self.backend.vector(1)?} else {v.vector.resident_range(start..start+len)?};let value=self.value(out,len)?;self.cost.record(||"range_view".into(),stamp,self.backend,0);Ok(value)
    }
    fn join(&self,parts:&[&Value<'d>])->Result<Value<'d>> {
        let stamp=self.cost.start(self.backend);let len=parts.iter().try_fold(0usize,|n,v|n.checked_add(v.words).ok_or("bounded join overflow"))?;
        let sources=parts.iter().map(|v|(&v.vector,v.words)).collect::<Vec<_>>();let out=self.backend.resident_join(&sources,&mut self.copies.borrow_mut())?;let value=self.value(out,len)?;self.cost.record(||"join_copy".into(),stamp,self.backend,0);Ok(value)
    }
    fn zeros(&self,len:usize)->Result<Value<'d>> {let stamp=self.cost.start(self.backend);let mut v=self.backend.vector(len.max(1))?;self.backend.zero_vector(&mut v)?;let value=self.value(v,len)?;self.cost.record(||"zero".into(),stamp,self.backend,0);Ok(value)}
    fn operation(&self,s:Shape,x:[Option<&Value<'d>>;4],w:&[f32])->Result<Value<'d>> {
        let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);
        let (lengths,_,units,words)=s.lengths().ok_or("invalid bounded device operation")?;
        for (v,count) in x.iter().zip(lengths) {need(v.map_or(0,|v|v.words)==count,"bounded device operation input extent differs")?;}
        let key=w.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        if !w.is_empty() && !self.parameters.borrow().contains_key(&key) {self.parameters.borrow_mut().insert(key.clone(),self.backend.upload_vector(w)?);}
        let owners=self.parameters.borrow();
        let input=x.map(|v|v.filter(|v|v.words>0).map(|v|&v.vector));
        let v=if s.op==Op::Step {
            let columns=self.backend.causal_operation(Shape {op:Op::StepColumns,..s},input,None)?;
            self.backend.causal_operation(Shape {op:Op::StepLayout,..s},[Some(&columns),None,None,None],None)?
        } else if s.op==Op::Pool {
            let channels=self.backend.causal_operation(Shape {op:Op::PoolChannels,..s},input,owners.get(&key))?;
            self.backend.causal_operation(Shape {op:Op::PoolLayout,..s},[Some(&channels),None,None,None],None)?
        } else if s.op==Op::Conv {
            let channels=self.backend.causal_operation(Shape {op:Op::ConvChannels,..s},input,owners.get(&key))?;
            self.backend.causal_operation(Shape {op:Op::ConvLayout,..s},[Some(&channels),None,None,None],None)?
        } else {self.backend.causal_operation(s,input,owners.get(&key))?};
        let value=self.value(v,units*words)?;self.cost.record(||format!("operation_{:?}",s.op),stamp,self.backend,w.len()*4);self.record(3,started);Ok(value)
    }
    fn norm(&self,name:&str,s:NormShape,x:&Value<'d>,w:&[f32])->Result<Value<'d>> {
        let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);
        if !self.norms.borrow().contains_key(name) {self.norms.borrow_mut().insert(name.into(),self.backend.resident_norm(s,w)?);}
        let input=self.backend.round_bf16_lanes(&x.vector)?;let owners=self.norms.borrow();
        need(owners[name].shape()==s,"resident norm shape changed")?;let norm=owners[name].apply(&input)?;
        let mut out=self.backend.vector(x.words)?;self.backend.copy_prefix(norm.vector(),&mut out,x.words)?;
        let value=self.value(out,x.words)?;self.cost.record(||"norm".into(),stamp,self.backend,w.len()*4);self.record(2,started);Ok(value)
    }
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Value<'d>)->Result<(Self::Pass,Value<'d>)> {
        let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);
        if !self.blocks.borrow().contains_key(&key) {self.blocks.borrow_mut().insert(key,p.upload(self.backend)?);}
        let pass=self.blocks.borrow()[&key].begin(self.backend.round_bf16_lanes(&x.vector)?)?;let words=pass.normalized().vector().len();let mut normalized=self.backend.vector(words)?;
        self.backend.copy_prefix(pass.normalized().vector(),&mut normalized,words)?;let value=self.value(normalized,words)?;self.cost.record(||"residual_begin".into(),stamp,self.backend,(p.mhc().weights().len()+p.mhc().base().len()+p.mhc().scale().len()+p.norm().weight().len())*4);self.record(0,started);Ok((pass,value))
    }
    fn finish(&self,p:Self::Pass,x:&Value<'d>)->Result<Value<'d>> {let started=std::time::Instant::now();let stamp=self.cost.start(self.backend);let combined=p.finish(self.backend.round_bf16_lanes(&x.vector)?)?;
        let mut out=self.backend.vector(combined.len())?;self.backend.copy_prefix(combined.vector(),&mut out,combined.len())?;
        let words=combined.len();let value=self.value(out,words)?;self.cost.record(||"residual_finish".into(),stamp,self.backend,0);self.record(1,started);Ok(value)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&Value<'d>)->Result<Value<'d>> {
        let out=self.weights.borrow_mut().apply(p.component_names()[0],self.backend,&x.vector,true)?;let words=out.len();self.value(out,words)
    }
    fn round(&self,x:&Value<'d>)->Result<Value<'d>> {let stamp=self.cost.start(self.backend);let rounded=self.backend.round_bf16_lanes(&x.vector)?;let mut out=self.backend.vector(x.words)?;
        self.backend.copy_prefix(rounded.vector(),&mut out,x.words)?;let value=self.value(out,x.words)?;self.cost.record(||"round".into(),stamp,self.backend,0);Ok(value)}
    fn mean(&self,x:&Value<'d>,streams:u32,hidden:u32)->Result<Value<'d>> {let mut s=shape(Op::StreamMean);s.heads=streams;s.dim=hidden;self.operation(s,[Some(x),None,None,None],&[])}
}
