//! Complete output-row split. Each row retains the decoder's fixed reduction.
use crate::{Result,bounded_decoder::{Compute,Decoder,State},mxfp8_projection::Mxfp8Projection};
use engine_model::plan::{admission::CheckpointAdmission,layer::Topology};
use spark_gpu_probe::{causal_ops::{Shape,Op},norm_boundary::NormShape};
use std::cell::{Cell,RefCell};
pub trait Exchange {
    fn topology(&self)->Topology;
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>>;
    fn command(&mut self,words:&mut [u8])->Result<()>;
    fn complete(&mut self)->Result<()>;
    fn abort(&mut self) {}
}
impl Exchange for crate::transport::Ring {
    fn topology(&self)->Topology {Topology {rank:self.rank() as u32,parts:self.world() as u32}}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {self.gather_batch_f32(local,rows,1,1)}
    fn command(&mut self,words:&mut [u8])->Result<()> {self.command(words)}
    fn complete(&mut self)->Result<()> {self.barrier()}
    fn abort(&mut self) {self.abort();}
}
pub struct Split<C,E> {compute:C,exchange:RefCell<E>,topology:Topology}
impl<C:Compute,E:Exchange> Split<C,E> {
    pub fn new(compute:C,exchange:E)->Result<Self> {
        let topology=exchange.topology();topology.validate().map_err(|e|e.to_string())?;
        Ok(Self {compute,exchange:RefCell::new(exchange),topology})
    }
    fn gathered(&self,v:&C::Value,rows:usize)->Result<C::Value> {
        let local=self.compute.read(v)?;let full=self.exchange.borrow_mut().gather(&local,rows)?;
        if full.len()!=rows || full.iter().any(|v|!v.is_finite()) {return Err("row gather returned invalid output".into());}
        self.compute.upload(&full)
    }
    fn command(&self,code:u32,token:u32)->Result<()> {let mut words=[0;8];words[..4].copy_from_slice(&code.to_le_bytes());words[4..].copy_from_slice(&token.to_le_bytes());self.exchange.borrow_mut().command(&mut words)}
}
/// Use the shared protocol parser and keep unnamed requests greedy.
/// Service memory admission after the resident load. `allocated` is the backend's
/// actual device allocation (every storage kind, this rank's row ownership, the
/// embedding and retained parameters). The session reservation must fit beside it.
pub fn admit_service_memory(allocated:usize,reservation:usize,budget:usize)->Result<usize> {
    let required=allocated.checked_add(reservation).ok_or("text memory sum overflow")?;
    if required>budget {return Err(format!("text service needs {required} device bytes; budget={budget}; resident={allocated}; session={reservation}; reduce CONTEXT"));}
    Ok(required)
}
#[test]
fn service_memory_counts_resident_weights_and_the_session() {
    assert_eq!(admit_service_memory(60,40,100),Ok(100));
    assert!(admit_service_memory(90,20,100).is_err(),"weights alone fit, weights plus session do not");
    assert!(admit_service_memory(usize::MAX,1,usize::MAX).is_err());
}
pub fn limits(context:usize,output:usize)->Result<spark_serve::Limits> {
    let mut limits=spark_serve::Limits {input_tokens:context.saturating_sub(1),output_tokens:output,context_tokens:context,
        greedy_if_unsampled:true,default_sampling:Some(spark_engine::sample::Sampling::greedy()),default_thinking:false,next_mtp_depth:Some(0),..Default::default()};
    limits.output_bytes=output.saturating_mul(256).clamp(1<<20,16<<20);limits.validate()?;Ok(limits)
}
impl<C:Compute,E:Exchange> Compute for Split<C,E> {
    type Value=C::Value;type Pass=C::Pass;
    fn plan_cost(&self,ns:u128) {self.compute.plan_cost(ns)}
    fn trace_start(&self) {self.compute.trace_start()}
    fn cost_report(&self,phase:&str) {self.compute.cost_report(phase)}
    fn projects_complete_rows(&self)->bool {self.compute.projects_complete_rows()}
    fn embedding(&self,token:u32,streams:usize)->Result<Option<C::Value>> {self.compute.embedding(token,streams)}
    fn prepare(&self,c:&engine_model::plan::admission::CheckpointAdmission)->Result<()> {self.compute.prepare(c)}
    fn projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {self.compute.projection(c,name,x,round)}
    fn projection_rows(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&[&C::Value],round:bool)->Result<Option<Vec<C::Value>>> {self.compute.projection_rows(c,name,x,round)}
    fn projection_batch(&self,c:&CheckpointAdmission,input:&[(&str,&C::Value)],round:bool)->Result<Option<Vec<C::Value>>> {self.compute.projection_batch(c,input,round)}
    fn repeated_projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {self.compute.repeated_projection(c,name,x,round)}
    fn upload(&self,v:&[f32])->Result<C::Value> {self.compute.upload(v)}
    fn read(&self,v:&C::Value)->Result<Vec<f32>> {self.compute.read(v)}
    fn range(&self,v:&C::Value,start:usize,len:usize)->Result<C::Value> {self.compute.range(v,start,len)}
    fn join(&self,v:&[&C::Value])->Result<C::Value> {self.compute.join(v)}
    fn zeros(&self,n:usize)->Result<C::Value> {self.compute.zeros(n)}
    fn operation(&self,mut s:Shape,x:[Option<&C::Value>;4],w:&[f32])->Result<C::Value> {
        if !matches!(s.op,Op::Project|Op::Nvfp4) {return self.compute.operation(s,x,w);}
        let rows=s.rows as usize;let cols=s.cols as usize;let r=crate::rows(rows,self.topology.rank as usize,self.topology.parts as usize);
        let scales=if s.op==Op::Nvfp4 {Some(self.compute.range(x[1].ok_or("row split lacks FP4 scales")?,r.start*(cols/16),r.len()*(cols/16))?)} else {None};
        s.rows=r.len().try_into().map_err(|_|"row split exceeds ABI")?;
        let mut x=x;if let Some(v)=scales.as_ref() {x[1]=Some(v);}
        let local=if r.is_empty() {self.compute.zeros(0)?} else {self.compute.operation(s,x,w.get(r.start*cols..r.end*cols).ok_or("row split weight extent differs")?)?};
        self.gathered(&local,rows)
    }
    fn norm(&self,name:&str,s:NormShape,x:&C::Value,w:&[f32])->Result<C::Value> {self.compute.norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value)->Result<(C::Pass,C::Value)> {self.compute.begin(key,p,x)}
    fn finish(&self,p:C::Pass,x:&C::Value)->Result<C::Value> {self.compute.finish(p,x)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&C::Value)->Result<C::Value> {if self.compute.projects_complete_rows() {return self.compute.mx(p,x);}self.gathered(&self.compute.mx(&p.split(self.topology)?,x)?,p.shape().rows())}
    fn round(&self,x:&C::Value)->Result<C::Value> {self.compute.round(x)}
    fn mean(&self,x:&C::Value,s:u32,h:u32)->Result<C::Value> {self.compute.mean(x,s,h)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&C::Value>;5],w:&[f32])->Result<C::Value> {self.compute.prompt(s,x,w)}
    fn rows_embedding(&self,t:&[u32],n:usize)->Result<Option<C::Value>> {self.compute.rows_embedding(t,n)}
    fn rows_project(&self,c:&CheckpointAdmission,name:&str,x:&C::Value,m:usize,round:bool)->Result<Option<C::Value>> {self.compute.rows_project(c,name,x,m,round)}
    fn rows_experts(&self,a:&[(&str,usize)],b:Option<(&[(&str,usize)],f32)>,x:&C::Value,p:&[f32])->Result<Option<C::Value>> {self.compute.rows_experts(a,b,x,p)}
    fn rows_norm(&self,name:&str,s:NormShape,x:&C::Value,m:usize,w:&[f32])->Result<C::Value> {self.compute.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value,m:usize)->Result<(C::Pass,C::Value)> {self.compute.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:C::Pass,x:&C::Value,m:usize)->Result<C::Value> {self.compute.rows_finish(p,x,m)}
}
impl<C:Compute> Compute for &C {
    type Value=C::Value;type Pass=C::Pass;
    fn plan_cost(&self,ns:u128) {(**self).plan_cost(ns)}
    fn trace_start(&self) {(**self).trace_start()}
    fn cost_report(&self,phase:&str) {(**self).cost_report(phase)}
    fn projects_complete_rows(&self)->bool {(**self).projects_complete_rows()}
    fn embedding(&self,token:u32,streams:usize)->Result<Option<C::Value>> {(**self).embedding(token,streams)}
    fn prepare(&self,c:&engine_model::plan::admission::CheckpointAdmission)->Result<()> {(**self).prepare(c)}
    fn projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {(**self).projection(c,name,x,round)}
    fn projection_rows(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&[&C::Value],round:bool)->Result<Option<Vec<C::Value>>> {(**self).projection_rows(c,name,x,round)}
    fn projection_batch(&self,c:&CheckpointAdmission,input:&[(&str,&C::Value)],round:bool)->Result<Option<Vec<C::Value>>> {(**self).projection_batch(c,input,round)}
    fn repeated_projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {(**self).repeated_projection(c,name,x,round)}
    fn upload(&self,v:&[f32])->Result<C::Value> {(**self).upload(v)}
    fn read(&self,v:&C::Value)->Result<Vec<f32>> {(**self).read(v)}
    fn range(&self,v:&C::Value,s:usize,n:usize)->Result<C::Value> {(**self).range(v,s,n)}
    fn join(&self,v:&[&C::Value])->Result<C::Value> {(**self).join(v)}
    fn zeros(&self,n:usize)->Result<C::Value> {(**self).zeros(n)}
    fn operation(&self,s:Shape,x:[Option<&C::Value>;4],w:&[f32])->Result<C::Value> {(**self).operation(s,x,w)}
    fn norm(&self,name:&str,s:NormShape,x:&C::Value,w:&[f32])->Result<C::Value> {(**self).norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value)->Result<(C::Pass,C::Value)> {(**self).begin(key,p,x)}
    fn finish(&self,p:C::Pass,x:&C::Value)->Result<C::Value> {(**self).finish(p,x)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&C::Value)->Result<C::Value> {(**self).mx(p,x)}
    fn round(&self,x:&C::Value)->Result<C::Value> {(**self).round(x)}
    fn mean(&self,x:&C::Value,s:u32,h:u32)->Result<C::Value> {(**self).mean(x,s,h)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&C::Value>;5],w:&[f32])->Result<C::Value> {(**self).prompt(s,x,w)}
    fn rows_embedding(&self,t:&[u32],n:usize)->Result<Option<C::Value>> {(**self).rows_embedding(t,n)}
    fn rows_project(&self,c:&CheckpointAdmission,name:&str,x:&C::Value,m:usize,round:bool)->Result<Option<C::Value>> {(**self).rows_project(c,name,x,m,round)}
    fn rows_experts(&self,a:&[(&str,usize)],b:Option<(&[(&str,usize)],f32)>,x:&C::Value,p:&[f32])->Result<Option<C::Value>> {(**self).rows_experts(a,b,x,p)}
    fn rows_norm(&self,name:&str,s:NormShape,x:&C::Value,m:usize,w:&[f32])->Result<C::Value> {(**self).rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value,m:usize)->Result<(C::Pass,C::Value)> {(**self).rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:C::Pass,x:&C::Value,m:usize)->Result<C::Value> {(**self).rows_finish(p,x,m)}
}
/// The server needs only each group's last row through the head.
fn prompt_group<C:Compute>(decoder:&Decoder<'_,C>,tokens:&[u32],old:&State<C::Value>)->Result<(Vec<Vec<f32>>,State<C::Value>)> {
    decoder.prefill_rows(tokens,old,crate::bounded_decoder::HeadRows::Last)
}
/// One active request. The checkpoint and projection owners outlive all sessions.
pub struct Factory<'f,C,E> {checkpoint:&'f CheckpointAdmission,split:Split<C,E>,context:usize,eos:Vec<u32>,reservation:usize,active:Cell<bool>,failed:Cell<bool>}
impl<'f,C:Compute,E:Exchange> Factory<'f,C,E> {
    pub fn controller(&self)->bool {self.split.topology.rank.checked_sub(1).is_none()}
    pub fn new(checkpoint:&'f CheckpointAdmission,split:Split<C,E>,context:usize,eos:Vec<u32>)->Result<Self> {
        Decoder::new(checkpoint,&split,context as u64)?;
        let state=engine_model::plan::state::StatePlan::new(checkpoint.description(),Topology {rank:0,parts:1},context as u64,isize::MAX as usize).map_err(|e|e.to_string())?;
        let largest=checkpoint.headers().weights().values().filter(|w|w.logical().shape.len()==2).map(|w|w.logical().shape.iter().try_fold(4usize,|n,&v|n.checked_mul(v as usize)).ok_or_else(||"service weight size overflow".to_string())).collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
        // One group's workspace plus this session's persistent expanded rows.
        let extra=crate::bounded_decoder::group_workspace_bytes(checkpoint,context)?.checked_add(crate::bounded_decoder::expanded_bytes(checkpoint,context)).ok_or("prompt tile reservation overflow")?;
        let reservation=state.working_bytes().checked_mul(2).and_then(|v|largest.checked_mul(2).and_then(|w|v.checked_add(w))).and_then(|v|v.checked_add(extra))
            .and_then(|v|v.checked_add(checkpoint.description().vocab as usize*16)).ok_or("service session size overflow")?;
        if eos.is_empty() || eos.iter().any(|&id|u64::from(id)>=checkpoint.description().vocab) {return Err("service EOS set is invalid".into());}
        Ok(Self {checkpoint,split,context,eos,reservation,active:Cell::new(false),failed:Cell::new(false)})
    }
    pub fn shutdown(&self)->Result<()> {
        if self.failed.get() {self.split.exchange.borrow_mut().abort();return Err("row split service failed".into());}
        if self.active.get() {return Err("service shutdown has a live session".into());}self.split.command(0,0)
    }
    pub fn worker(&self)->Result<()> {
        let mut decoder=None;let mut state=None;
        loop {let mut words=[0;8];self.split.exchange.borrow_mut().command(&mut words)?;
            let code=u32::from_le_bytes(words[..4].try_into().unwrap());let token=u32::from_le_bytes(words[4..].try_into().unwrap());
            match code {
                0=>{if state.is_some() {return Err("worker stopped with a live session".into());}return Ok(());},
                1=>{if state.is_some() {return Err("worker session already exists".into());}let started=std::time::Instant::now();let next=Decoder::new(self.checkpoint,&self.split,self.context as u64)?;state=Some(next.zero_state()?);decoder=Some(next);self.split.exchange.borrow_mut().complete()?;eprintln!("SERVED SESSION rank={} setup_seconds={:.9}",self.split.topology.rank,started.elapsed().as_secs_f64());self.split.cost_report("setup");self.split.trace_start();},
                2|4=>{let started=std::time::Instant::now();let old=state.as_ref().ok_or("worker has no session")?;let position=old.position;let (_,_,next)=decoder.as_ref().ok_or("worker has no decoder")?.step(token,old,|_,_|Ok(()))?;self.split.exchange.borrow_mut().complete()?;state=Some(next);let phase=if code==4 {"prefill"} else {"decode"};eprintln!("SERVED STEP rank={} phase={phase} position={position} seconds={:.9}",self.split.topology.rank,started.elapsed().as_secs_f64());self.split.cost_report(phase);},
                5=>{let started=std::time::Instant::now();let rows=token as usize;if rows==0 || rows>crate::bounded_decoder::PROMPT_GROUP {return Err("worker prompt group exceeds its bound".into());}
                    let mut bytes=vec![0;rows*4];self.split.exchange.borrow_mut().command(&mut bytes)?;let tokens=bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
                    let old=state.as_ref().ok_or("worker has no prompt session")?;let position=old.position;
                    let (_,next)=prompt_group(decoder.as_ref().ok_or("worker has no prompt decoder")?,&tokens,old)?;self.split.exchange.borrow_mut().complete()?;state=Some(next);
                    eprintln!("SERVED GROUP rank={} phase=prefill position={position} rows={rows} seconds={:.9}",self.split.topology.rank,started.elapsed().as_secs_f64());self.split.cost_report("prefill");},
                3=>{state=None;decoder=None;self.split.exchange.borrow_mut().complete()?;},
                _=>return Err("unknown row split service command".into()),
            }
        }
    }
}
pub struct Session<'a,C:Compute,E:Exchange> {factory:&'a Factory<'a,C,E>,decoder:Decoder<'a,&'a Split<C,E>>,state:Option<State<C::Value>>}
impl<C:Compute,E:Exchange> spark_serve::Engine for Factory<'_,C,E> {
    type Session<'a>=Session<'a,C,E> where Self:'a;
    fn session(&self)->Result<Self::Session<'_>> {
        if self.failed.get() || self.active.replace(true) {return Err("row split service has no free session".into());}
        let result=(|| {let started=std::time::Instant::now();self.split.command(1,0)?;let decoder=Decoder::new(self.checkpoint,&self.split,self.context as u64)?;
            let state=decoder.zero_state()?;self.split.exchange.borrow_mut().complete()?;eprintln!("SERVED SESSION rank={} setup_seconds={:.9}",self.split.topology.rank,started.elapsed().as_secs_f64());self.split.cost_report("setup");self.split.trace_start();Ok(Session {factory:self,decoder,state:Some(state)})})();
        if result.is_err() {self.failed.set(true);self.active.set(false);self.split.exchange.borrow_mut().abort();}result
    }
    fn context_limit(&self)->usize {self.context}
    fn vocab_size(&self)->usize {self.checkpoint.description().vocab as usize}
    fn eos_token_ids(&self)->&[u32] {&self.eos}
    fn session_reservation_bytes(&self)->Option<usize> {Some(self.reservation)}
    fn shared_prompt_limit(&self)->usize {0}
}
impl<C:Compute,E:Exchange> Session<'_,C,E> {
    fn prompt(&mut self,tokens:&[u32])->Result<Vec<f32>> {
        if tokens.is_empty() {return Err("empty row split prompt".into());}let mut logits=Vec::new();
        for tokens in tokens.chunks(crate::bounded_decoder::PROMPT_GROUP) {
            if self.factory.failed.get() {return Err("row split service previously failed".into());}
            let old=self.state.as_ref().ok_or("row split prompt has no state")?;
            if old.position.checked_add(tokens.len() as u64).is_none_or(|n|n>self.factory.context as u64)
                || tokens.iter().any(|&token|u64::from(token)>=self.factory.checkpoint.description().vocab) {return Err("row split prompt exceeds vocabulary or context".into());}
            let started=std::time::Instant::now();let position=old.position;self.factory.failed.set(true);self.factory.split.command(5,tokens.len() as u32)?;
            let mut bytes=tokens.iter().flat_map(|t|t.to_le_bytes()).collect::<Vec<_>>();self.factory.split.exchange.borrow_mut().command(&mut bytes)?;
            let (mut rows,next)=prompt_group(&self.decoder,tokens,old)?;self.factory.split.exchange.borrow_mut().complete()?;
            self.state=Some(next);self.factory.failed.set(false);logits=rows.pop().ok_or("prompt group has no final logits")?;
            eprintln!("SERVED GROUP rank={} phase=prefill position={position} rows={} seconds={:.9}",self.factory.split.topology.rank,tokens.len(),started.elapsed().as_secs_f64());self.factory.split.cost_report("prefill");
        }Ok(logits)
    }
    fn token(&mut self,token:u32,prefill:bool)->Result<Vec<f32>> {
        let started=std::time::Instant::now();
        if self.factory.failed.get() {return Err("row split service previously failed".into());}
        let old=self.state.as_ref().ok_or("row split session has no state")?;
        if old.position>=self.factory.context as u64 || u64::from(token)>=self.factory.checkpoint.description().vocab {return Err("row split request exceeds vocabulary or context".into());}
        let position=old.position;self.factory.failed.set(true);self.factory.split.command(if prefill {4} else {2},token)?;
        let (logits,_,next)=self.decoder.step(token,old,|_,_|Ok(()))?;self.factory.split.exchange.borrow_mut().complete()?;
        self.state=Some(next);self.factory.failed.set(false);let phase=if prefill {"prefill"} else {"decode"};eprintln!("SERVED STEP rank={} phase={phase} position={position} seconds={:.9}",self.factory.split.topology.rank,started.elapsed().as_secs_f64());self.factory.split.cost_report(phase);Ok(logits)
    }
}
impl<C:Compute,E:Exchange> spark_serve::Session for Session<'_,C,E> {
    fn configure_request(&mut self,r:&spark_serve::protocol::Request)->Result<()> {
        if r.mtp_depth.is_some_and(|v|v!=0) || r.draft.is_some_and(|v|!matches!(v,spark_serve::protocol::DraftMode::Auto|spark_serve::protocol::DraftMode::Plain)) {return Err("row split service supports plain decoding only".into());}Ok(())
    }
    fn step(&mut self,token:u32)->Result<Vec<f32>> {self.token(token,false)}
    fn prefill(&mut self,tokens:&[u32])->Result<Vec<f32>> {self.prompt(tokens)}
    fn prefill_advance(&mut self,tokens:&[u32])->Result<()> {self.prompt(tokens).map(|_|())}
    fn prefill_chunk_size(&self,_:usize)->usize {crate::bounded_decoder::PROMPT_GROUP}
    fn prefill_chunk_min(&self)->usize {crate::bounded_decoder::PROMPT_ROWS}
    fn prefill_slice(&mut self,tokens:&[u32],_final_chunk:bool,_budget:Option<std::time::Duration>,_prompt_end:usize)->Result<Option<Vec<f32>>> {
        self.prompt(tokens).map(Some)
    }
    fn can_fork(&self)->bool {false}
    fn can_fork_prefix(&self,_:usize)->bool {false}
}
impl<C:Compute,E:Exchange> Drop for Session<'_,C,E> {
    fn drop(&mut self) {if !self.factory.failed.get() {let result=self.factory.split.command(3,0).and_then(|()|self.factory.split.exchange.borrow_mut().complete());if result.is_err() {self.factory.failed.set(true);}}
        if self.factory.failed.get() {self.factory.split.exchange.borrow_mut().abort();}
        self.state=None;self.factory.active.set(false);}
}
