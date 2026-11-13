//! Complete output-row split. Each row retains the decoder's fixed reduction.
use crate::{Result,bounded_decoder::{Compute,Decoder,State,Verify},mxfp8_projection::Mxfp8Projection};
use engine_model::plan::{admission::CheckpointAdmission,layer::Topology};
use spark_gpu_probe::{causal_ops::{Shape,Op},norm_boundary::NormShape};
use std::{cell::{Cell,RefCell},collections::BTreeMap};
mod batch;
/// In the row count of a prompt group command: this group ends its prompt (decode steps follow).
const LAST_GROUP:u32=1<<31;
/// Prompts a service keeps on each rank at once (`Session::fork`). As many as it runs chats.
pub const KEPT_PROMPTS:usize=8;
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
    // The barrier of a served step rides the pair's mailbox, not the control stream (transport.rs step_barrier).
    fn complete(&mut self)->Result<()> {self.step_barrier()}
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
    fn command(&self,code:u32,token:u32)->Result<()> {self.command_for(code,token,0)}
    fn command_for(&self,code:u32,token:u32,id:u64)->Result<()> {let mut words=[0;16];words[..4].copy_from_slice(&code.to_le_bytes());words[4..8].copy_from_slice(&token.to_le_bytes());words[8..].copy_from_slice(&id.to_le_bytes());self.exchange.borrow_mut().command(&mut words)}
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
fn the_text_service_reads_no_thinking_switch_from_message_text() {
    // GLM's chat template has no /think or /no_think switch; the serve tokenizer tests pin the render.
    let limits=limits(4096,256).unwrap();assert!(!limits.thinking_soft_switch && !limits.default_thinking);
}
#[test]
fn service_memory_counts_resident_weights_and_the_session() {
    assert_eq!(admit_service_memory(60,40,100),Ok(100));
    assert!(admit_service_memory(90,20,100).is_err(),"weights alone fit, weights plus session do not");
    assert!(admit_service_memory(usize::MAX,1,usize::MAX).is_err());
}
pub fn limits(context:usize,output:usize)->Result<spark_serve::Limits> {
    let mut limits=spark_serve::Limits {input_tokens:context.saturating_sub(1),output_tokens:output,context_tokens:context,
        greedy_if_unsampled:true,default_sampling:Some(spark_engine::sample::Sampling::greedy()),default_thinking:false,
        // The GLM chat template has no /think or /no_think switch: message text never changes a request.
        thinking_soft_switch:false,next_mtp_depth:Some(0),
        generation_timeout:spark_serve::Limits::MAX_GENERATION_TIMEOUT,..Default::default()};
    limits.output_bytes=output.saturating_mul(256).clamp(1<<20,16<<20);limits.validate()?;Ok(limits)
}
impl<C:Compute,E:Exchange> Compute for Split<C,E> {
    fn work<T>(&self,run:impl FnOnce()->Result<T>)->Result<T> {self.compute.work(run)}
    fn resident_weights_ready(&self)->bool {self.compute.resident_weights_ready()}
    // Forwarded by hand: a default here would print zeros for the device.
    fn capture_counts(&self)->[u64;4] {self.compute.capture_counts()}
    fn fatal_error(&self)->Option<String> {self.compute.fatal_error()}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {self.compute.memory_bytes()}
    fn vector_allocations(&self)->Option<(usize,usize)> {self.compute.vector_allocations()}
    fn vector_pool_counts(&self)->Option<(usize,usize,usize)> {self.compute.vector_pool_counts()}
    type Value=C::Value;type Pass=C::Pass;
    fn plan_cost(&self,name:&str,ns:u128) {self.compute.plan_cost(name,ns)}
    fn serving_trace(&self)->bool {self.compute.serving_trace()}
    fn profile_tick_begin(&self,tick:u64) {self.compute.profile_tick_begin(tick)}
    fn profile_tick_end(&self) {self.compute.profile_tick_end()}
    fn trace_start(&self) {self.compute.trace_start()}
    fn cost_report(&self,phase:&str) {self.compute.cost_report(phase)}
    fn mhc_row_launches(&self)->u64 {self.compute.mhc_row_launches()}
    fn projects_complete_rows(&self)->bool {self.compute.projects_complete_rows()}
    fn sparse_topology(&self)->Topology {self.compute.sparse_topology()}
    fn owned_expansion(&self,name:&str,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Option<C::Value>> {self.compute.owned_expansion(name,input,tokens,layout)}
    fn owned_expansion_rows(&self,name:&str,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Option<C::Value>> {self.compute.owned_expansion_rows(name,input,tokens,layout)}
    fn gather_attended(&self,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<C::Value> {self.compute.gather_attended(input,tokens,layout)}
    fn embedding(&self,token:u32,streams:usize)->Result<Option<C::Value>> {self.compute.embedding(token,streams)}
    fn indirect_attention(&self,shape:Shape,parts:&[&C::Value],query:&C::Value)->Result<Option<C::Value>> {self.compute.indirect_attention(shape,parts,query)}
    fn has_latent_absorption(&self,name:&str)->bool {self.compute.has_latent_absorption(name)}
    fn latent_absorption(&self,name:&str,shape:Shape,query:&C::Value,latent:&C::Value,count:&C::Value,slots:usize)->Result<C::Value> {self.compute.latent_absorption(name,shape,query,latent,count,slots)}
    fn latent_absorption_rows(&self,name:&str,shape:spark_gpu_probe::prompt_ops::Shape,query:&C::Value,latent:&C::Value,selected:&C::Value)->Result<C::Value> {self.compute.latent_absorption_rows(name,shape,query,latent,selected)}
    fn planar_kda(&self,shape:Shape)->bool {self.compute.planar_kda(shape)}
    fn prepare(&self,c:&engine_model::plan::admission::CheckpointAdmission)->Result<()> {self.compute.prepare(c)}
    fn prepare_beside(&self,c:&engine_model::plan::admission::CheckpointAdmission,live:usize)->Result<()> {self.compute.prepare_beside(c,live)}
    fn prepare_names(&self,c:&engine_model::plan::admission::CheckpointAdmission,names:&[String])->Result<()> {self.compute.prepare_names(c,names)}
    fn projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {self.compute.projection(c,name,x,round)}
    /// Only a backend that holds complete rows can pick them where they are; any other reads the logits.
    fn rows_greedy(&self,logits:&C::Value,rows:usize)->Result<Option<Vec<u32>>> {if self.compute.projects_complete_rows() {self.compute.rows_greedy(logits,rows)} else {Ok(None)}}
    fn greedy_projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value)->Result<u32> {
        if self.compute.projects_complete_rows() {return self.compute.greedy_projection(c,name,x);}
        let head=Mxfp8Projection::matrix(c,name,spark_gpu_probe::norm_boundary::WeightType::F32)?;let values=self.mx(&head,x)?;
        crate::bounded_decoder::greedy_logits(&self.read(&values)?,head.shape().rows())
    }
    fn projection_rows(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&[&C::Value],round:bool)->Result<Option<Vec<C::Value>>> {self.compute.projection_rows(c,name,x,round)}
    fn projection_batch(&self,c:&CheckpointAdmission,input:&[(&str,&C::Value)],round:bool)->Result<Option<Vec<C::Value>>> {self.compute.projection_batch(c,input,round)}
    fn projection_stage(&self,c:&CheckpointAdmission,jobs:&[crate::projection_stage::Job<'_,C::Value>])->Result<Option<Vec<C::Value>>> {self.compute.projection_stage(c,jobs)}
    fn routed_experts(&self,steps:&[engine_model::plan::layer::Step],ids:&C::Value,input:&C::Value)->Result<Option<Vec<C::Value>>> {self.compute.routed_experts(steps,ids,input)}
    fn routed_fold(&self,steps:&[engine_model::plan::layer::Step],ids:&C::Value,input:&C::Value,weights:&C::Value,shared:Option<&C::Value>,
        fold:spark_gpu_probe::causal_ops::Shape)->Result<Option<C::Value>> {self.compute.routed_fold(steps,ids,input,weights,shared,fold)}
    fn folds_routed(&self)->bool {self.compute.folds_routed()}
    fn projection_origin(&self,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {self.compute.projection_origin(name,x,round)}
    fn projection_origins(&self,name:&str,rows:&C::Value,width:usize,slots:&[usize],round:bool)->Result<Vec<Option<C::Value>>> {self.compute.projection_origins(name,rows,width,slots,round)}
    fn prepare_origins(&self,name:&str,rows:&C::Value,width:usize,round:bool)->Result<()> {self.compute.prepare_origins(name,rows,width,round)}
    fn remember_projections(&self,name:&str,round:bool,kept:&[(&C::Value,&C::Value)])->Result<()> {self.compute.remember_projections(name,round,kept)}
    fn repeated_projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {self.compute.repeated_projection(c,name,x,round)}
    fn upload(&self,v:&[f32])->Result<C::Value> {self.compute.upload(v)}
    fn upload_known(&self,v:&[f32])->Result<C::Value> {self.compute.upload_known(v)}
    fn read(&self,v:&C::Value)->Result<Vec<f32>> {self.compute.read(v)}
    fn read_pair(&self,first:&C::Value,second:&C::Value)->Result<(Vec<f32>,Vec<f32>)> {self.compute.read_pair(first,second)}
    fn select_known(&self,ids:&C::Value,count:&C::Value,row:&[f32])->Result<()> {self.compute.select_known(ids,count,row)}
    fn range(&self,v:&C::Value,start:usize,len:usize)->Result<C::Value> {self.compute.range(v,start,len)}
    fn words(&self,v:&C::Value)->usize {self.compute.words(v)}
    fn join(&self,v:&[&C::Value])->Result<C::Value> {self.compute.join(v)}
    fn cache_append(&self,old:&C::Value,new:&C::Value,rows:usize,width:usize,room:usize)->Result<C::Value> {self.compute.cache_append(old,new,rows,width,room)}
    fn join_reserved(&self,v:&[&C::Value],reserve:usize)->Result<C::Value> {self.compute.join_reserved(v,reserve)}
    fn zeros(&self,n:usize)->Result<C::Value> {self.compute.zeros(n)}
    fn gather_reserved(&self,s:Shape,x:[Option<&C::Value>;4],reserve:usize)->Result<C::Value> {self.compute.gather_reserved(s,x,reserve)}
    fn gather_absorbed(&self,s:Shape,x:[Option<&C::Value>;4],reserve:usize,expected:f32)->Result<C::Value> {self.compute.gather_absorbed(s,x,reserve,expected)}
    fn operation_prepared(&self,s:Shape,x:[Option<&C::Value>;4],w:&std::rc::Rc<crate::bounded_decoder::StaticParameters>)->Result<C::Value> {
        if matches!(s.op,Op::Project|Op::Nvfp4) {self.operation(s,x,w.values())} else {self.compute.operation_prepared(s,x,w)}
    }
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
    // CHAIN-SHADOW: forward the next stage's name; the trait default would drop it.
    fn begin_next(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value,next:&[&str])->Result<(C::Pass,C::Value)> {self.compute.begin_next(key,p,x,next)}
    fn finish(&self,p:C::Pass,x:&C::Value)->Result<C::Value> {self.compute.finish(p,x)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&C::Value)->Result<C::Value> {if self.compute.projects_complete_rows() {return self.compute.mx(p,x);}self.gathered(&self.compute.mx(&p.split(self.topology)?,x)?,p.shape().rows())}
    fn round(&self,x:&C::Value)->Result<C::Value> {self.compute.round(x)}
    fn mean(&self,x:&C::Value,s:u32,h:u32)->Result<C::Value> {self.compute.mean(x,s,h)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&C::Value>;5],w:&[f32])->Result<C::Value> {self.compute.prompt(s,x,w)}
    fn rows_embedding(&self,t:&[u32],n:usize)->Result<Option<C::Value>> {self.compute.rows_embedding(t,n)}
    fn rows_project(&self,c:&CheckpointAdmission,name:&str,x:&C::Value,m:usize,round:bool)->Result<Option<C::Value>> {self.compute.rows_project(c,name,x,m,round)}
    fn rows_stage(&self,c:&CheckpointAdmission,jobs:&[(&str,&C::Value,bool)],m:usize)->Result<Option<Vec<C::Value>>> {self.compute.rows_stage(c,jobs,m)}
    fn batch_rows(&self,chats:usize,state_bytes:usize)->Result<()> {self.compute.batch_rows(chats,state_bytes)}
    fn rows_experts(&self,a:&[(&str,usize)],b:Option<(&[(&str,usize)],f32)>,x:&C::Value,p:&[f32])->Result<Option<C::Value>> {self.compute.rows_experts(a,b,x,p)}
    fn rows_routed(&self,steps:&[engine_model::plan::layer::Step],bound:Option<f32>,ids:&C::Value,x:&C::Value,m:usize)->Result<Option<C::Value>> {self.compute.rows_routed(steps,bound,ids,x,m)}
    fn routes_rows(&self,m:usize)->bool {self.compute.routes_rows(m)}
    fn note_batch_experts(&self,picks:usize,distinct:usize) {self.compute.note_batch_experts(picks,distinct)}
    fn rows_release(&self)->Result<()> {self.compute.rows_release()}
    fn rows_keep(&self)->Result<()> {self.compute.rows_keep()}
    fn rows_unrecorded(&self,p:bool)->Result<bool> {self.compute.rows_unrecorded(p)}
    fn prompt_end(&self)->Result<()> {self.compute.prompt_end()}
    fn prompt_seeds(&self)->bool {self.compute.prompt_seeds()}
    fn rows_charge(&self,bytes:usize) {self.compute.rows_charge(bytes)}
    fn decode_selection_reads(&self,reads:bool) {self.compute.decode_selection_reads(reads)}
    fn rows_peak_begin(&self)->usize {self.compute.rows_peak_begin()}
    fn rows_peak_end(&self,whole:usize)->usize {self.compute.rows_peak_end(whole)}
    fn rows_reserve(&self,words:usize)->Result<C::Value> {self.compute.rows_reserve(words)}
    fn rows_fill(&self,target:&mut C::Value,offset:usize,part:&C::Value,expected:usize)->Result<()> {self.compute.rows_fill(target,offset,part,expected)}
    fn rows_append(&self,held:Option<&C::Value>,rows:usize,new:&C::Value,width:usize)->Result<C::Value> {self.compute.rows_append(held,rows,new,width)}
    fn rows_extend(&self,held:Option<&C::Value>,rows:usize,new:&C::Value,width:usize,room:usize)->Result<C::Value> {self.compute.rows_extend(held,rows,new,width,room)}
    fn rows_norm(&self,name:&str,s:NormShape,x:&C::Value,m:usize,w:&[f32])->Result<C::Value> {self.compute.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value,m:usize)->Result<(C::Pass,C::Value)> {self.compute.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:C::Pass,x:&C::Value,m:usize)->Result<C::Value> {self.compute.rows_finish(p,x,m)}
}
impl<C:Compute> Compute for &C {
    fn work<T>(&self,run:impl FnOnce()->Result<T>)->Result<T> {(**self).work(run)}
    fn resident_weights_ready(&self)->bool {(**self).resident_weights_ready()}
    fn capture_counts(&self)->[u64;4] {(**self).capture_counts()}
    fn fatal_error(&self)->Option<String> {(**self).fatal_error()}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {(**self).memory_bytes()}
    fn vector_allocations(&self)->Option<(usize,usize)> {(**self).vector_allocations()}
    fn vector_pool_counts(&self)->Option<(usize,usize,usize)> {(**self).vector_pool_counts()}
    type Value=C::Value;type Pass=C::Pass;
    fn plan_cost(&self,name:&str,ns:u128) {(**self).plan_cost(name,ns)}
    fn serving_trace(&self)->bool {(**self).serving_trace()}
    fn profile_tick_begin(&self,tick:u64) {(**self).profile_tick_begin(tick)}
    fn profile_tick_end(&self) {(**self).profile_tick_end()}
    fn trace_start(&self) {(**self).trace_start()}
    fn cost_report(&self,phase:&str) {(**self).cost_report(phase)}
    fn mhc_row_launches(&self)->u64 {(**self).mhc_row_launches()}
    fn projects_complete_rows(&self)->bool {(**self).projects_complete_rows()}
    fn rows_greedy(&self,logits:&C::Value,rows:usize)->Result<Option<Vec<u32>>> {(**self).rows_greedy(logits,rows)}
    fn greedy_projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value)->Result<u32> {(**self).greedy_projection(c,name,x)}
    fn sparse_topology(&self)->Topology {(**self).sparse_topology()}
    fn owned_expansion(&self,name:&str,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Option<C::Value>> {(**self).owned_expansion(name,input,tokens,layout)}
    fn owned_expansion_rows(&self,name:&str,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Option<C::Value>> {(**self).owned_expansion_rows(name,input,tokens,layout)}
    fn gather_attended(&self,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<C::Value> {(**self).gather_attended(input,tokens,layout)}
    fn embedding(&self,token:u32,streams:usize)->Result<Option<C::Value>> {(**self).embedding(token,streams)}
    fn indirect_attention(&self,shape:Shape,parts:&[&C::Value],query:&C::Value)->Result<Option<C::Value>> {(**self).indirect_attention(shape,parts,query)}
    fn has_latent_absorption(&self,name:&str)->bool {(**self).has_latent_absorption(name)}
    fn latent_absorption(&self,name:&str,shape:Shape,query:&C::Value,latent:&C::Value,count:&C::Value,slots:usize)->Result<C::Value> {(**self).latent_absorption(name,shape,query,latent,count,slots)}
    fn latent_absorption_rows(&self,name:&str,shape:spark_gpu_probe::prompt_ops::Shape,query:&C::Value,latent:&C::Value,selected:&C::Value)->Result<C::Value> {(**self).latent_absorption_rows(name,shape,query,latent,selected)}
    fn planar_kda(&self,shape:Shape)->bool {(**self).planar_kda(shape)}
    fn prepare(&self,c:&engine_model::plan::admission::CheckpointAdmission)->Result<()> {(**self).prepare(c)}
    fn prepare_beside(&self,c:&engine_model::plan::admission::CheckpointAdmission,live:usize)->Result<()> {(**self).prepare_beside(c,live)}
    fn prepare_names(&self,c:&engine_model::plan::admission::CheckpointAdmission,names:&[String])->Result<()> {(**self).prepare_names(c,names)}
    fn projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {(**self).projection(c,name,x,round)}
    fn projection_rows(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&[&C::Value],round:bool)->Result<Option<Vec<C::Value>>> {(**self).projection_rows(c,name,x,round)}
    fn projection_batch(&self,c:&CheckpointAdmission,input:&[(&str,&C::Value)],round:bool)->Result<Option<Vec<C::Value>>> {(**self).projection_batch(c,input,round)}
    fn projection_stage(&self,c:&CheckpointAdmission,jobs:&[crate::projection_stage::Job<'_,C::Value>])->Result<Option<Vec<C::Value>>> {(**self).projection_stage(c,jobs)}
    fn routed_experts(&self,steps:&[engine_model::plan::layer::Step],ids:&C::Value,input:&C::Value)->Result<Option<Vec<C::Value>>> {(**self).routed_experts(steps,ids,input)}
    fn routed_fold(&self,steps:&[engine_model::plan::layer::Step],ids:&C::Value,input:&C::Value,weights:&C::Value,shared:Option<&C::Value>,
        fold:spark_gpu_probe::causal_ops::Shape)->Result<Option<C::Value>> {(**self).routed_fold(steps,ids,input,weights,shared,fold)}
    fn folds_routed(&self)->bool {(**self).folds_routed()}
    fn projection_origin(&self,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {(**self).projection_origin(name,x,round)}
    fn projection_origins(&self,name:&str,rows:&C::Value,width:usize,slots:&[usize],round:bool)->Result<Vec<Option<C::Value>>> {(**self).projection_origins(name,rows,width,slots,round)}
    fn prepare_origins(&self,name:&str,rows:&C::Value,width:usize,round:bool)->Result<()> {(**self).prepare_origins(name,rows,width,round)}
    fn remember_projections(&self,name:&str,round:bool,kept:&[(&C::Value,&C::Value)])->Result<()> {(**self).remember_projections(name,round,kept)}
    fn repeated_projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {(**self).repeated_projection(c,name,x,round)}
    fn upload(&self,v:&[f32])->Result<C::Value> {(**self).upload(v)}
    fn upload_known(&self,v:&[f32])->Result<C::Value> {(**self).upload_known(v)}
    fn read(&self,v:&C::Value)->Result<Vec<f32>> {(**self).read(v)}
    fn read_pair(&self,first:&C::Value,second:&C::Value)->Result<(Vec<f32>,Vec<f32>)> {(**self).read_pair(first,second)}
    fn select_known(&self,ids:&C::Value,count:&C::Value,row:&[f32])->Result<()> {(**self).select_known(ids,count,row)}
    fn range(&self,v:&C::Value,s:usize,n:usize)->Result<C::Value> {(**self).range(v,s,n)}
    fn words(&self,v:&C::Value)->usize {(**self).words(v)}
    fn join(&self,v:&[&C::Value])->Result<C::Value> {(**self).join(v)}
    fn cache_append(&self,old:&C::Value,new:&C::Value,rows:usize,width:usize,room:usize)->Result<C::Value> {(**self).cache_append(old,new,rows,width,room)}
    fn join_reserved(&self,v:&[&C::Value],reserve:usize)->Result<C::Value> {(**self).join_reserved(v,reserve)}
    fn zeros(&self,n:usize)->Result<C::Value> {(**self).zeros(n)}
    fn gather_reserved(&self,s:Shape,x:[Option<&C::Value>;4],reserve:usize)->Result<C::Value> {(**self).gather_reserved(s,x,reserve)}
    fn gather_absorbed(&self,s:Shape,x:[Option<&C::Value>;4],reserve:usize,expected:f32)->Result<C::Value> {(**self).gather_absorbed(s,x,reserve,expected)}
    fn operation(&self,s:Shape,x:[Option<&C::Value>;4],w:&[f32])->Result<C::Value> {(**self).operation(s,x,w)}
    fn operation_prepared(&self,s:Shape,x:[Option<&C::Value>;4],w:&std::rc::Rc<crate::bounded_decoder::StaticParameters>)->Result<C::Value> {(**self).operation_prepared(s,x,w)}
    fn norm(&self,name:&str,s:NormShape,x:&C::Value,w:&[f32])->Result<C::Value> {(**self).norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value)->Result<(C::Pass,C::Value)> {(**self).begin(key,p,x)}
    fn begin_next(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value,next:&[&str])->Result<(C::Pass,C::Value)> {(**self).begin_next(key,p,x,next)}
    fn finish(&self,p:C::Pass,x:&C::Value)->Result<C::Value> {(**self).finish(p,x)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&C::Value)->Result<C::Value> {(**self).mx(p,x)}
    fn round(&self,x:&C::Value)->Result<C::Value> {(**self).round(x)}
    fn mean(&self,x:&C::Value,s:u32,h:u32)->Result<C::Value> {(**self).mean(x,s,h)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&C::Value>;5],w:&[f32])->Result<C::Value> {(**self).prompt(s,x,w)}
    fn rows_embedding(&self,t:&[u32],n:usize)->Result<Option<C::Value>> {(**self).rows_embedding(t,n)}
    fn rows_project(&self,c:&CheckpointAdmission,name:&str,x:&C::Value,m:usize,round:bool)->Result<Option<C::Value>> {(**self).rows_project(c,name,x,m,round)}
    fn rows_stage(&self,c:&CheckpointAdmission,jobs:&[(&str,&C::Value,bool)],m:usize)->Result<Option<Vec<C::Value>>> {(**self).rows_stage(c,jobs,m)}
    fn batch_rows(&self,chats:usize,state_bytes:usize)->Result<()> {(**self).batch_rows(chats,state_bytes)}
    fn rows_experts(&self,a:&[(&str,usize)],b:Option<(&[(&str,usize)],f32)>,x:&C::Value,p:&[f32])->Result<Option<C::Value>> {(**self).rows_experts(a,b,x,p)}
    fn rows_routed(&self,steps:&[engine_model::plan::layer::Step],bound:Option<f32>,ids:&C::Value,x:&C::Value,m:usize)->Result<Option<C::Value>> {(**self).rows_routed(steps,bound,ids,x,m)}
    fn routes_rows(&self,m:usize)->bool {(**self).routes_rows(m)}
    fn note_batch_experts(&self,picks:usize,distinct:usize) {(**self).note_batch_experts(picks,distinct)}
    fn rows_release(&self)->Result<()> {(**self).rows_release()}
    fn rows_keep(&self)->Result<()> {(**self).rows_keep()}
    fn rows_unrecorded(&self,p:bool)->Result<bool> {(**self).rows_unrecorded(p)}
    fn prompt_end(&self)->Result<()> {(**self).prompt_end()}
    fn prompt_seeds(&self)->bool {(**self).prompt_seeds()}
    fn rows_charge(&self,bytes:usize) {(**self).rows_charge(bytes)}
    fn decode_selection_reads(&self,reads:bool) {(**self).decode_selection_reads(reads)}
    fn rows_peak_begin(&self)->usize {(**self).rows_peak_begin()}
    fn rows_peak_end(&self,whole:usize)->usize {(**self).rows_peak_end(whole)}
    fn rows_reserve(&self,words:usize)->Result<C::Value> {(**self).rows_reserve(words)}
    fn rows_fill(&self,target:&mut C::Value,offset:usize,part:&C::Value,expected:usize)->Result<()> {(**self).rows_fill(target,offset,part,expected)}
    fn rows_append(&self,held:Option<&C::Value>,rows:usize,new:&C::Value,width:usize)->Result<C::Value> {(**self).rows_append(held,rows,new,width)}
    fn rows_extend(&self,held:Option<&C::Value>,rows:usize,new:&C::Value,width:usize,room:usize)->Result<C::Value> {(**self).rows_extend(held,rows,new,width,room)}
    fn rows_norm(&self,name:&str,s:NormShape,x:&C::Value,m:usize,w:&[f32])->Result<C::Value> {(**self).rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value,m:usize)->Result<(C::Pass,C::Value)> {(**self).rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:C::Pass,x:&C::Value,m:usize)->Result<C::Value> {(**self).rows_finish(p,x,m)}
}
/// Charges for phase-separated prompt groups. The caller must clear its decode
/// memo before prompt work, drain row scratch before restoring it, and charge
/// resident bytes including pinned staging. Default service scheduling is unchanged.
/// Not for a service that keeps the device's memo between or beside sessions: see `service_memory::workspace`.
pub fn prompt_group_service_charges(checkpoint:&CheckpointAdmission,context:usize,resident_ready:bool,group:usize)->Result<(usize,usize)> {
    let reservation=crate::service_memory::session(checkpoint,context,context)?;
    let staging=if resident_ready {0} else {checkpoint.headers().weights().values().filter(|w|w.logical().shape.len()==2).map(|w|w.logical().shape.iter().try_fold(8usize,|n,&v|n.checked_mul(v as usize)).ok_or("weight staging overflow")).collect::<std::result::Result<Vec<_>,_>>()?.into_iter().max().unwrap_or(0)};
    let workspace=crate::service_memory::workspace(checkpoint,context,staging,group)?;
    Ok((reservation,workspace))
}
pub fn prompt_group_sparse_charges(checkpoint:&CheckpointAdmission,topology:Topology,context:usize,resident_ready:bool,group:usize)->Result<(usize,usize)> {
    let workspace=prompt_group_service_charges(checkpoint,context,resident_ready,group)?.1;
    Ok((crate::service_memory::session_for_sparse_heads(checkpoint,topology,context,context)?,workspace))
}
/// Reserve a group against the shared owner of ALL live session allowances.
/// Reusing a resident snapshot cannot create a second unfunded reservation.
pub fn admitted_prompt_group(admission:&mut crate::group_admission::PromptGroupAdmission<'_>,session:u64,prompt:usize,tokens:usize)->Result<usize> {admission.reserve(session,prompt,tokens)}

/// The server needs only each group's last row through the head.
/// What admission charges at this context: each full session, and the shared workspace.
pub fn service_charges(checkpoint:&CheckpointAdmission,context:usize,resident_ready:bool)->Result<(usize,usize)> {
    service_charges_owned(checkpoint,None,context,resident_ready)
}
pub fn sparse_service_charges(checkpoint:&CheckpointAdmission,topology:Topology,context:usize,resident_ready:bool)->Result<(usize,usize)> {
    service_charges_owned(checkpoint,Some(topology),context,resident_ready)
}
fn service_charges_owned(checkpoint:&CheckpointAdmission,topology:Option<Topology>,context:usize,resident_ready:bool)->Result<(usize,usize)> {
    let state=engine_model::plan::state::StatePlan::new(checkpoint.description(),Topology {rank:0,parts:1},context as u64,isize::MAX as usize).map_err(|e|e.to_string())?;
        let largest=checkpoint.headers().weights().values().filter(|w|w.logical().shape.len()==2).map(|w|w.logical().shape.iter().try_fold(4usize,|n,&v|n.checked_mul(v as usize)).ok_or_else(||"service weight size overflow".to_string())).collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
        // Execution scratch is shared. Expanded prompt rows are retained by each
        // request, and both its committed and unpublished state must fit.
        let expanded=match topology {Some(topology)=>crate::admission_plan::SessionChargePlan::for_sparse_heads(checkpoint.description(),topology,context)?.expanded_bytes(context),None=>crate::bounded_decoder::expanded_bytes(checkpoint,context)};
        let expanded=expanded.checked_mul(2).ok_or("expanded state size overflow")?;
        let cache=crate::admission_plan::append_reservation_bytes(checkpoint.description())?;
        let reservation=state.state_layout().bytes().checked_mul(2).and_then(|v|v.checked_add(expanded)).and_then(|v|v.checked_add(cache))
            .and_then(|v|v.checked_add(checkpoint.description().vocab as usize*16)).ok_or("service state reservation overflow")?;
        let extra=crate::bounded_decoder::group_workspace_bytes(checkpoint,context)?;
        // A resident owner has already funded every projection weight. Retain
        // separate scalar and prompt scratch allowances for cached buffers.
        let staging=if resident_ready {0} else {largest.checked_mul(2).ok_or("weight staging size overflow")?};
        let workspace=state.scratch_bytes().checked_mul(2).and_then(|v|v.checked_add(extra)).and_then(|v|v.checked_add(staging))
            .and_then(|v|v.checked_add(crate::bounded_decoder::PROJECTION_MEMO_BYTES+(1<<30))).ok_or("shared workspace size overflow")?;
    let session=match topology {Some(topology)=>crate::service_memory::session_for_sparse_heads(checkpoint,topology,context,context)?,None=>crate::service_memory::session(checkpoint,context,context)?};
    let reservation=reservation.max(session);
    Ok((reservation,workspace))
}
/// Independent request states advance in the controller scheduler order. The checkpoint and projection owners outlive all sessions.
/// `tails`: for a service with the predictor loaded, each chat's last prompt groups (tokens and
/// hidden rows, the fewest groups that hold PROMPT_TRAIL rows) until its first verify pass takes them.
pub struct Factory<'f,C:Compute,E> {checkpoint:&'f CheckpointAdmission,split:Split<C,E>,context:usize,eos:Vec<u32>,prepared:crate::bounded_decoder::Prepared<'f>,reservation:usize,workspace:usize,resident:usize,
    /// Rows of the prompt group a chat reads in while no other chat waits. `negotiate_prompt_group` sets it.
    group:usize,active:Cell<usize>,next_id:Cell<u64>,failed:Cell<bool>,failure:RefCell<Option<String>>,predictor:usize,charge_plan:crate::admission_plan::SessionChargePlan,with_layer:Option<crate::admission_plan::StateChargePlan>,tails:RefCell<BTreeMap<u64,Vec<(Vec<u32>,crate::bounded_decoder::OpenRows<C::Value>)>>>,
    /// Print only: when the last served batch step ended (the SERVED BATCH line gives the time since then).
    batch_ended:Cell<Option<std::time::Instant>>,
    /// Whether sessions of this service can be forked (`with_prompt_keep`). Off: no prompt is kept, as before.
    keep:bool,
    /// The prompts this rank keeps now, and the state bytes admission charges for them (`SessionChargePlan::state_bytes`).
    kept:Cell<usize>,kept_bytes:Cell<usize>}
/// The worker rank's live sessions by number: decoder, committed state, open verify rows.
type WorkerSessions<'f,'s,C,E>=BTreeMap<u64,(Decoder<'f,&'s Split<C,E>>,Option<State<<C as Compute>::Value>>,Option<Verify<<C as Compute>::Value>>)>;
impl<'f,C:Compute,E:Exchange> Factory<'f,C,E> {
    pub fn controller(&self)->bool {self.split.topology.rank.checked_sub(1).is_none()}
    pub fn new(checkpoint:&'f CheckpointAdmission,split:Split<C,E>,context:usize,eos:Vec<u32>)->Result<Self> {
        // Session setup reuses these host copies of immutable parameters.
        let prepared=Decoder::new(checkpoint,&split,context as u64)?.prepared();
        let topology=split.sparse_topology();
        let (reservation,workspace)=sparse_service_charges(checkpoint,topology,context,split.resident_weights_ready())?;
        if eos.is_empty() || eos.iter().any(|&id|u64::from(id)>=checkpoint.description().vocab) {return Err("service EOS set is invalid".into());}
        let resident=split.memory_bytes().map_or(0,|m|m.0);
        Ok(Self {checkpoint,split,context,eos,prepared,reservation,workspace,resident,group:crate::bounded_decoder::PROMPT_GROUP,active:Cell::new(0),next_id:Cell::new(1),failed:Cell::new(false),batch_ended:Cell::new(None),failure:RefCell::new(None),predictor:0,charge_plan:crate::admission_plan::SessionChargePlan::for_sparse_heads(checkpoint.description(),topology,context)?,with_layer:None,tails:RefCell::new(BTreeMap::new()),keep:false,kept:Cell::new(0),kept_bytes:Cell::new(0)})
    }
    /// Let the server's prompt cache fork this service's sessions (`Session::fork`). Every rank of the service
    /// makes the same call. Without it a session cannot be forked and no prompt is kept.
    pub fn with_prompt_keep(mut self)->Self {self.keep=true;self}
    /// The prompts this rank keeps now and their charged state bytes.
    pub fn kept_prompts(&self)->(usize,usize) {(self.kept.get(),self.kept_bytes.get())}
    fn kept_charge(&self,position:u64)->usize {usize::try_from(position).ok().and_then(|tokens|self.charge_plan.state_bytes(tokens.max(1)).ok()).unwrap_or(usize::MAX)}
    fn keep_count(&self,position:u64) {self.kept.set(self.kept.get()+1);self.kept_bytes.set(self.kept_bytes.get().saturating_add(self.kept_charge(position)));}
    fn release_count(&self,position:u64) {self.kept.set(self.kept.get().saturating_sub(1));self.kept_bytes.set(self.kept_bytes.get().saturating_sub(self.kept_charge(position)));}
    /// Load the checkpoint's predictor layer on this rank and let requests ask for up to `depth`
    /// proposals a pass (mtp_depth). Every rank of the service makes the same call.
    pub fn with_predictor(mut self,depth:usize)->Result<Self> {
        if !(1..=3).contains(&depth) {return Err("predictor depth is outside 1..=3".into());}
        let prepared={let mut decoder=Decoder::from_prepared(&self.prepared,&self.split,self.context as u64,self.active.get())?;decoder.enable_predictor()?;decoder.prepared()};
        // A chat that proposes keeps the layer's own cache: charge the state of 46 layers, not 45.
        let mut with_layer=self.checkpoint.description().with_draft_layer().map_err(|e|e.to_string())?.ok_or("checkpoint declares no predictor layer")?;
        with_layer.residual=self.checkpoint.description().residual;self.with_layer=Some(crate::admission_plan::StateChargePlan::new(&with_layer,self.context)?);
        self.reservation=self.prompt_state_reservation(self.context,self.context)?;
        self.prepared=prepared;self.predictor=depth;self.resident=self.split.memory_bytes().map_or(self.resident,|m|m.0);Ok(self)
    }
    pub fn predictor_depth(&self)->usize {self.predictor}
    /// Only one synchronous execution runs at a time. Charge its scratch once;
    /// each live request separately funds committed and unpublished state.
pub fn workspace_reservation_bytes(&self)->usize {self.workspace}
    /// One prompt group on this rank, with its peak device bytes against what admission
    /// charged for the sessions now live. Printed for a refused group too.
    fn prompt_group(&self,decoder:&Decoder<'_,&Split<C,E>>,tokens:&[u32],old:&State<C::Value>,live:usize,id:u64,last:bool)->Result<(Vec<Vec<f32>>,State<C::Value>)> {
        // Kept prompts hold state too. With none kept this adds zero.
        let charge=self.reservation.saturating_mul(live).saturating_add(self.resident).saturating_add(self.workspace).saturating_add(self.kept_bytes.get());
        // The group's spare buffers stay under what admission charged for the sessions now live (`group_pool_cap`).
        self.split.rows_charge(charge);
        let whole=self.split.rows_peak_begin();decoder.set_memo_seed(last && self.split.prompt_seeds());
        // With the predictor loaded the group also keeps the hidden states of its last PROMPT_TRAIL rows (the predictor's
        // trail reads no earlier row), in storage of their own: 8 MB at most, whatever the group size.
        let result=if self.predictor>0 {decoder.prefill_rows_hidden(tokens,old,crate::bounded_decoder::HeadRows::Last).and_then(|(logits,next,open)|{
                let open=decoder.open_tail(open,crate::bounded_decoder::PROMPT_TRAIL)?;let kept=tokens.len().min(crate::bounded_decoder::PROMPT_TRAIL);
                let mut tails=self.tails.borrow_mut();let groups=tails.entry(id).or_default();groups.push((tokens[tokens.len()-kept..].to_vec(),open));
                while groups.len()>1 && groups[1..].iter().map(|group|group.0.len()).sum::<usize>()>=crate::bounded_decoder::PROMPT_TRAIL {groups.remove(0);}Ok((logits,next))})}
            else {decoder.prefill_rows(tokens,old,crate::bounded_decoder::HeadRows::Last)};let peak=self.split.rows_peak_end(whole);
        eprintln!("PROMPT GROUP PEAK rank={} position={} rows={} live={live} peak_bytes={peak} charge_bytes={charge} resident_bytes={} workspace_bytes={} full_session_bytes={} ok={}",self.split.topology.rank,old.position,tokens.len(),self.resident,self.workspace,self.reservation,result.is_ok());
        decoder.set_memo_seed(false);
        result
    }
    fn remember(&self,error:String)->String {let mut failure=self.failure.borrow_mut();let first=failure.is_none();let saved=failure.get_or_insert(error).clone();drop(failure);self.failed.set(true);if first {self.split.exchange.borrow_mut().abort();}saved}
    fn failure_reason(&self)->Option<String> {let saved=self.failure.borrow().clone();saved.or_else(||self.split.fatal_error()).or_else(||self.failed.get().then(||"row split service previously failed".into())).map(|error|self.remember(error))}
    fn finish<T>(&self,result:Result<T>)->Result<T> {result.map_err(|error|self.remember(error))}
    pub fn full_state_reservation_bytes(&self)->usize {self.reservation}
    pub fn cache_append_reservation_bytes(&self)->usize {self.with_layer.as_ref().map_or_else(||self.charge_plan.cache_append_reservation_bytes(),|plan|plan.cache_append_reservation_bytes())}
    pub fn index_append_reservation_bytes(&self)->usize {self.with_layer.as_ref().map_or_else(||self.charge_plan.index_append_reservation_bytes(),|plan|plan.index_append_reservation_bytes())}
    /// Choose the prompt group: the largest of 1024, 512 and 256 rows whose workspace fits
    /// `budget` beside this rank's resident bytes and one full session. Each rank offers its
    /// largest and all take the smallest, so the ranks agree. A group reads each expert it
    /// routes to once (nearly all of them from about 100 rows up), so a larger group pays that
    /// read, and every exchange and read-back of a layer, for more rows. The workspace grows by
    /// the larger group's buffers; past 256 rows a recurrent state copy can wait behind its
    /// source, so those copies are charged too. Row bits do not depend on the group.
    /// Every rank calls this after the predictor is loaded and before the session pool is negotiated.
    pub fn negotiate_prompt_group(mut self,budget:usize)->Result<Self> {
        use crate::bounded_decoder::{PROMPT_GROUP,PROMPT_GROUP_MAX};
        let resident=self.split.memory_bytes().map_or(self.resident,|m|m.0);
        let offer=crate::service_memory::largest_prompt_group(self.checkpoint,self.context,self.workspace,resident,self.reservation,budget)?;
        let group=self.negotiate_available_bytes(offer)?;
        if ![PROMPT_GROUP,PROMPT_GROUP_MAX/2,PROMPT_GROUP_MAX].contains(&group) || group>offer {return Err("ranks disagree on the prompt group".into());}
        self.workspace=crate::service_memory::workspace_for_prompt_group(self.checkpoint,self.context,self.workspace,group)?;self.group=group;Ok(self)
    }
    /// Rows of the prompt group a lone chat reads in.
    pub fn prompt_group_rows(&self)->usize {self.group}
    pub fn negotiate_available_bytes(&self,bytes:usize)->Result<usize> {
        let local=(0..4).map(|i|((bytes as u64>>(i*16))&65535) as f32).collect::<Vec<_>>();
        let count=self.split.topology.parts as usize*local.len();let values=self.split.exchange.borrow_mut().gather(&local,count)?;
        if values.len()!=count {return Err("peer memory bounds have an invalid count".into());}
        let mut minimum=bytes;for words in values.as_chunks::<4>().0 {let mut value=0u64;
            for (i,&v) in words.iter().enumerate() {if !v.is_finite() || !(0.0..=65535.0).contains(&v) || v.fract()!=0.0 {return Err("invalid peer memory bound".into());}value|=(v as u64)<<(i*16);}
            minimum=minimum.min(value.try_into().map_err(|_|"peer memory bound exceeds address space")?);
        }Ok(minimum)
    }
    fn state_reservation(&self,tokens:usize)->Result<usize> {
        self.prompt_state_reservation(tokens,tokens)
    }
    fn prompt_state_reservation(&self,prompt:usize,tokens:usize)->Result<usize> {
        if tokens==0 || tokens>self.context || prompt>tokens {return Err("state reservation exceeds context".into());}
        let plain=self.charge_plan.request(prompt,tokens)?;
        let Some(with_layer)=self.with_layer.as_ref() else {return Ok(plain);};
        // With the predictor, on top of the plain charge: the predictor layer's own state (twice, as
        // the state); the held rows follow the decoded rows, so every token of the request and the
        // room behind them are charged, not the prompt alone (twice); and the hidden rows of the
        // last prompt groups wait for the first proposal (two planes of the largest group).
        let own=with_layer.bytes(tokens)?.saturating_sub(self.charge_plan.state_bytes(tokens)?);
        let rows=self.charge_plan.expanded_bytes(tokens+crate::bounded_decoder::HELD_ROOM).saturating_sub(self.charge_plan.expanded_bytes(prompt));
        let waiting=crate::bounded_decoder::PROMPT_GROUP_MAX*self.checkpoint.description().hidden as usize*8;
        let cache=with_layer.append_reservation_bytes().saturating_sub(self.charge_plan.append_reservation_bytes());
        own.checked_mul(2).and_then(|v|v.checked_add(rows.checked_mul(2)?)).and_then(|v|v.checked_add(waiting)).and_then(|v|v.checked_add(plain)).and_then(|v|v.checked_add(cache)).ok_or_else(||"request state reservation overflow".into())
    }
    /// Print only: the graph captures of this rank since the load, when a session ends. Two sessions' lines give the
    /// captures of the one between them.
    fn capture_report(&self,session:u64) {
        let [new,seen,evicted,updated]=self.split.capture_counts();
        eprintln!("SERVED CAPTURES rank={} session={session} captures_new={new} captures_seen={seen} piece_evictions={evicted} piece_updates={updated}",self.split.topology.rank);
    }
    pub fn shutdown(&self)->Result<()> {
        if let Some(error)=self.failure_reason() {return Err(error);}
        if self.active.get()!=0 {return Err("service shutdown has a live session".into());}
        // The other rank still holds every kept prompt and refuses to stop with one (command 0). So refuse here,
        // before the command: the caller drops its kept sessions first, as it ends its live ones first.
        if self.kept.get()!=0 {return Err("service shutdown has a kept prompt".into());}self.split.command(0,0)
    }
    pub fn worker(&self)->Result<()> {
        let mut sessions=BTreeMap::new();let mut kept=BTreeMap::<u64,State<C::Value>>::new();
        loop {let mut words=[0;16];self.split.exchange.borrow_mut().command(&mut words)?;
            let code=u32::from_le_bytes(words[..4].try_into().unwrap());let token=u32::from_le_bytes(words[4..8].try_into().unwrap());let id=u64::from_le_bytes(words[8..].try_into().unwrap());
            match code {
                0=>{if !sessions.is_empty() || !kept.is_empty() {return Err("worker stopped with live sessions or kept prompts".into());}return Ok(());},
                1=>{if id==0 || sessions.contains_key(&id) || sessions.len()>=8 {return Err("worker session identity or capacity differs".into());}
                    let decoder=Decoder::from_prepared(&self.prepared,&self.split,self.context as u64,sessions.len())?;let state=decoder.zero_state()?;sessions.insert(id,(decoder,Some(state),None::<Verify<C::Value>>));self.split.exchange.borrow_mut().complete()?;},
                2|4|5|12=>{let live=sessions.len();self.split.work(|| {let (decoder,state,_)=sessions.get_mut(&id).ok_or("worker command has no session")?;let state=state.as_mut().ok_or("worker session has open verify rows")?;
                    let last=code==5 && token&LAST_GROUP!=0;
                    let next=if code==5 {let rows=(token&(LAST_GROUP-1)) as usize;if rows==0 || rows>self.group {return Err("worker prompt group exceeds its bound".into());}
                        let mut bytes=vec![0;rows*4];self.split.exchange.borrow_mut().command(&mut bytes)?;
                        let tokens=bytes.as_chunks::<4>().0.iter().map(|b|u32::from_le_bytes(*b)).collect::<Vec<_>>();self.prompt_group(decoder,&tokens,state,live,id,last)?.1
                    } else if code==12 {decoder.step_greedy(token,state,|_,_|Ok(()))?.1} else {decoder.step(token,state,|_,_|Ok(()))?.2};
                    self.split.exchange.borrow_mut().complete()?;*state=next;
                    // The prompt's last group is done and the state before it is gone: decode steps follow. The pool is trimmed
                    // now, on this rank as on the controller, off the first decode step.
                    if last {self.split.prompt_end()?;}Ok(())
                })?;},
                3=>{self.tails.borrow_mut().remove(&id);if sessions.remove(&id).is_none() {return Err("worker drop has no session".into());}self.capture_report(id);self.split.exchange.borrow_mut().complete()?;},
                // Command 13: one batch step. `token` is the row count; each row follows as a session ID and its input token.
                13=>self.worker_batch(&mut sessions,token as usize)?,
                // The same verify pass and the same kept prefix as the controller, so both ranks hold one state.
                // Commands 9 to 11: a proposal chain of depth 1 to 3 after `token`. Both ranks run it,
                // because every projection of the predictor layer is split by rows.
                9..=11=>{self.split.work(|| {let (decoder,_,open)=sessions.get_mut(&id).ok_or("worker command has no session")?;
                    open.as_mut().ok_or("worker proposal has no open rows")?.propose(decoder,token,(code-8) as usize)?;self.split.exchange.borrow_mut().complete()
                })?;},
                // Command 8 is a pass of one row: its token rides in the command. Command 14 is the same
                // pass as the plain greedy step (`Verify::row_greedy`): the controller sends it only with no row carried.
                6|8|14=>{self.split.work(|| {let (decoder,state,open)=sessions.get_mut(&id).ok_or("worker command has no session")?;
                    let tokens=if code!=6 {vec![token]} else {let rows=token as usize;if !(2..=crate::bounded_decoder::VERIFY_ROWS).contains(&rows) {return Err("worker verify pass exceeds its bound".into());}
                        let mut bytes=vec![0;rows*4];self.split.exchange.borrow_mut().command(&mut bytes)?;
                        bytes.as_chunks::<4>().0.iter().map(|b|u32::from_le_bytes(*b)).collect::<Vec<_>>()};
                    if open.is_none() {let mut fresh=Verify::new(state.take().ok_or("worker session has no state")?);if self.predictor>0 {fresh.track_proposer();if let Some(groups)=self.tails.borrow_mut().remove(&id) {fresh.seed_trail(decoder,&groups)?;}}*open=Some(fresh);}
                    if code==14 {open.as_mut().unwrap().row_greedy(decoder,token)?.ok_or("worker greedy pass has carried rows")?;} else {open.as_mut().unwrap().rows(decoder,&tokens)?;}
                    self.split.exchange.borrow_mut().complete()
                })?;},
                7=>{self.split.work(|| {let (decoder,_,open)=sessions.get_mut(&id).ok_or("worker command has no session")?;open.as_mut().ok_or("worker truncation has no open rows")?.truncate(decoder,u64::from(token))?;self.split.exchange.borrow_mut().complete()})?;},
                // Commands 15 to 17: a kept prompt (`kept_command`). Out of this loop, so the step arms keep their code.
                15..=17=>self.kept_command(code,id,&mut sessions,&mut kept)?,
                _=>return Err("unknown row split service command".into()),
            }
        }
    }
    /// Commands 15 to 17 of the worker: a kept prompt (`Session::fork`). The number of the new owner follows 15 and 16
    /// as eight bytes. 15 keeps a second handle of session `id`'s committed state. 16 opens a session at kept state
    /// `id`. 17 forgets kept state `id`. No word of state is copied, and both ranks hold the same numbers.
    /// Cold and never inlined: a request sends one of these at most, and the worker's loop stays the code it was.
    #[cold] #[inline(never)]
    fn kept_command<'s>(&'s self,code:u32,id:u64,sessions:&mut WorkerSessions<'f,'s,C,E>,kept:&mut BTreeMap<u64,State<C::Value>>)->Result<()> {
        if code==17 {let state=kept.remove(&id).ok_or("worker release has no kept prompt")?;self.release_count(state.position);drop(state);return self.split.exchange.borrow_mut().complete();}
        let mut word=[0;8];self.split.exchange.borrow_mut().command(&mut word)?;let new=u64::from_le_bytes(word);
        if code==15 {
            if new==0 || kept.contains_key(&new) || sessions.contains_key(&new) || kept.len()>=KEPT_PROMPTS {return Err("worker kept prompt identity or capacity differs".into());}
            let state=sessions.get(&id).and_then(|session|session.1.as_ref()).ok_or("worker keep has no committed state")?;
            let shared=crate::bounded_decoder::share_state(&self.split,state)?;self.keep_count(shared.position);kept.insert(new,shared);return self.split.exchange.borrow_mut().complete();}
        if new==0 || sessions.contains_key(&new) || kept.contains_key(&new) || sessions.len()>=8 {return Err("worker session identity or capacity differs".into());}
        let decoder=Decoder::from_prepared(&self.prepared,&self.split,self.context as u64,sessions.len())?;
        let state=crate::bounded_decoder::share_state(&self.split,kept.get(&id).ok_or("worker resume has no kept prompt")?)?;
        sessions.insert(new,(decoder,Some(state),None::<Verify<C::Value>>));self.split.exchange.borrow_mut().complete()
    }
}

mod gate;
mod price;
use gate::Gate;pub use gate::Account;
/// `open` holds the state once a request decodes by verify passes. `state` is then empty.
pub struct Session<'a,C:Compute,E:Exchange> {factory:&'a Factory<'a,C,E>,decoder:Decoder<'a,&'a Split<C,E>>,state:Option<State<C::Value>>,id:u64,open:Option<Verify<C::Value>>,strict:bool,depth:usize,
    /// This request's own timed ms: a plain step that follows a plain step, and a pass of each
    /// row count it has run. The chooser gets no other cost: nothing here is measured elsewhere.
    step_ms:Option<f64>,pass_ms:[Option<f64>;17],after_step:bool,gate:Gate,
    /// The depth the Speculator's chooser set for the next proposal.
    limit:usize,
    /// For a session that only holds a kept prompt (`fork`): the state at the prompt's end. Such a
    /// session has no `state` and no `open`, so no step or prompt group can run in it.
    kept:Option<State<C::Value>>}
impl<C:Compute,E:Exchange> spark_serve::Engine for Factory<'_,C,E> {
    type Session<'a>=Session<'a,C,E> where Self:'a;
    fn profile_serving(&self)->bool {self.split.serving_trace()}
    fn profile_tick_begin(&self,tick:u64) {self.split.profile_tick_begin(tick)}
    fn profile_tick_end(&self) {self.split.profile_tick_end()}
    fn profile_stream_output(&self,request:u64,tokens:usize,ns:u128) {eprintln!("SERVED OUTPUT request={request} completion_tokens={tokens} wall_ns={ns}");}
    #[allow(clippy::possible_missing_else)] // dense one-line guards.
    fn session(&self)->Result<Self::Session<'_>> {
        if let Some(error)=self.failure_reason() {return Err(error);}if self.active.get()>=8 {return Err("row split service has no free session".into());}
        let id=self.next_id.get();let next=id.checked_add(1).ok_or("session identity exhausted")?;self.next_id.set(next);self.active.set(self.active.get()+1);
        let result=(|| {let started=std::time::Instant::now();self.split.command_for(1,0,id)?;
            // The other sessions that are open now: the worker counts the same ones.
            let decoder=Decoder::from_prepared(&self.prepared,&self.split,self.context as u64,self.active.get()-1)?;
            let state=decoder.zero_state()?;self.split.exchange.borrow_mut().complete()?;eprintln!("SERVED SESSION rank={} session={id} active={} setup_seconds={:.9}",self.split.topology.rank,self.active.get(),started.elapsed().as_secs_f64());self.split.cost_report("setup");self.split.trace_start();Ok(Session {factory:self,decoder,state:Some(state),id,open:None,strict:false,depth:0,step_ms:None,pass_ms:[None;17],after_step:false,gate:Gate::default(),limit:0,kept:None})})();
        if result.is_err() {self.active.set(self.active.get()-1);}self.finish(result)
    }
    fn fatal_error(&self)->Option<String> {self.failure_reason()}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {self.split.memory_bytes()}
    fn reserve_full_output(&self)->bool {true}
    /// A kept prompt also holds its expanded rows (the first `RESERVED_ROWS` rows at most) when no live chat shares
    /// them, and `retained_bytes` does not count them. The cache leaves this many bytes of the pool free for them.
    /// Zero while nothing is kept.
    fn cache_hold_bytes(&self)->usize {self.kept.get().saturating_mul(self.charge_plan.expanded_bytes(self.context))}
    fn context_limit(&self)->usize {self.context}
    fn vocab_size(&self)->usize {self.checkpoint.description().vocab as usize}
    fn eos_token_ids(&self)->&[u32] {&self.eos}
    fn session_reservation_bytes(&self)->Option<usize> {self.state_reservation(1).ok()}
    fn request_reservation_bytes(&self,tokens:usize)->Option<usize> {self.state_reservation(tokens).ok()}
    fn request_reservation_bytes_for_prompt(&self,prompt:usize,tokens:usize,_plain:bool,_cached:Option<usize>,_retain:bool)->Option<usize> {self.prompt_state_reservation(prompt,tokens).ok()}
    fn reserve_foreground_slot(&self)->bool {false}
    fn shared_prompt_limit(&self)->usize {0}
    fn shared_prefill_batch(&self)->bool {true}
    fn shared_prefill_limit(&self)->usize {crate::bounded_decoder::PROMPT_GROUP}
    fn shared_prefill_max_tokens(&self)->usize {crate::bounded_decoder::PROMPT_GROUP}
    // step_batch runs these independent prompts sequentially. Each pass fits the
    // funded group; the row sum bounds scheduler work, not simultaneous scratch.
    fn shared_prefill_rows(&self)->usize {crate::bounded_decoder::BATCH_ROWS*crate::bounded_decoder::PROMPT_GROUP}
    /// Proposed tokens are verified for one chat. More chats decode plain rows.
    fn single_chat_draft(&self)->bool {true}
    fn lone_chat_rows(&self)->usize {crate::bounded_decoder::VERIFY_ROWS}
    /// One decode row a tick for each plain chat; `step_batch` joins them in one batch step when enough ride.
    fn shared_decode_rows(&self)->usize {crate::bounded_decoder::BATCH_ROWS}
    /// Fewer chats in decode than a batch step takes run their own plain steps, one a tick as on an engine with no batch step.
    /// Two chats alone on the server (two live sessions) ride one batch step; with any other count the floor is BATCH_MIN_ROWS.
    fn shared_decode_min(&self)->usize {if self.active.get()==crate::bounded_decoder::BATCH_PAIR_ROWS {crate::bounded_decoder::BATCH_PAIR_ROWS} else {crate::bounded_decoder::BATCH_MIN_ROWS}}
    /// A request that drafts sends its rows as one verify pass.
    fn step_batch<'s>(&'s self,rows:&mut [spark_serve::BatchRows<'_,Self::Session<'s>>])->std::result::Result<Vec<spark_serve::Predictions>,spark_serve::BatchError> {
        use spark_serve::{BatchOutput,Session as _};
        // Chats in plain decode ride one batch step when there are enough of them to gain (`shared_decode_min`).
        // Each gets the ID (greedy) or the logits of its own plain step.
        let plain=rows.iter().map(|row|matches!(row.output,BatchOutput::All) && row.tokens.len()==1 && !row.session.strict && row.session.open.is_none()).collect::<Vec<_>>();
        let mut shared=Vec::new();
        if plain.iter().filter(|&&ride|ride).count()>=spark_serve::Engine::shared_decode_min(self) {
            let tokens=rows.iter().zip(&plain).filter(|(_,ride)|**ride).map(|(row,_)|row.tokens[0]).collect::<Vec<_>>();
            // Every riding chat greedy: none needs its logits, so each ID is picked where the logits are.
            let greedy=rows.iter().zip(&plain).filter(|(_,ride)|**ride).all(|(row,_)|row.greedy);
            let mut sessions=rows.iter_mut().zip(&plain).filter(|(_,ride)|**ride).map(|(row,_)|&mut *row.session).collect::<Vec<_>>();
            shared=self.batch_tokens(&mut sessions,&tokens,greedy).map_err(|message|spark_serve::BatchError {request:None,message})?;
        }
        let mut shared=shared.into_iter();
        rows.iter_mut().zip(&plain).map(|(row,&ride)| {
            if ride {
                if let Some((logits,best))=shared.next() {return Ok(if row.greedy {spark_serve::Predictions::Greedy(vec![best])} else {spark_serve::Predictions::Logits(vec![logits])});}
                if row.greedy {return row.session.token_prediction(row.tokens[0],false,true).map_err(|message|spark_serve::BatchError {request:None,message});}
            }
            // A strict request's pass: a greedy one of one row answers with the token alone (`Session::pass`).
            if matches!(row.output,BatchOutput::All) && row.session.strict {
                return row.session.pass(row.tokens,row.greedy).map_err(|message|spark_serve::BatchError {request:None,message});
            }
            let result=match row.output {
            BatchOutput::All=>row.tokens.iter().map(|&t|row.session.step(t)).collect(),
            BatchOutput::Last=>row.session.prefill(row.tokens).map(|v|vec![v]),
            BatchOutput::None=>row.session.prefill_advance(row.tokens).map(|_|Vec::new())};
            result.map(spark_serve::Predictions::Logits).map_err(|message|spark_serve::BatchError {request:None,message})}).collect()
    }
}
impl<C:Compute,E:Exchange> Session<'_,C,E> {
    #[allow(clippy::possible_missing_else)] // dense one-line guards.
    /// `last`: these tokens end the prompt, and decode steps follow.
    fn prompt(&mut self,tokens:&[u32],last:bool)->Result<Vec<f32>> {if let Some(error)=self.factory.failure_reason() {return Err(error);}
        // A session that only holds a kept prompt reads nothing. Refused here, before `finish`: the service stays whole.
        if self.kept.is_some() {return Err("row split session only holds a kept prompt".into());}
        if tokens.is_empty() {return Err("empty row split prompt".into());}let factory=self.factory;factory.finish(factory.split.work(||self.prompt_inner(tokens,last)))}
    fn prompt_inner(&mut self,tokens:&[u32],last:bool)->Result<Vec<f32>> {
        if tokens.is_empty() {return Err("empty row split prompt".into());}let mut logits=Vec::new();let groups=tokens.len().div_ceil(self.factory.group);
        for (group,tokens) in tokens.chunks(self.factory.group).enumerate() {let last=last && group+1==groups;
            if self.factory.failed.get() {return Err("row split service previously failed".into());}
            let old=self.state.as_ref().ok_or("row split prompt has no state")?;
            if old.position.checked_add(tokens.len() as u64).is_none_or(|n|n>self.factory.context as u64)
                || tokens.iter().any(|&token|u64::from(token)>=self.factory.checkpoint.description().vocab) {return Err("row split prompt exceeds vocabulary or context".into());}
            let started=std::time::Instant::now();let position=old.position;let pool=self.factory.split.vector_pool_counts();self.factory.failed.set(true);self.factory.split.command_for(5,tokens.len() as u32|if last {LAST_GROUP} else {0},self.id)?;
            let mut bytes=tokens.iter().flat_map(|t|t.to_le_bytes()).collect::<Vec<_>>();self.factory.split.exchange.borrow_mut().command(&mut bytes)?;
            let (mut rows,next)=self.factory.prompt_group(&self.decoder,tokens,old,self.factory.active.get(),self.id,last)?;self.factory.split.exchange.borrow_mut().complete()?;
            self.state=Some(next);self.factory.failed.set(false);logits=rows.pop().ok_or("prompt group has no final logits")?;
            eprintln!("SERVED GROUP rank={} session={} phase=prefill position={position} rows={} seconds={:.9} {}",self.factory.split.topology.rank,self.id,tokens.len(),started.elapsed().as_secs_f64(),crate::bounded_decoder::group_allocation_fields(pool,self.factory.split.vector_pool_counts()));self.factory.split.cost_report("prefill");
            // The prompt ends: one line of every prompt kernel's launch count in this process, so a served log
            // proves which paths ran (large-group runs, row stages, pool windows).
            if last {eprintln!("SERVED PROMPT COUNTS rank={} session={} {}",self.factory.split.topology.rank,self.id,spark_gpu_probe::prompt_launches::line());}
            if last {self.factory.split.prompt_end()?;}
        }Ok(logits)
    }
    fn token(&mut self,token:u32,prefill:bool)->Result<Vec<f32>> {
        match self.token_prediction(token,prefill,false)? {spark_serve::Predictions::Logits(mut rows)=>rows.pop().ok_or_else(||"row split step has no logits".into()),_=>Err("row split step output differs".into())}
    }
    #[allow(clippy::possible_missing_else)] // dense one-line guards.
    fn token_prediction(&mut self,token:u32,prefill:bool,greedy:bool)->Result<spark_serve::Predictions> {let factory=self.factory;if let Some(error)=factory.failure_reason() {return Err(error);}if self.state.as_ref().is_none_or(|s|s.position>=factory.context as u64) || u64::from(token)>=factory.checkpoint.description().vocab {return Err("row split request exceeds vocabulary or context".into());}factory.finish(factory.split.work(||self.token_inner(token,prefill,greedy)))}
    fn token_inner(&mut self,token:u32,prefill:bool,greedy:bool)->Result<spark_serve::Predictions> {
        let started=std::time::Instant::now();
        if self.factory.failed.get() {return Err("row split service previously failed".into());}
        let old=self.state.as_ref().ok_or("row split session has no state")?;
        if old.position>=self.factory.context as u64 || u64::from(token)>=self.factory.checkpoint.description().vocab {return Err("row split request exceeds vocabulary or context".into());}
        let position=old.position;let allocations=self.factory.split.vector_allocations();self.factory.failed.set(true);self.factory.split.command_for(if greedy {12} else if prefill {4} else {2},token,self.id)?;let sent=started.elapsed().as_secs_f64();
        let (output,next)=if greedy {let (best,next)=self.decoder.step_greedy(token,old,|_,_|Ok(()))?;(spark_serve::Predictions::Greedy(vec![best]),next)}
            else {let (logits,_,next)=self.decoder.step(token,old,|_,_|Ok(()))?;(spark_serve::Predictions::Logits(vec![logits]),next)};
        let stepped=started.elapsed().as_secs_f64();self.factory.split.exchange.borrow_mut().complete()?;
        self.state=Some(next);self.factory.failed.set(false);let phase=if prefill {"prefill"} else {"decode"};eprintln!("SERVED STEP rank={} session={} phase={phase} position={position} seconds={:.9} command_seconds={sent:.6} step_seconds={:.6} {}",self.factory.split.topology.rank,self.id,started.elapsed().as_secs_f64(),stepped-sent,crate::bounded_decoder::allocation_fields(allocations,self.factory.split.vector_allocations()));self.factory.split.cost_report(phase);Ok(output)
    }
}
impl<C:Compute,E:Exchange> Session<'_,C,E> {
    /// One verify pass on both ranks. The rows stay open until a later pass or `truncate`.
    fn rows(&mut self,tokens:&[u32])->Result<Vec<Vec<f32>>> {
        match self.pass(tokens,false)? {spark_serve::Predictions::Logits(rows)=>Ok(rows),_=>Err("row split verify pass output differs".into())}
    }
    /// The same pass. With `greedy`, a pass of one row with no row carried is the plain greedy step
    /// (command 14, `Verify::row_greedy`): its answer is the token the head picks, and no logits are read.
    fn pass(&mut self,tokens:&[u32],greedy:bool)->Result<spark_serve::Predictions> {let factory=self.factory;if let Some(error)=factory.failure_reason() {return Err(error);}
        // A session that only holds a kept prompt verifies nothing. Refused here, before `finish`: the service stays whole.
        if self.kept.is_some() {return Err("row split session only holds a kept prompt".into());}
        if tokens.is_empty() || tokens.len()>crate::bounded_decoder::VERIFY_ROWS || tokens.iter().any(|&token|u64::from(token)>=factory.checkpoint.description().vocab) {return Err("row split verify pass exceeds its rows or vocabulary".into());}
        factory.finish(factory.split.work(||self.rows_inner(tokens,greedy)))}
    fn rows_inner(&mut self,tokens:&[u32],greedy:bool)->Result<spark_serve::Predictions> {
        let started=std::time::Instant::now();let factory=self.factory;if factory.failed.get() {return Err("row split service previously failed".into());}
        if self.open.is_none() {let mut fresh=Verify::new(self.state.take().ok_or("row split session has no state")?);if factory.predictor>0 {fresh.track_proposer();if let Some(groups)=factory.tails.borrow_mut().remove(&self.id) {fresh.seed_trail(&self.decoder,&groups)?;}}self.open=Some(fresh);}
        let open=self.open.as_mut().unwrap();let (position,carried)=(open.position(),open.carried());
        if position.checked_add(tokens.len() as u64).is_none_or(|n|n>factory.context as u64) {return Err("row split verify pass exceeds context".into());}
        factory.failed.set(true);let picked=greedy && tokens.len()==1 && carried==0;
        if picked {factory.split.command_for(14,tokens[0],self.id)?;} else if tokens.len()==1 {factory.split.command_for(8,tokens[0],self.id)?;} else {factory.split.command_for(6,tokens.len() as u32,self.id)?;
            let mut bytes=tokens.iter().flat_map(|t|t.to_le_bytes()).collect::<Vec<_>>();factory.split.exchange.borrow_mut().command(&mut bytes)?;}
        let whole=factory.split.rows_peak_begin();
        let output=if picked {spark_serve::Predictions::Greedy(vec![open.row_greedy(&self.decoder,tokens[0])?.ok_or("row split greedy pass has carried rows")?])}
            else {spark_serve::Predictions::Logits(open.rows(&self.decoder,tokens)?)};
        let peak=factory.split.rows_peak_end(whole);
        factory.split.exchange.borrow_mut().complete()?;factory.failed.set(false);
        {let ms=started.elapsed().as_secs_f64()*1000.0;let mix=|old:Option<f64>,new:f64|Some(old.map_or(new,|old|0.75*old+0.25*new));
            self.gate.pass(tokens.len(),ms,picked);
            // A step is timed when the call before it was a step too: the first step after a
            // prompt or a pass is slower, and the chooser compares a pass with steady plain steps.
            let one=tokens.len()==1 && carried==0;
            if one {if self.after_step {self.step_ms=mix(self.step_ms,ms);}} else {let rows=(tokens.len()+carried).min(16);self.pass_ms[rows]=mix(self.pass_ms[rows],ms);}
            self.after_step=one;}
        eprintln!("SERVED VERIFY rank={} session={} phase=decode position={position} rows={} carried={carried} picked={} peak_bytes={peak} seconds={:.9}",factory.split.topology.rank,self.id,tokens.len(),u8::from(picked),started.elapsed().as_secs_f64());factory.split.cost_report("decode");Ok(output)
    }
    /// Proposals of the predictor layer for the tokens after `next`. Empty until this request
    /// has a hidden state of its own (the first call after the prompt).
    fn propose(&mut self,next:u32,limit:usize)->Result<(Vec<u32>,f64)> {let factory=self.factory;if let Some(error)=factory.failure_reason() {return Err(error);}
        let depth=limit.min(self.depth).min(factory.predictor);
        let Some(open)=self.open.as_mut().filter(|open|open.can_propose()) else {return Ok((Vec::new(),0.0));};
        if depth==0 || u64::from(next)>=factory.checkpoint.description().vocab {return Ok((Vec::new(),0.0));}
        // The layer runs now: the pass after it is not a steady step, whatever its row count.
        let (decoder,id)=(&self.decoder,self.id);self.after_step=false;
        factory.finish((|| -> Result<(Vec<u32>,f64)> {factory.failed.set(true);factory.split.command_for(8+depth as u32,next,id)?;
            let tokens=factory.split.work(||open.propose(decoder,next,depth))?;
            factory.split.exchange.borrow_mut().complete()?;factory.failed.set(false);Ok(tokens)})())
    }
    /// The proposer's account of this request so far.
    pub fn proposer_account(&self)->Account {self.gate.account()}
    /// What a pass of `rows` rows costs this request (`learned_pass_ms`). None until a steady step is timed.
    fn pass_cost(&self,rows:usize)->Option<f64> {Some(crate::bounded_decoder::learned_pass_ms(self.step_ms?,&self.pass_ms,rows))}
    fn measured_costs(&self)->Option<spark_engine::draft::CostCurve> {
        Some(spark_engine::draft::CostCurve([self.pass_cost(1)?,self.pass_cost(2)?,self.pass_cost(4)?,self.pass_cost(8)?,self.pass_cost(16)?]))
    }
    fn keep(&mut self,position:usize)->Result<()> {let factory=self.factory;if let Some(error)=factory.failure_reason() {return Err(error);}
        let now=self.open.as_ref().map(Verify::position).or(self.state.as_ref().map(|s|s.position)).ok_or("row split session has no state")?;
        // A verdict that keeps every row of a pass still goes to both ranks: it releases what the
        // pass kept. After a plain step nothing is kept, so nothing is sent.
        let id=self.id;let Some(open)=self.open.as_mut() else {return if position as u64==now {Ok(())} else {Err("row split truncation has no open rows".into())};};let decoder=&self.decoder;
        if position as u64==now && !open.has_kept() {return Ok(());}
        // A refused request never reaches the other rank, so the service stays whole.
        if position as u64>now || (position as u64)<open.committed() {return Err("row split truncation is outside the open rows".into());}
        let started=std::time::Instant::now();let kept_to=position as u64;let partial=kept_to<now;self.after_step=false;
        let result=factory.finish((|| -> Result<()> {let position=u32::try_from(position).map_err(|_|"row split truncation exceeds its bound")?;factory.failed.set(true);factory.split.command_for(7,position,id)?;
            factory.split.work(||open.truncate(decoder,u64::from(position)))?;factory.split.exchange.borrow_mut().complete()?;factory.failed.set(false);Ok(())})());
        self.gate.verdict(kept_to,started.elapsed().as_secs_f64()*1000.0,partial);
        result
    }
}
/// The two sides of the prompt cache's fork (spark_serve cache.rs). The cache forks a LIVE session only to
/// keep it (a snapshot at the prompt's end) and a KEPT session only to run a request from it. So:
/// a live session forks to a kept one, and both ranks keep a second handle of its committed state under a
/// new number (command 15); a kept session forks to a live one that starts at the kept state (command 16).
/// Dropping a kept session releases it on both ranks (command 17). No word of state is copied anywhere.
impl<'a,C:Compute,E:Exchange> Session<'a,C,E> {
    // Cold and never inlined, with `release_kept`: a request forks and releases once, and the step code stays as it was.
    #[cold] #[inline(never)]
    fn fork_inner(&self)->Result<Self> {
        let factory=self.factory;if let Some(error)=factory.failure_reason() {return Err(error);}
        if !factory.keep {return Err("row split service keeps no prompt".into());}
        let id=factory.next_id.get();let next=id.checked_add(1).ok_or("session identity exhausted")?;
        let fresh=|decoder,state,kept|Session {factory,decoder,state,id,open:None,strict:false,depth:0,step_ms:None,pass_ms:[None;17],after_step:false,gate:Gate::default(),limit:0,kept};
        match (&self.kept,&self.state) {
            (None,Some(state)) if self.open.is_none()=>{
                // A refusal here reaches no rank, so the service stays whole and the cache only skips its snapshot.
                if factory.kept.get()>=KEPT_PROMPTS {return Err("row split service keeps no more prompts".into());}
                // The kept session's owner is made before command 15: if it were refused, no rank would hold a prompt
                // that nothing can release.
                let decoder=Decoder::from_prepared_idle(&factory.prepared,&factory.split,factory.context as u64)?;
                factory.next_id.set(next);let started=std::time::Instant::now();
                let kept=factory.finish((|| {factory.failed.set(true);factory.split.command_for(15,0,self.id)?;let mut word=id.to_le_bytes();factory.split.exchange.borrow_mut().command(&mut word)?;
                    let kept=crate::bounded_decoder::share_state(&factory.split,state)?;factory.split.exchange.borrow_mut().complete()?;factory.failed.set(false);Ok(kept)})())?;
                factory.keep_count(kept.position);
                eprintln!("SERVED KEEP rank={} session={} kept={id} position={} kept_prompts={} kept_bytes={} seconds={:.9}",factory.split.topology.rank,self.id,kept.position,factory.kept.get(),factory.kept_bytes.get(),started.elapsed().as_secs_f64());
                Ok(fresh(decoder,None,Some(kept)))
            },
            (Some(kept),None)=>{
                if factory.active.get()>=8 {return Err("row split service has no free session".into());}
                factory.next_id.set(next);factory.active.set(factory.active.get()+1);let started=std::time::Instant::now();
                let result=(|| {factory.failed.set(true);factory.split.command_for(16,0,self.id)?;let mut word=id.to_le_bytes();factory.split.exchange.borrow_mut().command(&mut word)?;
                    // The other sessions that are open now: the worker counts the same ones.
                    let decoder=Decoder::from_prepared(&factory.prepared,&factory.split,factory.context as u64,factory.active.get()-1)?;
                    let state=crate::bounded_decoder::share_state(&factory.split,kept)?;factory.split.exchange.borrow_mut().complete()?;factory.failed.set(false);
                    eprintln!("SERVED RESUME rank={} session={id} kept={} position={} active={} seconds={:.9}",factory.split.topology.rank,self.id,state.position,factory.active.get(),started.elapsed().as_secs_f64());
                    factory.split.trace_start();Ok(fresh(decoder,Some(state),None))})();
                if result.is_err() {factory.active.set(factory.active.get()-1);}factory.finish(result)
            },
            _=>Err("row split session has no committed state to fork".into()),
        }
    }
    /// The drop of a kept prompt: both ranks forget it. It never counted as an active session.
    #[cold] #[inline(never)]
    fn release_kept(&mut self,kept:State<C::Value>) {let factory=self.factory;factory.release_count(kept.position);drop(kept);
        if !factory.failed.get() {let result=factory.split.command_for(17,0,self.id).and_then(|()|factory.split.exchange.borrow_mut().complete());if let Err(error)=result {factory.remember(error);}}
        if factory.failed.get() {factory.split.exchange.borrow_mut().abort();}
    }
}
/// Exact verification for the server's speculator: every row is a prompt-path row, whose bits equal a plain step.
impl<C:Compute,E:Exchange> spark_engine::draft::Target for Session<'_,C,E> {
    fn checkpoint(&self)->usize {self.open.as_ref().map(Verify::position).or(self.state.as_ref().map(|s|s.position)).unwrap_or(0) as usize}
    fn truncate(&mut self,position:usize)->Result<()> {self.keep(position)}
    fn step(&mut self,token:u32)->Result<Vec<f32>> {self.rows(&[token])?.pop().ok_or_else(||"verify pass has no row".into())}
    fn verify_batch(&mut self,tokens:&[u32])->Result<Vec<Vec<f32>>> {self.rows(tokens)}
    fn max_verify_rows(&self)->usize {crate::bounded_decoder::VERIFY_ROWS}
    fn model_draft_width(&self)->usize {self.depth.min(self.factory.predictor)}
    // With model proposals the Speculator's native chooser picks the depth: it counts each depth's
    // hits over the whole request and prices rows with this request's own measured times.
    fn native_draft_policy(&self)->bool {self.depth>0}
    fn adaptive_model_depth(&self)->bool {true}
    // How deep the next proposal is (price.rs): a request of depth 2 or more proposes its whole depth or nothing, by
    // its own measured record or when the caller asks for the whole depth; a request of depth 1 takes the caller's depth.
    fn limit_model_depth(&mut self,depth:usize) {
        let whole=self.depth.min(self.factory.predictor);
        self.limit=match self.step_ms {Some(step)=>self.gate.pay.depth(whole,self.gate.begun(),depth,step),None=>depth};
    }
    fn model_verify_ms(&self,rows:usize)->Option<f64> {
        if rows==0 {return None;}let cost=self.pass_cost(rows)?;
        // A pass of two rows or more also pays what a cycle costs beside it: the waiting rows and the verdict.
        Some(if rows==1 {cost} else {cost+self.gate.beside_ms()})
    }
    fn model_draft_costs(&self)->Option<spark_engine::draft::CostCurve> {self.measured_costs()}
    fn propose_model(&mut self,_history:&[u32],next:u32,limit:usize)->Result<Option<spark_engine::draft::ModelProposal>> {
        // The rows that waited for the layer are catch-up: the chooser prices a proposal without them.
        self.gate.begin();
        if !self.gate.allows(self.step_ms) {return Ok(Some(spark_engine::draft::ModelProposal {tokens:Vec::new(),proposal_ms:0.0,catchup_ms:0.0}));}
        let limit=limit.min(self.limit);if limit==0 {return Ok(Some(spark_engine::draft::ModelProposal {tokens:Vec::new(),proposal_ms:0.0,catchup_ms:0.0}));}
        let entry=spark_engine::draft::Target::checkpoint(self) as u64;let waiting=self.open.as_ref().map_or(0,Verify::waiting);
        let started=std::time::Instant::now();let (tokens,waited)=self.propose(next,limit)?;let total=started.elapsed().as_secs_f64();
        if !tokens.is_empty() {self.gate.proposed((total-waited).max(0.0)*1000.0,waited*1000.0,entry,started,tokens.len(),waiting.saturating_sub(1));}
        Ok(Some(spark_engine::draft::ModelProposal {tokens,proposal_ms:(total-waited).max(0.0)*1000.0,catchup_ms:waited*1000.0}))
    }
    fn verify_costs(&self,_context:usize)->Option<spark_engine::draft::CostCurve> {self.measured_costs()}
}
impl<C:Compute,E:Exchange> spark_serve::Session for Session<'_,C,E> {
    fn configure_request(&mut self,r:&spark_serve::protocol::Request)->Result<()> {
        use spark_serve::protocol::DraftMode;
        if r.mtp_depth.is_some_and(|v|v>self.factory.predictor) || r.draft.is_some_and(|v|!matches!(v,DraftMode::Auto|DraftMode::Plain|DraftMode::Strict)) {return Err("row split service supports plain decoding, strict drafting, and mtp_depth up to its loaded predictor depth".into());}
        // Only the explicit options draft. Every other request decodes by plain steps, as before.
        // mtp_depth asks the predictor layer for proposals; without it strict uses the free ones.
        self.strict=r.draft==Some(DraftMode::Strict);self.depth=if self.strict {r.mtp_depth.unwrap_or(0)} else {0};
        if self.strict && !spark_engine::draft::plain_greedy(r.sampling) {return Err("strict drafting on the row split service requires greedy sampling".into());}Ok(())
    }
    fn draft_target(&mut self)->Option<&mut dyn spark_engine::draft::Target> {if self.strict {Some(self)} else {None}}
    fn step(&mut self,token:u32)->Result<Vec<f32>> {if self.strict || self.open.is_some() {return self.rows(&[token])?.pop().ok_or_else(||"verify pass has no row".into());}self.token(token,false)}
    fn prefill(&mut self,tokens:&[u32])->Result<Vec<f32>> {self.prompt(tokens,true)}
    fn prefill_advance(&mut self,tokens:&[u32])->Result<()> {self.prompt(tokens,false).map(|_|())}
    fn prefill_chunk_size(&self,_:usize)->usize {self.factory.group}
    fn prefill_chunk_min(&self)->usize {crate::bounded_decoder::PROMPT_SHARED}
    fn prefill_owns_chunks(&self)->bool {true}
    fn prefill_slice(&mut self,tokens:&[u32],final_chunk:bool,_budget:Option<std::time::Duration>,_prompt_end:usize)->Result<Option<Vec<f32>>> {
        self.prompt(tokens,final_chunk).map(Some)
    }
    /// A live session forks where it holds a committed state (the cache asks at the prompt's end and at
    /// the end of an answer that stopped). A kept session always forks.
    fn can_fork(&self)->bool {self.factory.keep && !self.factory.failed.get() && (self.kept.is_some() || (self.state.is_some() && self.open.is_none()))}
    /// The recurrent layers cannot be wound back: only the kept position itself restores. A live session
    /// restores nothing for another request; the kept copy of its prompt does.
    fn can_fork_prefix(&self,position:usize)->bool {self.factory.keep && !self.factory.failed.get() && self.kept.as_ref().is_some_and(|kept|kept.position==position as u64)}
    fn checkpoint_reusable_tokens(&self,tokens:usize)->usize {match &self.kept {Some(kept)=>usize::try_from(kept.position).ok().filter(|&position|position<=tokens).unwrap_or(0),None=>tokens}}
    fn fork(&self)->Result<Self> {self.fork_inner()}
    fn fork_prefix(&self,position:usize)->Result<Self> {if !self.can_fork_prefix(position) {return Err("row split service restores only the end of a kept prompt".into());}self.fork_inner()}
    /// What a kept prompt holds: its state at that length. None without the switch, as before.
    fn retained_bytes(&self)->Option<usize> {if !self.factory.keep {return None;}let state=self.kept.as_ref().or(self.state.as_ref())?;Some(self.factory.kept_charge(state.position))}
    /// Drop open rows of a strict request. The scheduler asks for this when a pass answered with a
    /// token the tokenizer does not hold: it then runs the row again by `step`, which reads the logits.
    fn truncate(&mut self,position:usize)->Result<()> {
        if !self.strict || self.open.is_none() {return Err("backend does not support prefix truncation".into());}
        self.keep(position)
    }
}
impl<C:Compute,E:Exchange> Drop for Session<'_,C,E> {
    fn drop(&mut self) {
        if let Some(kept)=self.kept.take() {return self.release_kept(kept);}
        self.factory.capture_report(self.id);
        if !self.factory.failed.get() {let result=self.factory.split.command_for(3,0,self.id).and_then(|()|self.factory.split.exchange.borrow_mut().complete());if let Err(error)=result {self.factory.remember(error);}}
        if self.factory.failed.get() {self.factory.split.exchange.borrow_mut().abort();}
        if self.depth>0 {self.gate.report(self.factory.split.topology.rank,self.id,self.depth,self.step_ms.unwrap_or(0.0));}
        self.state=None;self.open=None;self.factory.tails.borrow_mut().remove(&self.id);self.factory.active.set(self.factory.active.get()-1);}
}
