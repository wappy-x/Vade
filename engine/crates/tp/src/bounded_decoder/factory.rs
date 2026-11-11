//! Bounded checkpoint Factory for the causal recipe. Drafting and cache forks stay closed.
use super::*;
use spark_gpu_probe::driver::Backend;
use spark_serve::text_program::{Factory as FactoryTrait,TokenProgram};
pub struct Factory<'f,'b,'d> {pub checkpoint:&'f CheckpointAdmission,pub backend:&'b Backend<'d>}
/// All sessions share immutable resident weights and the ordered rank group.
/// The caller schedules both ranks in the same token and request order.
pub struct ResidentFactory<'f,'b,'d> {pub checkpoint:&'f CheckpointAdmission,pub compute:Device<'b,'d>}
pub struct Program<'f,'b,'d> {decoder:Decoder<'f,Device<'b,'d>>,state:State<Value<'b,'d>>,pending:Option<Vec<BTreeMap<String,Value<'b,'d>>>>,next:usize,stage:u8}
impl<'f,'b,'d> FactoryTrait for Factory<'f,'b,'d> {
    type Program<'a>=Program<'f,'b,'d> where Self:'a;
    fn create(&self,description:&engine_model::plan::TextDescription,edges:&engine_model::plan::text_edges::TextEdges,context:usize)->Result<Self::Program<'_>> {
        program(self.checkpoint,Device::new(self.backend),description,edges,context)
    }
}
impl<'f,'b,'d> FactoryTrait for ResidentFactory<'f,'b,'d> {
    type Program<'a>=Program<'f,'b,'d> where Self:'a;
    fn create(&self,description:&engine_model::plan::TextDescription,edges:&engine_model::plan::text_edges::TextEdges,context:usize)->Result<Self::Program<'_>> {
        program(self.checkpoint,self.compute.clone(),description,edges,context)
    }
}
fn program<'f,'b,'d>(checkpoint:&'f CheckpointAdmission,compute:Device<'b,'d>,description:&engine_model::plan::TextDescription,edges:&engine_model::plan::text_edges::TextEdges,context:usize)->Result<Program<'f,'b,'d>> {
    need(description==checkpoint.description() && edges.hidden==description.hidden && edges.vocab==description.vocab
        && edges.streams==description.residual.streams,"bounded Factory checkpoint identity differs")?;
    compute.backend.require_resident_stream()?;
    let decoder=Decoder::new(checkpoint,compute,context as u64)?;let state=decoder.zero_state()?;
    Ok(Program {decoder,state,pending:None,next:0,stage:0})
}
impl<'f,'b,'d> TokenProgram for Program<'f,'b,'d> {
    type Residual=Value<'b,'d>;type Hidden=Value<'b,'d>;
    fn embed(&mut self,token:u32,position:usize)->Result<Value<'b,'d>> {
        need(self.pending.is_none() && position as u64==self.state.position && self.state.position<self.decoder.capacity,"bounded text transaction position differs")?;
        let v=self.decoder.embedding(token)?;
        self.pending=Some(Vec::new());self.next=0;self.stage=0;Ok(v)
    }
    fn layer(&mut self,index:usize,operation:&engine_model::plan::description::DecoderDescription,residual:Value<'b,'d>)->Result<Value<'b,'d>> {
        need(self.pending.is_some() && self.stage==0 && index==self.next,"bounded layer order differs")?;
        need(self.decoder.checkpoint.description().layers.get(index)==Some(operation),"bounded layer recipe identity differs")?;
        let (out,next)=self.decoder.layer(index,self.state.position+1,&residual,&self.state.layers[index])?;
        self.pending.as_mut().unwrap().push(next);self.next+=1;eprintln!("BOUNDED LAYER position={} layer={index} complete=true",self.state.position);Ok(out)
    }
    fn mean(&mut self,residual:Value<'b,'d>)->Result<Value<'b,'d>> {
        let d=self.decoder.checkpoint.description();need(self.pending.is_some() && self.stage==0 && self.next==d.layers.len(),"bounded text mean precedes a complete stack")?;
        let out=self.decoder.compute.mean(&residual,number(d.residual.streams)?,number(d.hidden)?)?;self.stage=1;Ok(out)
    }
    fn norm(&mut self,hidden:Value<'b,'d>)->Result<Value<'b,'d>> {
        need(self.pending.is_some() && self.stage==1,"bounded final norm order differs")?;let c=self.decoder.checkpoint;let name="model.language_model.norm.weight";
        let out=self.decoder.compute.norm(name,norm_shape(c,name,c.description().hidden as usize)?,&hidden,&self.decoder.parameters[name])?;self.stage=2;Ok(out)
    }
    fn head(&mut self,hidden:Value<'b,'d>)->Result<Vec<f32>> {
        need(self.pending.is_some() && self.stage==2,"bounded text head order differs")?;
        let p=Mxfp8Projection::matrix(self.decoder.checkpoint,"lm_head.weight",WeightType::F32)?;
        let v=self.decoder.compute.read(&self.decoder.compute.mx(&p,&hidden)?)?;
        need(v.len()==self.decoder.checkpoint.description().vocab as usize && v.iter().all(|v|v.is_finite()),"bounded text head returned invalid logits")?;self.stage=3;Ok(v)
    }
    fn commit(&mut self)->Result<()> {
        need(self.stage==3 && self.next==self.decoder.checkpoint.description().layers.len(),"bounded commit precedes the text head")?;
        self.decoder.compute.backend.synchronize()?;if let Some(e)=self.decoder.compute.backend.fatal_error() {return Err(e);}
        let layers=self.pending.take().ok_or("bounded commit has no transaction")?;self.state=State {position:self.state.position+1,layers,expanded:BTreeMap::new()};self.stage=0;self.next=0;self.decoder.compute.report();Ok(())
    }
    fn abort(&mut self)->Result<()> {let result=self.decoder.compute.backend.synchronize();self.pending=None;self.stage=0;self.next=0;result}
}
impl Drop for Program<'_,'_,'_> {fn drop(&mut self) {let _=self.decoder.compute.backend.synchronize();}}
