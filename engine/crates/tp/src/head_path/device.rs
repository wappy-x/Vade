//! Completed head outputs and stored token feedback. No decoder state is changed.
use super::*;
use spark_gpu_probe::driver::{Backend,DeviceVector,ResidentNorm};
use crate::dynamic_projection::ResidentDynamicProjection;
enum ResidentProjection<'b,'d> {Mx(ResidentDynamicProjection<'b,'d>),Bf16 {shape:causal_ops::Shape,weight:DeviceVector<'d>}}
pub struct ResidentHead<'h,'c,'b,'d> {head:&'h Head<'c>,backend:&'b Backend<'d>,norm:ResidentNorm<'b,'d>,projection:ResidentProjection<'b,'d>}
pub struct HeadPass<'d> {pub mean:DeviceVector<'d>,pub normalized:DeviceVector<'d>,pub accumulator:DeviceVector<'d>,pub logits:DeviceVector<'d>,pub next_embedding:DeviceVector<'d>,pub token:u32}
fn copy<'d>(b:&Backend<'d>,x:&DeviceVector<'_>)->Result<DeviceVector<'d>> {
    let mut out=b.vector(x.len())?;b.copy_prefix(x,&mut out,x.len())?;b.synchronize()?;Ok(out)
}
impl<'c> Head<'c> {
    pub fn upload<'h,'b,'d>(&'h self,backend:&'b Backend<'d>)->Result<ResidentHead<'h,'c,'b,'d>> {
        backend.require_resident_stream()?;
        let norm=backend.resident_norm(self.norm,&self.norm_weight)?;
        let projection=match &self.projection {Projection::Mx(p)=>ResidentProjection::Mx(p.load_dynamic(backend)?),
            Projection::Bf16 {name,shape}=>ResidentProjection::Bf16 {shape:*shape,weight:backend.upload_vector(&plain(self.checkpoint,name)?)?}};
        backend.synchronize()?;Ok(ResidentHead {head:self,backend,norm,projection})
    }
}
impl<'d> ResidentHead<'_,'_,'_,'d> {
    /// Compute the final stream mean and complete the head and token feedback.
    pub fn step(&self,residual:&DeviceVector<'_>)->Result<HeadPass<'d>> {
        // The typed decoder boundary guarantees BF16. Check external callers too.
        let b=self.backend;let values=b.read_vector_values(residual)?;
        need(values.len()==self.head.shape.residual(),"head residual extent differs")?;
        spark_gpu_probe::mhc_boundary::check_bf16(&values)?;
        let hidden=b.text_mean(self.head.shape,residual)?;self.logits(&hidden)
    }
    /// Accept an already completed BF16 stream mean for the final position.
    pub fn logits(&self,hidden:&DeviceVector<'_>)->Result<HeadPass<'d>> {
        let b=self.backend;b.require_resident_stream()?;
        need(hidden.len()==self.head.shape.hidden() && !hidden.is_segmented(),"final hidden extent differs")?;
        spark_gpu_probe::mhc_boundary::check_bf16(&b.read_vector_values(hidden)?)?;
        let input=b.round_bf16_lanes(hidden)?;let normalized=self.norm.apply(&input)?;
        let (accumulator,logits)=match &self.projection {
            ResidentProjection::Mx(p)=>{let pass=p.apply_vector(normalized.vector())?;
                (copy(b,pass.projection().accumulator())?,copy(b,pass.projection().output().vector())?)},
            ResidentProjection::Bf16 {shape,weight}=>{let a=b.causal_operation(*shape,[Some(normalized.vector()),None,None,None],Some(weight))?;
                let y=copy(b,b.round_bf16_lanes(&a)?.vector())?;(a,y)},
        };
        let mut partials=b.vector(self.head.shape.vocab().div_ceil(1024)*3)?;let mut selected=b.vector(3)?;
        let (token,_)=b.argmax_device(&logits,&mut partials,&mut selected)?;
        let next_embedding=self.embed(token)?;b.synchronize()?;if let Some(e)=b.fatal_error() {return Err(e);}
        Ok(HeadPass {mean:copy(b,hidden)?,normalized:copy(b,normalized.vector())?,accumulator,logits,next_embedding,token})
    }
    /// Use the proved stored-row loader and batch embedding kernel from 29d596b.
    pub fn embed(&self,token:u32)->Result<DeviceVector<'d>> {
        use spark_gpu_probe::load::{WeightRange,WeightFile,WeightDtype};
        let b=self.backend;let h=self.head;h.shape.token(token)?;
        let file=h.checkpoint.component_file(&h.embedding).map_err(|e|e.to_string())?;let tensor=&file.header().tensors[&h.embedding];
        let bytes=h.shape.hidden()*2;
        let offset=file.header().data_start.checked_add(tensor.start).and_then(|n|n.checked_add(u64::from(token)*bytes as u64)).ok_or("head embedding offset overflow")?;
        let ranges=[WeightRange {file_offset:offset,destination:0,bytes,row_bytes:0,row_stride:0}];
        let arena=b.load_weight_arena(bytes,&[WeightFile {file:file.file(),ranges:&ranges}])?;
        let table=b.weight_matrix(&arena,0,1,h.shape.hidden(),WeightDtype::Bf16)?;
        let ids=b.upload_vector(&vec![0.0;h.shape.streams()])?;let mut out=b.vector(h.shape.residual())?;
        let launched=b.batch_embedding_device(&table,&ids,&mut out);let completed=b.synchronize();launched.and(completed)?;Ok(out)
    }
}
