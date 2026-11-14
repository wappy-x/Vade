//! Bounded expert device path. Device copies preserve selected-slot order.
use super::*;
use spark_gpu_probe::driver::{Backend,DeviceVector};
struct ResidentProjection<'d> {shape:Shape,weight:DeviceVector<'d>,scales:DeviceVector<'d>}
struct ResidentExpert<'d> {gate:ResidentProjection<'d>,up:ResidentProjection<'d>,down:ResidentProjection<'d>,middle:Shape}
pub struct ResidentExpertPath<'p,'b,'d> {source:&'p ExpertPath,backend:&'b Backend<'d>,experts:BTreeMap<u32,ResidentExpert<'d>>,shared:Option<ResidentExpert<'d>>}
fn raw<'d>(b:&Backend<'d>,bytes:&[u8])->Result<DeviceVector<'d>> {
    let mut v=b.upload_vector(&vec![0.0;bytes.len().div_ceil(4)])?;b.write_vector_bytes(&mut v,bytes)?;b.synchronize()?;Ok(v)
}
impl Projection {
    fn upload<'d>(&self,b:&Backend<'d>)->Result<ResidentProjection<'d>> {Ok(ResidentProjection {shape:self.shape,weight:raw(b,&self.weight.weight)?,scales:raw(b,&self.weight.scales)?})}
}
impl Expert {
    fn upload<'d>(&self,b:&Backend<'d>)->Result<ResidentExpert<'d>> {Ok(ResidentExpert {gate:self.gate.upload(b)?,up:self.up.upload(b)?,down:self.down.upload(b)?,middle:self.middle_shape()})}
}
impl ExpertPath {
    pub fn upload<'p,'b,'d>(&'p self,b:&'b Backend<'d>)->Result<ResidentExpertPath<'p,'b,'d>> {
        b.require_resident_stream()?;let mut experts=BTreeMap::new();for (&id,e) in &self.experts {experts.insert(id,e.upload(b)?);}
        Ok(ResidentExpertPath {source:self,backend:b,experts,shared:self.shared.as_ref().map(|e|e.upload(b)).transpose()?})
    }
}
impl<'d> ResidentProjection<'d> {
    fn apply(&self,b:&Backend<'d>,x:&DeviceVector<'_>,name:&str,observe:&mut impl FnMut(&str,&DeviceVector<'_>)->Result<()>)->Result<DeviceVector<'d>> {
        let q=if self.shape.op==Op::NvProject {let q=b.expert_operation(Shape {op:Op::Quantize,..self.shape},[Some(x),None,None,None],None,None)?;
            observe(&format!("{name}.quant"),&q)?;Some(q)} else {None};
        let v=b.expert_operation(self.shape,[Some(q.as_ref().unwrap_or(x)),None,None,None],Some(&self.weight),Some(&self.scales))?;
        observe(name,&v)?;Ok(v)
    }
}
impl<'d> ResidentExpert<'d> {
    fn apply(&self,b:&Backend<'d>,x:&DeviceVector<'_>,name:&str,observe:&mut impl FnMut(&str,&DeviceVector<'_>)->Result<()>)->Result<DeviceVector<'d>> {
        let g=self.gate.apply(b,x,&format!("{name}.gate"),observe)?;let u=self.up.apply(b,x,&format!("{name}.up"),observe)?;
        let m=b.expert_operation(self.middle,[Some(&g),Some(&u),None,None],None,None)?;observe(&format!("{name}.middle"),&m)?;
        self.down.apply(b,&m,&format!("{name}.down"),observe)
    }
}
impl<'d> ResidentExpertPath<'_,'_,'d> {
    /// The caller supplies corrected routing. This method never selects experts.
    /// The bounded probe copies complete down rows. It makes no speed claim.
    pub fn apply(&self,input:&DeviceVector<'_>,ids:&[u32],weights:&[f32],mut observe:impl FnMut(&str,&DeviceVector<'_>)->Result<()>)->Result<DeviceVector<'d>> {
        self.source.route(ids,weights)?;need(input.len()==self.source.hidden && !input.is_segmented(),"expert input extent differs")?;
        let b=self.backend;let values=b.read_vector_values(input)?;need(values.iter().all(|v|v.is_finite()),"expert input is not finite")?;
        observe("input",input)?;let mut down=b.upload_vector(&vec![0.0;self.source.topk*self.source.hidden])?;
        for (slot,id) in ids.iter().enumerate() {let v=self.experts[id].apply(b,input,&format!("slot{slot}"),&mut observe)?;
            b.copy_range(&v,&mut down,0,slot*self.source.hidden,self.source.hidden)?;b.synchronize()?;}
        let shared=if let Some(e)=&self.shared {e.apply(b,input,"shared",&mut observe)?} else {b.upload_vector(&vec![0.0;self.source.hidden])?};
        observe("shared_output",&shared)?;observe("expert_down",&down)?;
        let ids=b.upload_vector(&ids.iter().map(|&id|id as f32).collect::<Vec<_>>())?;let weights=b.upload_vector(weights)?;
        let out=b.expert_operation(self.source.fold_shape(),[Some(&down),Some(&ids),Some(&weights),Some(&shared)],None,None)?;
        observe("output",&out)?;Ok(out)
    }
}
