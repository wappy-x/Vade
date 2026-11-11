//! Checkpoint output head and greedy token feedback. One final position per call.
use crate::{Result,mxfp8_projection::Mxfp8Projection};
use engine_model::{plan::{admission::CheckpointAdmission,glm_index::TensorRole,storage::WeightStorage},safetensors::Dtype};
use spark_gpu_probe::{bf16::rounded,norm_boundary::{NormShape,WeightType},text_edges::{Shape,greedy},causal_ops::{self,Op}};

pub struct Head<'c> {pub(crate) checkpoint:&'c CheckpointAdmission,pub(crate) shape:Shape,pub(crate) norm:NormShape,
    pub(crate) norm_weight:Vec<f32>,pub(crate) projection:Projection<'c>,pub(crate) embedding:String}
pub(crate) enum Projection<'c> {Mx(Mxfp8Projection<'c>),Bf16 {name:String,shape:causal_ops::Shape}}
pub struct HeadReference {pub mean:Vec<f32>,pub normalized:Vec<f32>,pub accumulator:Vec<f32>,pub logits:Vec<f32>,pub next_embedding:Vec<f32>,pub token:u32}
pub(crate) fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(why.into())}}
fn plain(c:&CheckpointAdmission,name:&str)->Result<Vec<f32>> {
    let p=c.upload(name).map_err(|e|e.to_string())?;need(p.storage==WeightStorage::Plain,"head parameter storage is not plain")?;
    let values=match p.dtype {Dtype::Bf16=>spark_gpu_probe::text_reference::embedding(&p.weight)?,
        Dtype::F32=>p.weight.as_chunks::<4>().0.iter().map(|b|f32::from_le_bytes(*b)).collect(),
        _=>return Err("head parameter dtype is unsupported".into())};
    need(values.iter().all(|v|v.is_finite()),"head parameter is not finite")?;Ok(values)
}
impl<'c> Head<'c> {
    pub fn new(checkpoint:&'c CheckpointAdmission)->Result<Self> {
        let d=checkpoint.description();let shape=Shape::new(d.hidden,d.vocab,d.residual.streams)?;
        let mut embedding=None;let mut head=None;let mut norm=None;
        for (name,view) in checkpoint.headers().weights() {
            let slot=match view.binding().operation {TensorRole::Embedding=>&mut embedding,TensorRole::Head=>&mut head,TensorRole::FinalNorm=>&mut norm,_=>continue};
            need(slot.replace(name.clone()).is_none(),"head tensor role is repeated")?;
        }
        let embedding=embedding.ok_or("head embedding is absent")?;let head=head.ok_or("output head is absent")?;let norm=norm.ok_or("final norm is absent")?;
        let e=&checkpoint.headers().weights()[&embedding];
        need(e.logical().storage==WeightStorage::Plain && e.logical().shape==[d.vocab,d.hidden]
            && e.components().len()==1 && e.components()[0].dtype==Dtype::Bf16,"head feedback requires stored BF16 embedding")?;
        let n=&checkpoint.headers().weights()[&norm];
        need(n.logical().shape==[d.hidden] && n.logical().storage==WeightStorage::Plain && n.components().len()==1,"final norm shape or storage differs")?;
        let dtype=match n.components()[0].dtype {Dtype::Bf16=>WeightType::Bf16,Dtype::F32=>WeightType::F32,_=>return Err("final norm dtype is unsupported".into())};
        let norm_shape=NormShape::new(shape.hidden(),d.epsilon as f32,dtype)?;let norm_weight=plain(checkpoint,&norm)?;norm_shape.check_weights(&norm_weight)?;
        let h=&checkpoint.headers().weights()[&head];need(h.logical().shape==[d.vocab,d.hidden],"output head shape differs")?;
        let projection=match h.logical().storage {
            WeightStorage::Mxfp8=>Projection::Mx(Mxfp8Projection::bind_head(checkpoint,&head)?),
            WeightStorage::Plain=>{
                need(h.components().len()==1 && h.components()[0].dtype==Dtype::Bf16,"plain output head requires stored BF16 weights")?;
                let s=causal_ops::Shape {op:Op::Project,heads:0,dim:0,taps:0,rows:shape.vocab() as u32,cols:shape.hidden() as u32,epsilon:0.0,bound:0.0};
                need(s.lengths().is_some(),"BF16 head exceeds projection ABI")?;Projection::Bf16 {name:head,shape:s}
            },_=>return Err("output head storage recipe is unsupported".into()),
        };
        Ok(Self {checkpoint,shape,norm:norm_shape,norm_weight,projection,embedding})
    }
    pub fn shape(&self)->Shape {self.shape}
    pub fn storage(&self)->WeightStorage {match self.projection {Projection::Mx(_)=>WeightStorage::Mxfp8,Projection::Bf16 {..}=>WeightStorage::Plain}}
    /// Stream-major BF16 lanes for one final position. No earlier positions are read.
    pub fn reference(&self,residual:&[f32])->Result<HeadReference> {
        self.reference_hidden(&mean(self.shape,residual)?)
    }
    pub fn reference_hidden(&self,hidden:&[f32])->Result<HeadReference> {
        need(hidden.len()==self.shape.hidden(),"final hidden extent differs")?;spark_gpu_probe::mhc_boundary::check_bf16(hidden)?;
        let normalized=crate::causal_layer::norm(self.norm,hidden,&self.norm_weight);
        let (accumulator,logits)=match &self.projection {
            Projection::Mx(p)=>{let (_,a,y)=p.dynamic_reference_positive_zero(&normalized)?;(a,y)},
            Projection::Bf16 {name,shape}=>{let a=causal_ops::reference(*shape,[&normalized,&[],&[],&[]],&plain(self.checkpoint,name)?)?;
                let y=a.iter().copied().map(rounded).collect();(a,y)},
        };
        let token=greedy(&logits)?;let next_embedding=self.embedding(token)?;
        Ok(HeadReference {mean:hidden.to_vec(),normalized,accumulator,logits,next_embedding,token})
    }
    /// Read only the selected stored row. Repeat it in ascending stream order.
    pub fn embedding(&self,token:u32)->Result<Vec<f32>> {
        self.shape.token(token)?;let width=self.shape.hidden()*2;let mut bytes=vec![0;width];
        self.checkpoint.component_file(&self.embedding).map_err(|e|e.to_string())?
            .read_range(&self.embedding,u64::from(token)*width as u64,&mut bytes).map_err(|e|e.to_string())?;
        Ok(spark_gpu_probe::text_reference::embedding(&bytes)?.repeat(self.shape.streams()))
    }
    /// Hash only consumed parameters. Head rows are read with bounded staging.
    pub fn parameter_record(&self,mut digest:impl FnMut(&[u8])->String)->Result<String> {
        let mut text=format!("shape={:?}\nnorm={:?}\nhead_storage={:?}\n",self.shape,self.norm,self.storage());
        text.push_str(&format!("norm_bits={}\n",digest(&self.norm_weight.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())));
        let names=match &self.projection {Projection::Mx(p)=>p.component_names().into_iter().map(str::to_owned).collect::<Vec<_>>(),Projection::Bf16 {name,..}=>vec![name.clone()]};
        for name in names {
            let f=self.checkpoint.component_file(&name).map_err(|e|e.to_string())?;let t=&f.header().tensors[&name];
            text.push_str(&format!("component={name} dtype={:?} shape={:?} bytes={}\n",t.dtype,t.shape,t.byte_len()));
            let mut offset=0;let mut bytes=vec![0;1024*1024];while offset<t.byte_len() {
                let count=(t.byte_len()-offset).min(bytes.len() as u64) as usize;f.read_range(&name,offset,&mut bytes[..count]).map_err(|e|e.to_string())?;
                text.push_str(&format!("offset={offset} bytes={count} sha256={}\n",digest(&bytes[..count])));offset+=count as u64;
            }
        }Ok(text)
    }
}
pub fn mean(shape:Shape,residual:&[f32])->Result<Vec<f32>> {
    need(residual.len()==shape.residual(),"head residual extent differs")?;spark_gpu_probe::mhc_boundary::check_bf16(residual)?;
    let mut out=vec![0.0;shape.hidden()];for j in 0..shape.hidden() {
        let mut sum=0.0;for stream in 0..shape.streams() {sum+=residual[stream*shape.hidden()+j];}out[j]=rounded(sum/shape.streams() as f32);
    }need(out.iter().all(|v|v.is_finite()),"head stream mean is not finite")?;Ok(out)
}
#[cfg(target_os="linux")]
#[path="head_path/device.rs"] mod device;
#[cfg(target_os="linux")]
pub use device::{ResidentHead,HeadPass};
