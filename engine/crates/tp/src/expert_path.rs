//! Checkpoint-bound routed and shared experts. Routing remains a caller input.
use crate::Result;
use engine_model::{plan::{admission::{CheckpointAdmission,UploadWeight},storage::WeightStorage},shapes::Ffn};
use spark_gpu_probe::expert_ops::{self,Shape,Op};
use std::collections::{BTreeMap,BTreeSet};
pub type Planes=BTreeMap<String,Vec<f32>>;
pub(crate) struct Projection {pub(crate) shape:Shape,pub(crate) weight:UploadWeight}
pub(crate) struct Expert {pub(crate) gate:Projection,pub(crate) up:Projection,pub(crate) down:Projection}
pub struct ExpertPath {pub(crate) experts:BTreeMap<u32,Expert>,pub(crate) shared:Option<Expert>,pub(crate) hidden:usize,
    pub(crate) topk:usize,pub(crate) count:u64,pub(crate) bound:f32,pub(crate) layer:usize}
fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(why.into())}}
fn projection(c:&CheckpointAdmission,name:&str,rows:u64,cols:u64,routed:bool,bound:f32,role:u32)->Result<Projection> {
    let w=c.upload(name).map_err(|e|e.to_string())?;
    need(w.shape==[rows,cols],"expert weight shape differs from checkpoint recipe")?;
    let (op,br,bc,global,activation)=match w.storage {
        WeightStorage::Nvfp4 {activation_scale:true,..} if routed=>(Op::NvProject,1,16,w.global_scale.ok_or("expert global scale is absent")?,w.input_scale.ok_or("expert input scale is absent")?),
        WeightStorage::Fp8Block {block:[br,bc]} if !routed=>(Op::Fp8Project,br,bc,1.0,1.0),
        _=>return Err("expert storage differs from the calibrated NVFP4 or weight-only FP8 contract".into()),
    };
    let s=Shape {op,tokens:1,rows:rows.try_into().map_err(|_|"expert rows exceed ABI")?,cols:cols.try_into().map_err(|_|"expert columns exceed ABI")?,
        block_rows:br.try_into().map_err(|_|"expert block rows exceed ABI")?,block_cols:bc.try_into().map_err(|_|"expert block columns exceed ABI")?,global,activation,bound,role};
    let (_,wb,sb,_,_)=s.lengths().ok_or("expert geometry exceeds ABI")?;
    need(w.weight.len()==wb && w.scales.len()==sb,"expert weight byte extents differ")?;
    if routed {need(w.scales.iter().all(|&b|b<127),"invalid expert NVFP4 scale")?;}
    else {need(w.weight.iter().all(|b|b&127!=127),"invalid shared expert FP8 weight")?;}
    Ok(Projection {shape:s,weight:w})
}
fn expert(c:&CheckpointAdmission,prefix:&str,h:u64,width:u64,routed:bool,bound:f32)->Result<Expert> {
    // The word function of a routed (NVFP4) product follows the role of the matrix: down is a SECOND matrix (quarter
    // chains), gate and up are FIRST matrices. The same rule as the resident weights (`resident_weights.rs`).
    let role=|name:&str|if routed {crate::matrix_role::nv_role(name)} else {expert_ops::NV_FIRST};
    let (gate,up,down)=(format!("{prefix}.gate_proj.weight"),format!("{prefix}.up_proj.weight"),format!("{prefix}.down_proj.weight"));
    Ok(Expert {gate:projection(c,&gate,width,h,routed,bound,role(&gate))?,up:projection(c,&up,width,h,routed,bound,role(&up))?,
        down:projection(c,&down,h,width,routed,bound,role(&down))?})
}
impl ExpertPath {
    /// Load only the named expert union. IDs need not have selected-slot order.
    /// The retained checkpoint descriptors supply all bytes and calibration scales.
    pub fn new(c:&CheckpointAdmission,layer:usize,ids:&[u32])->Result<Self> {
        let d=c.description();let block=d.layers.get(layer).ok_or("expert layer is outside the decoder")?;
        let Ffn::Experts(e)=block.ffn else {return Err("expert path requires an MoE layer".into());};
        need(!ids.is_empty() && ids.iter().all(|&id|u64::from(id)<e.count),"expert union is empty or outside checkpoint geometry")?;
        let prefix=format!("model.language_model.layers.{layer}.mlp");let bound=d.swiglu_limit as f32;
        need(bound.is_finite() && bound>0.0,"expert SwiGLU bound is invalid")?;
        let mut experts=BTreeMap::new();for id in ids.iter().copied().collect::<BTreeSet<_>>() {
            experts.insert(id,expert(c,&format!("{prefix}.experts.{id}"),d.hidden,e.width,true,bound)?);
        }
        let shared=if e.shared_width>0 {Some(expert(c,&format!("{prefix}.shared_experts"),d.hidden,e.shared_width,false,bound)?)} else {None};
        Ok(Self {experts,shared,hidden:d.hidden as usize,topk:e.selected as usize,count:e.count,bound,layer})
    }
    pub fn hidden(&self)->usize {self.hidden}
    /// Chains a word (`expert_ops::nv_chains`) of one loaded routed expert's gate, up and down products.
    pub fn chains(&self,id:u32)->Option<[u32;3]> {self.experts.get(&id).map(|e|[&e.gate,&e.up,&e.down].map(|p|expert_ops::nv_chains(&p.shape)))}
    pub fn top_k(&self)->usize {self.topk}
    pub(crate) fn route(&self,ids:&[u32],weights:&[f32])->Result<()> {
        need(ids.len()==self.topk && weights.len()==ids.len() && weights.iter().all(|v|v.is_finite() && *v>=0.0)
            && ids.iter().all(|id|self.experts.contains_key(id) && u64::from(*id)<self.count)
            && ids.iter().copied().collect::<BTreeSet<_>>().len()==ids.len(),"expert route extent, ID, or weight differs")
    }
    pub fn reference(&self,input:&[f32],ids:&[u32],weights:&[f32])->Result<Planes> {
        self.route(ids,weights)?;need(input.len()==self.hidden,"expert input feature extent differs")?;
        let mut planes=Planes::new();planes.insert("input".into(),input.to_vec());let mut down=Vec::new();
        for (slot,id) in ids.iter().enumerate() {down.extend(self.experts[id].reference(input,&format!("slot{slot}"),&mut planes)?);}
        let shared=if let Some(e)=&self.shared {e.reference(input,"shared",&mut planes)?} else {vec![0.0;self.hidden]};
        let id_lanes=ids.iter().map(|&id|id as f32).collect::<Vec<_>>();
        let output=expert_ops::reference(self.fold_shape(),[&down,&id_lanes,weights,&shared],&[],&[])?;
        planes.insert("expert_down".into(),down);planes.insert("shared_output".into(),shared);planes.insert("output".into(),output);Ok(planes)
    }
    pub(crate) fn fold_shape(&self)->Shape {Shape {op:Op::Fold,tokens:1,rows:self.hidden as u32,cols:self.topk as u32,block_rows:1,block_cols:1,global:1.0,activation:1.0,bound:self.bound,role:0}}
    pub fn parameter_record(&self,mut digest:impl FnMut(&[u8])->String)->String {
        let mut text=format!("layer={} hidden={} topk={} bound={}\n",self.layer,self.hidden,self.topk,self.bound);
        for (name,e) in self.experts.iter().map(|(id,e)|(format!("expert.{id}"),e)).chain(self.shared.iter().map(|e|("shared".into(),e))) {
            for (part,p) in [("gate",&e.gate),("up",&e.up),("down",&e.down)] {
                text.push_str(&format!("{name}.{part} shape={:?} weight={} scales={}\n",p.shape,digest(&p.weight.weight),digest(&p.weight.scales)));
            }
        }text
    }
}
impl Projection {
    fn reference(&self,x:&[f32],name:&str,planes:&mut Planes)->Result<Vec<f32>> {
        let input=if self.shape.op==Op::NvProject {
            let q=expert_ops::reference(Shape {op:Op::Quantize,..self.shape},[x,&[],&[],&[]],&[],&[])?;
            planes.insert(format!("{name}.quant"),q.clone());q
        } else {x.to_vec()};
        let v=expert_ops::reference(self.shape,[&input,&[],&[],&[]],&self.weight.weight,&self.weight.scales)?;
        planes.insert(name.into(),v.clone());Ok(v)
    }
}
impl Expert {
    pub(crate) fn middle_shape(&self)->Shape {Shape {op:Op::SwiGlu,..self.gate.shape}}
    fn reference(&self,x:&[f32],name:&str,planes:&mut Planes)->Result<Vec<f32>> {
        let gate=self.gate.reference(x,&format!("{name}.gate"),planes)?;let up=self.up.reference(x,&format!("{name}.up"),planes)?;
        let middle=expert_ops::reference(self.middle_shape(),[&gate,&up,&[],&[]],&[],&[])?;
        planes.insert(format!("{name}.middle"),middle.clone());self.down.reference(&middle,&format!("{name}.down"),planes)
    }
}
#[cfg(target_os="linux")]
#[path="expert_path/device.rs"] mod device;
#[cfg(target_os="linux")]
pub use device::ResidentExpertPath;
