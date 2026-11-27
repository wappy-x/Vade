//! Stored-precision norm and plain BF16 projection operands for deferred layers.
//! These candidate bindings do not open the full layer's execution gate.
use crate::{Result,layer_driver::{LayerBoundary,MhcOperands}};
use engine_model::{plan::{admission::{CheckpointAdmission,ComponentView,UploadWeight,WeightView},
    layer::{Kind,Topology,cast::Dtype},storage::WeightStorage},safetensors::Dtype as StoredDtype};
use spark_gpu_probe::norm_boundary::{NormShape,WeightType};

fn need(ok: bool,why: &str) -> Result<()> {if ok {Ok(())} else {Err(why.into())}}
fn loaded(view: &WeightView,checkpoint: &CheckpointAdmission) -> Result<UploadWeight> {
    let name=&view.logical().name;
    need(checkpoint.headers().weights().get(name)==Some(view),"operand checkpoint headers differ from the bound weight")?;
    checkpoint.upload(name).map_err(|e| e.to_string())
}
pub struct NormOperands {shape: NormShape,weight: Vec<f32>,stored: StoredDtype}
impl NormOperands {
    pub fn shape(&self) -> NormShape {self.shape}
    pub fn weight(&self) -> &[f32] {&self.weight}
    pub fn stored_dtype(&self) -> StoredDtype {self.stored}
    #[cfg(target_os="linux")]
    pub fn upload<'b,'d>(&self,backend: &'b spark_gpu_probe::driver::Backend<'d>) -> Result<spark_gpu_probe::driver::ResidentNorm<'b,'d>> {
        backend.resident_norm(self.shape,&self.weight)
    }
}
pub struct ProjectionOperands {shape: [usize;2],origin: u64,weight: Vec<u8>,topology: Topology}
impl ProjectionOperands {
    pub fn shape(&self) -> [usize;2] {self.shape}
    pub fn row_origin(&self) -> u64 {self.origin}
    pub fn weight(&self) -> &[u8] {&self.weight}
    pub fn topology(&self) -> Topology {self.topology}
}
impl LayerBoundary<'_> {
    /// Use the exact stored parameter dtype. This is a candidate identity load,
    /// not proof of the checkpoint runtime's effective parameter conversion.
    pub fn prepare_stored_norm(&self,step: usize,mut read: impl FnMut(&ComponentView)->engine_model::Result<Vec<u8>>) -> Result<NormOperands> {
        self.pack_norm(step,|view| view.upload(&mut read).map_err(|e| e.to_string()))
    }
    pub fn prepare_stored_norm_checkpoint(&self,step: usize,checkpoint: &CheckpointAdmission) -> Result<NormOperands> {
        self.pack_norm(step,|view| loaded(view,checkpoint))
    }
    fn norm_view(&self,step: usize) -> Result<&WeightView> {
        let op=self.program().steps.get(step).ok_or("layer step is outside the program")?;
        need(op.kind==Kind::Norm,"step is not an ordinary RMSNorm")?;
        let weights=self.weights(step)?;
        need(weights.len()==1 && weights[0].view().logical().shape.len()==1
            && weights[0].view().logical().storage==WeightStorage::Plain,"ordinary norm requires one plain weight vector")?;
        Ok(weights[0].view())
    }
    fn pack_norm(&self,step: usize,mut load: impl FnMut(&WeightView)->Result<UploadWeight>) -> Result<NormOperands> {
        let view=self.norm_view(step)?;
        let width=usize::try_from(view.logical().shape[0]).map_err(|_| "norm width exceeds address space")?;
        let dtype=view.components()[0].dtype;
        let weight_type=match dtype {StoredDtype::Bf16=>WeightType::Bf16,StoredDtype::F32=>WeightType::F32,
            _=>return Err("norm parameter has no lossless FP32 operand conversion".into())};
        let shape=NormShape::new(width,self.program().settings.epsilon as f32,weight_type)?;
        let upload=load(view)?;
        need(upload.dtype==dtype,"norm upload dtype differs from the bound header")?;
        let weight=match dtype {
            StoredDtype::Bf16=>upload.weight.as_chunks::<2>().0.iter().map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b))<<16)).collect::<Vec<_>>(),
            StoredDtype::F32=>upload.weight.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
            _=>unreachable!(),
        };
        shape.check_weights(&weight)?;Ok(NormOperands {shape,weight,stored:dtype})
    }
    /// Only an ordinary BF16 projection with a BF16 output is bound here.
    /// Router FP32 arithmetic and quantized Linear replacements remain refused.
    pub fn prepare_bf16_projection(&self,step: usize,mut read: impl FnMut(&ComponentView)->engine_model::Result<Vec<u8>>) -> Result<ProjectionOperands> {
        self.pack_projection(step,|view| view.upload(&mut read).map_err(|e| e.to_string()))
    }
    pub fn prepare_bf16_projection_checkpoint(&self,step: usize,checkpoint: &CheckpointAdmission) -> Result<ProjectionOperands> {
        self.pack_projection(step,|view| loaded(view,checkpoint))
    }
    fn pack_projection(&self,step: usize,mut load: impl FnMut(&WeightView)->Result<UploadWeight>) -> Result<ProjectionOperands> {
        let p=self.program();let op=p.steps.get(step).ok_or("layer step is outside the program")?;
        need(op.kind==Kind::Project && self.casts().steps[step].outputs[0].dtype==Dtype::Bf16,
            "step requires an unbound projection kind or output precision")?;
        let weights=self.weights(step)?;need(weights.len()==1,"projection requires one weight")?;
        let view=weights[0].view();let logical=view.logical();
        need(logical.storage==WeightStorage::Plain && logical.shape.len()==2 && view.components()[0].dtype==StoredDtype::Bf16,
            "projection requires plain BF16 storage; FP32 and quantized bindings remain open")?;
        let (origin,rows)=weights[0].rows().map_or((0,logical.shape[0]),|r| {let range=r.rows();(range.start,range.end-range.start)});
        let cols=logical.shape[1];
        need(cols%16==0,"native BF16 projection requires complete input tiles")?;
        let shape=[usize::try_from(rows).map_err(|_| "projection rows exceed address space")?,
            usize::try_from(cols).map_err(|_| "projection columns exceed address space")?];
        need(shape.iter().all(|&n| n>0 && n<=u32::MAX as usize),"projection dimensions exceed the kernel ABI")?;
        let upload=load(view)?;
        let bytes=upload.matrix_view([origin,0],[rows,cols]).map_err(|e| e.to_string())?;
        need(bytes.weight.len()==shape[0].checked_mul(shape[1]).and_then(|n| n.checked_mul(2)).ok_or("projection byte extent overflow")?,
            "projection slice is not a complete row interval")?;
        need(bytes.weight.as_chunks::<2>().0.iter().all(|b| u16::from_le_bytes(*b)&0x7f80!=0x7f80),
            "projection operand is not finite BF16")?;
        Ok(ProjectionOperands {shape,origin,weight:bytes.weight.to_vec(),topology:p.topology})
    }
    /// Bind the adjacent mHC prepare, collapse, and ordinary norm operations.
    /// Retain the mHC pass until the future mixer or FFN producer finishes.
    pub fn prepare_residual_block_checkpoint(&self,step: usize,checkpoint: &CheckpointAdmission) -> Result<ResidualBlockOperands> {
        let steps=&self.program().steps;
        let ops=steps.get(step..step.checked_add(3).ok_or("block step overflow")?).ok_or("residual block is incomplete")?;
        need(ops[0].kind==Kind::MhcPrepare && ops[1].kind==Kind::Collapse && ops[2].kind==Kind::Norm
            && ops[1].inputs==[ops[0].inputs[0].clone(),ops[0].outputs[0].name.clone()]
            && ops[2].inputs==[ops[1].outputs[0].name.clone()],"residual block data flow differs")?;
        let mhc=self.prepare_mhc_checkpoint(step,checkpoint)?;
        let norm=self.prepare_stored_norm_checkpoint(step+2,checkpoint)?;
        need(norm.shape().width()==mhc.shape().hidden as usize,"residual block norm width differs")?;
        Ok(ResidualBlockOperands {mhc,norm})
    }
}
pub struct ResidualBlockOperands {mhc: MhcOperands,norm: NormOperands}
impl ResidualBlockOperands {
    pub fn mhc(&self) -> &MhcOperands {&self.mhc}
    pub fn norm(&self) -> &NormOperands {&self.norm}
}
#[cfg(target_os="linux")]
#[path="layer_operands/device.rs"]
pub mod device;
