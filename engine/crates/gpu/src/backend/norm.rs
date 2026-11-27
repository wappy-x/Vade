//! Owned ordinary norm operands. Each call drains its stream before return.
//! The existing model RMS and pointwise kernels supply one fixed reduction tree.
use super::*;
use crate::norm_boundary::{NormShape,WeightType};

pub struct ResidentNorm<'b,'a> {backend: &'b Backend<'a>,shape: NormShape,weight: DeviceVector<'a>}
pub enum NormOutput<'a> {Bf16(Bf16Vector<'a>),F32(DeviceVector<'a>)}
impl<'a> NormOutput<'a> {
    pub fn vector(&self) -> &DeviceVector<'_> {match self {Self::Bf16(v)=>v.vector(),Self::F32(v)=>v}}
    pub fn into_vector(self)->DeviceVector<'a> {match self {Self::Bf16(v)=>v.into_vector(),Self::F32(v)=>v}}
}
impl<'a> Backend<'a> {
    /// Resident adapters allocate private scratch and complete before return.
    /// Check this before allocation or submission on their ordered stream.
    pub fn require_resident_stream(&self) -> Result<(),String> {
        if self.context.capturing.get() {return Err("resident operands require an uncaptured stream".into());}
        if let Some(error)=self.fatal_error() {return Err(error);} Ok(())
    }
    /// Preserve declared parameter precision. The checkpoint binder must supply
    /// its load policy separately. Shape validation precedes device allocation.
    pub fn resident_norm(&self,shape: NormShape,weight: &[f32]) -> Result<ResidentNorm<'_,'a>,String> {
        self.require_resident_stream()?;shape.check_weights(weight)?;
        self.require_layer_kernels()?;
        let weight=self.upload_vector(weight)?;self.synchronize()?;
        Ok(ResidentNorm {backend:self,shape,weight})
    }
}
impl<'a> ResidentNorm<'_,'a> {
    pub fn shape(&self) -> NormShape {self.shape}
    /// One causal token row. Normalize and round before multiplying the weight.
    /// The output precision follows the explicit parameter operand precision.
    pub fn apply(&self,input: &Bf16Vector<'_>) -> Result<NormOutput<'a>,String> {
        let b=self.backend;let s=self.shape;let h=s.width();b.require_resident_stream()?;
        b.check_vector(input.vector(),h)?;
        if input.vector().is_segmented() {return Err("resident norm requires a contiguous input".into());}
        let mut normalized=b.vector(h)?;let mut partials=b.vector(h.div_ceil(128))?;
        // An owned queue records the same three kernels with the same arguments
        // and grids, so a standalone norm does not end the recorded piece.
        b.check_vector(&self.weight,h)?;
        let rms=crate::kernel_params::ModelRms {input:input.vector().buffer.address,weight:0,partials:partials.buffer.address,output:normalized.buffer.address,
            width:u32::try_from(h).map_err(|_|"norm width exceeds ABI")?,rows:1,epsilon:s.epsilon()};
        let scale=crate::kernel_params::ModelElementwise {input:normalized.buffer.address,peer:self.weight.buffer.address,output:normalized.buffer.address,elements:rms.width,operation:4,scalar:1.0};
        if b.record_resident_norm(rms,scale)? {b.resident_completion()?;} else {
        let launched=b.model_rms_device(input.vector(),None,&mut normalized,&mut partials,h,1,s.epsilon());
        let completed=b.resident_completion();launched.and(completed)?;
        let launched=b.model_elementwise_device(&normalized,Some(&self.weight),&normalized,h,4,1.0);
        let completed=b.resident_completion();launched.and(completed)?;
        }
        match s.weight_type() {
            WeightType::Bf16=>Ok(NormOutput::Bf16(b.round_bf16_lanes(&normalized)?)),
            WeightType::F32=>Ok(NormOutput::F32(normalized)),
        }
    }
    /// Many independent token rows. Each row keeps the one-row RMS reduction
    /// tree, normalized BF16 boundary, weight product and output boundary.
    pub fn apply_rows(&self,input: &Bf16Vector<'_>,rows: usize) -> Result<DeviceVector<'a>,String> {
        let b=self.backend;let s=self.shape;let h=s.width();b.require_resident_stream()?;
        b.check_vector(input.vector(),h.checked_mul(rows).ok_or("norm rows overflow")?)?;
        if input.vector().is_segmented() {return Err("resident norm requires a contiguous input".into());}
        let mut normalized=b.vector(h*rows)?;let mut partials=b.vector(h.div_ceil(128)*rows)?;
        b.model_rms_device(input.vector(),None,&mut normalized,&mut partials,h,rows,s.epsilon())?;
        let mut shape=crate::prompt_ops::Shape::new(crate::prompt_ops::Op::NormApply,rows).ok_or("norm rows overflow")?;
        shape.dim=u32::try_from(h).map_err(|_|"norm width exceeds ABI")?;shape.cols=u32::from(s.weight_type()==WeightType::Bf16);
        b.prompt_operation(shape,[Some(&normalized),None,None,None,None],Some(&self.weight))
    }
}
