//! Owned complete MXFP8 rows. Calls complete before borrowed operands can leave.
use super::*;
use crate::{mxfp8_projection::Shape,norm_boundary::WeightType};

pub struct ResidentMxfp8Projection<'b,'a> {backend:&'b Backend<'a>,shape:Shape,weights:Buffer<'a>,scales:Buffer<'a>}
pub struct Mxfp8ProjectionPass<'a> {accumulator:DeviceVector<'a>,output:Bf16Vector<'a>}
impl<'a> Backend<'a> {
    /// The arena holds E4M3 rows followed by their original UE8M0 scale rows.
    pub fn resident_mxfp8_projection(&self,arena:&WeightArena<'a>,shape:Shape)->Result<ResidentMxfp8Projection<'_,'a>,String> {
        self.require_resident_stream()?;
        if !ptr::eq(arena.buffer._context,self.context) || arena.bytes()!=shape.arena_bytes()? {
            return Err("MXFP8 arena context or complete row extent differs".into());
        }
        self.require_arm_kernels(&["flash_next_mxfp8_m1"])?;
        let weights=Buffer::view(&arena.buffer,0,shape.weight_bytes())?;
        let scales=Buffer::view(&arena.buffer,shape.weight_bytes(),shape.scale_bytes())?;
        Ok(ResidentMxfp8Projection {backend:self,shape,weights,scales})
    }
}
impl<'a> ResidentMxfp8Projection<'_,'a> {
    /// Preserve the completed norm's actual dtype. Do not round FP32 input lanes.
    /// Output rows stay local until the layer's declared gather is complete.
    pub fn apply(&self,input:&NormOutput<'_>)->Result<Mxfp8ProjectionPass<'a>,String> {
        let b=self.backend;let s=self.shape;b.require_resident_stream()?;
        let dtype=match input {NormOutput::Bf16(_)=>WeightType::Bf16,NormOutput::F32(_)=>WeightType::F32};
        if dtype!=s.input() {return Err("MXFP8 norm output dtype differs from the bound producer".into());}
        let input=input.vector();b.check_vector(input,s.cols())?;
        if input.is_segmented() {return Err("MXFP8 projection requires complete contiguous input features".into());}
        let accumulator=b.vector(s.rows())?;
        let mut args=crate::next_params::MxLayer::default();
        args.push(crate::next_params::MxProjection {weights:self.weights.address,scales:self.scales.address,
            input:input.device_address(),output:accumulator.device_address(),rows:s.rows() as u32,
            cols:s.cols() as u32,tokens:1,input_bf16:0,round:0})?;
        // SAFETY: Shape bounds complete E4M3 rows and UE8M0 groups in the retained arena views.
        // check_vector checks the input context and full feature extent. MxLayer checks alignment and ABI limits.
        // The accumulator is a new disjoint allocation. All owners and arguments live through stream completion.
        let launched=unsafe {b.launch_flash("flash_next_mxfp8_m1",args.ends[1],128,&mut [parameter(&mut args)])};
        let completed=b.synchronize();launched.and(completed)?;
        let output=b.round_bf16_lanes(&accumulator)?;
        Ok(Mxfp8ProjectionPass {accumulator,output})
    }
}
impl<'a> Mxfp8ProjectionPass<'a> {
    pub fn accumulator(&self)->&DeviceVector<'a> {&self.accumulator}
    pub fn output(&self)->&Bf16Vector<'a> {&self.output}
}
