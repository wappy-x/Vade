//! Owned mHC operand and output binding. Each launch completes before return.
//! This binds candidate primitive arithmetic. It does not admit a checkpoint.
use super::*;
use crate::{mhc_params::{Shape,Prepare,Collapse,Combine},mhc_boundary::{Span,check_spans,check_bf16}};

/// Logical BF16 values in four-byte FP32 lanes. This is not packed BF16 storage.
/// The private owner prevents an unrounded FP32 vector from claiming this type.
pub struct Bf16Vector<'a> { vector: DeviceVector<'a> }
impl Bf16Vector<'_> {
    pub fn vector(&self) -> &DeviceVector<'_> { &self.vector }
    pub fn len(&self) -> usize { self.vector.len() }
    pub fn is_empty(&self) -> bool { self.vector.is_empty() }
}
/// Losslessly widened checkpoint parameters. The vectors are private and cannot
/// be changed between preparation and consumption by another safe caller.
pub struct ResidentMhc<'b,'a> {
    backend: &'b Backend<'a>, shape: Shape,
    weights: DeviceVector<'a>, base: DeviceVector<'a>, scale: DeviceVector<'a>,
}
/// The input, normalized scratch, coefficients and collapsed output stay owned
/// through the block. Coefficients cannot be paired with another residual row.
pub struct MhcPass<'b,'a> {
    backend: &'b Backend<'a>, shape: Shape, input: Bf16Vector<'a>,
    normalized: DeviceVector<'a>, coefficients: DeviceVector<'a>, collapsed: Bf16Vector<'a>,
}
impl<'a> Backend<'a> {
    fn mhc_ready(&self) -> Result<(),String> {
        if self.context.capturing.get() {return Err("resident mHC requires an uncaptured stream".into());}
        if let Some(error)=self.fatal_error() {return Err(error);}
        Ok(())
    }
    pub fn upload_bf16_lanes(&self, values: &[f32]) -> Result<Bf16Vector<'a>,String> {
        self.mhc_ready()?;
        check_bf16(values)?;Ok(Bf16Vector {vector:self.upload_vector(values)?})
    }
    /// Insert an explicit RN-even BF16 boundary without a host read or upload.
    /// Nonfinite results remain BF16 values. This does not check model quality.
    pub fn round_bf16_lanes(&self, input: &DeviceVector<'_>) -> Result<Bf16Vector<'a>,String> {
        self.mhc_ready()?;
        let words=u32::try_from(input.len()).ok().filter(|&n| n>0).ok_or("BF16 lane count exceeds the ABI")?;
        self.check_vector(input,words as usize)?;
        let output=Bf16Vector {vector:self.vector(words as usize)?};
        self.mhc_planes(&[input,&output.vector],&[words as usize,words as usize],1)?;
        let mut args=crate::mhc_boundary::LaneCast {input:input.device_address(),output:output.vector.device_address(),words};
        // SAFETY: Both exact FP32 spans have the backend context and do not
        // overlap. The output is private and contiguous. Both borrowed owners
        // and the argument remain live through launch and stream completion.
        let launched=unsafe {self.launch_flash("bf16_lanes_round",words.div_ceil(128),128,&mut [parameter(&mut args)])};
        let completed=self.synchronize();launched.and(completed)?;Ok(output)
    }
    /// Shape and parameter validation precede allocation. No packed weights or
    /// implicit dtype conversion enter this API. The caller widens stored bits.
    pub fn resident_mhc(&self, shape: Shape, weights: &[f32], base: &[f32], scale: &[f32]) -> Result<ResidentMhc<'_,'a>,String> {
        self.mhc_ready()?;
        shape.validate()?;
        if weights.len()!=shape.coefficients()*shape.wide() || base.len()!=shape.coefficients() || scale.len()!=3 {
            return Err("resident mHC parameter extents differ".into());
        }
        if weights.iter().chain(base).chain(scale).any(|v| !v.is_finite()) {return Err("resident mHC parameter is not finite".into());}
        self.require_arm_kernels(&["mhc_prepare","mhc_bf16_collapse","mhc_bf16_combine"])?;
        let weights=self.upload_vector(weights)?;let base=self.upload_vector(base)?;let scale=self.upload_vector(scale)?;
        self.synchronize()?;
        Ok(ResidentMhc {backend:self,shape,weights,base,scale})
    }
    fn mhc_planes(&self, vectors: &[&DeviceVector<'_>], lengths: &[usize], writable: usize) -> Result<(),String> {
        self.mhc_ready()?;
        if vectors.len()!=lengths.len() || writable>vectors.len() {return Err("resident mHC plane count differs".into());}
        let mut spans=Vec::new();
        for (i,(vector,&length)) in vectors.iter().zip(lengths).enumerate() {
            self.check_vector(vector,length)?;
            if vector.is_segmented() || (i>=writable && !vector.writable_from(0)) {return Err("resident mHC requires private contiguous outputs".into());}
            spans.push(Span {address:vector.device_address(),words:vector.len()});
        }
        check_spans(&spans,lengths).map_err(str::to_string)
    }
    fn mhc_prepare_owned(&self, shape: Shape, vectors: [&DeviceVector<'_>;6]) -> Result<(),String> {
        shape.validate()?;let t=shape.rows as usize;let w=shape.wide();let p=shape.coefficients();
        self.mhc_planes(&vectors,&[t*w,p*w,p,3,t*w,t*p],4)?;
        let mut args=Prepare {input:vectors[0].device_address(),weights:vectors[1].device_address(),
            base:vectors[2].device_address(),scale:vectors[3].device_address(),normalized:vectors[4].device_address(),
            coefficients:vectors[5].device_address(),shape};
        // SAFETY: Shape and every FP32 extent, context, alignment, and alias were
        // checked above. The borrowed owners remain live through synchronization.
        // The argument struct remains live until launch copies its value.
        let launched=unsafe {self.launch_flash("mhc_prepare",shape.rows.div_ceil(128),128,&mut [parameter(&mut args)])};
        let completed=self.synchronize();launched.and(completed)
    }
    fn mhc_collapse_owned(&self, shape: Shape, vectors: [&DeviceVector<'_>;3]) -> Result<(),String> {
        shape.validate()?;let t=shape.rows as usize;let h=shape.hidden as usize;
        self.mhc_planes(&vectors,&[t*shape.wide(),t*shape.coefficients(),t*h],2)?;
        let mut args=Collapse {input:vectors[0].device_address(),coefficients:vectors[1].device_address(),
            output:vectors[2].device_address(),rows:shape.rows,streams:shape.streams,hidden:shape.hidden};
        // SAFETY: Shape and all FP32 spans are checked and disjoint. Each owner
        // and the by-value launch argument remain live through stream completion.
        // The output uses private contiguous storage with the exact kernel extent.
        let launched=unsafe {self.launch_flash("mhc_bf16_collapse",(t*h).div_ceil(128) as u32,128,&mut [parameter(&mut args)])};
        let completed=self.synchronize();launched.and(completed)
    }
    fn mhc_combine_owned(&self, shape: Shape, vectors: [&DeviceVector<'_>;4]) -> Result<(),String> {
        shape.validate()?;let t=shape.rows as usize;let w=shape.wide();let h=shape.hidden as usize;
        self.mhc_planes(&vectors,&[t*w,t*shape.coefficients(),t*h,t*w],3)?;
        let mut args=Combine {input:vectors[0].device_address(),coefficients:vectors[1].device_address(),
            block:vectors[2].device_address(),output:vectors[3].device_address(),rows:shape.rows,streams:shape.streams,hidden:shape.hidden};
        // SAFETY: Shape and all FP32 spans are checked and disjoint. Each owner
        // and the by-value launch argument remain live through stream completion.
        // The output uses private contiguous storage with the exact kernel extent.
        let launched=unsafe {self.launch_flash("mhc_bf16_combine",(t*w).div_ceil(128) as u32,128,&mut [parameter(&mut args)])};
        let completed=self.synchronize();launched.and(completed)
    }
}
impl<'b,'a> ResidentMhc<'b,'a> {
    pub fn shape(&self) -> Shape {self.shape}
    pub fn begin(&self, input: Bf16Vector<'a>) -> Result<MhcPass<'b,'a>,String> {
        let b=self.backend;let s=self.shape;let t=s.rows as usize;
        b.mhc_ready()?;
        // Check input context and extent before scratch allocation or launch.
        b.check_vector(&input.vector,t*s.wide())?;
        let normalized=b.vector(t*s.wide())?;let coefficients=b.vector(t*s.coefficients())?;
        let collapsed=Bf16Vector {vector:b.vector(t*s.hidden as usize)?};
        b.mhc_prepare_owned(s,[&input.vector,&self.weights,&self.base,&self.scale,&normalized,&coefficients])?;
        b.mhc_collapse_owned(s,[&input.vector,&coefficients,&collapsed.vector])?;
        Ok(MhcPass {backend:b,shape:s,input,normalized,coefficients,collapsed})
    }
}
impl<'a> MhcPass<'_,'a> {
    pub fn normalized(&self) -> &DeviceVector<'a> {&self.normalized}
    pub fn coefficients(&self) -> &DeviceVector<'a> {&self.coefficients}
    pub fn collapsed(&self) -> &Bf16Vector<'a> {&self.collapsed}
    pub fn finish(self, block: Bf16Vector<'a>) -> Result<Bf16Vector<'a>,String> {
        let b=self.backend;let s=self.shape;let t=s.rows as usize;
        b.mhc_ready()?;
        b.check_vector(&block.vector,t*s.hidden as usize)?;
        let output=Bf16Vector {vector:b.vector(t*s.wide())?};
        b.mhc_combine_owned(s,[&self.input.vector,&self.coefficients,&block.vector,&output.vector])?;
        Ok(output)
    }
}
