//! Owned mHC binding. Preparation stays queued on the backend's ordered stream.
//! This binds candidate primitive arithmetic. It does not admit a checkpoint.
use super::*;
use crate::{mhc_params::{Shape,Prepare,Combine},mhc_boundary::{Span,check_spans,check_bf16}};

/// Logical BF16 values in four-byte FP32 lanes. This is not packed BF16 storage.
/// The private owner prevents an unrounded FP32 vector from claiming this type.
pub struct Bf16Vector<'a> { vector: DeviceVector<'a> }
impl<'a> Bf16Vector<'a> {
    pub fn vector(&self) -> &DeviceVector<'_> { &self.vector }
    /// Transfer the completed rounded storage without a device copy.
    pub fn into_vector(self)->DeviceVector<'a> {self.vector}
    pub fn len(&self) -> usize { self.vector.len() }
    pub fn is_empty(&self) -> bool { self.vector.is_empty() }
}
/// Losslessly widened checkpoint parameters. The vectors are private and cannot
/// be changed between preparation and consumption by another safe caller.
pub struct ResidentMhc<'b,'a> {
    backend: &'b Backend<'a>, shape: Shape,
    weights: Rc<DeviceVector<'a>>, base: Rc<DeviceVector<'a>>, scale: Rc<DeviceVector<'a>>,
}
/// The input, normalized scratch, coefficients and collapsed output stay owned
/// through the block. Immutable parameter owners also survive queued work.
/// Coefficients cannot be paired with another residual row. Accessors return
/// device storage, not host-ready data; consumers use the same ordered stream.
pub struct MhcPass<'b,'a> {
    backend: &'b Backend<'a>, shape: Shape, input: Bf16Vector<'a>,
    normalized: DeviceVector<'a>, coefficients: DeviceVector<'a>, collapsed: Bf16Vector<'a>,
    _parameters: [Rc<DeviceVector<'a>>;3],
}
impl<'a> Backend<'a> {
    pub(super) fn mhc_ready(&self) -> Result<(),String> {
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
        // remain live through completion: queue drops defer allocations until
        // the real drain, and ordinary callers wait. Arguments are copied at launch.
        let launched=unsafe {self.launch_flash("bf16_lanes_round",words.div_ceil(128),128,&mut [parameter(&mut args)])};
        let completed=self.resident_completion();launched.and(completed)?;Ok(output)
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
        self.require_arm_kernels(&["mhc_prepare","mhc_bf16_combine"])?;
        let weights=Rc::new(self.upload_vector(weights)?);let base=Rc::new(self.upload_vector(base)?);let scale=Rc::new(self.upload_vector(scale)?);
        self.synchronize()?;
        Ok(ResidentMhc {backend:self,shape,weights,base,scale})
    }
    pub(super) fn mhc_planes(&self, vectors: &[&DeviceVector<'_>], lengths: &[usize], writable: usize) -> Result<(),String> {
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
    fn mhc_prepare_owned(&self, shape: Shape, vectors: [&DeviceVector<'_>;7]) -> Result<(),String> {
        shape.validate()?;let t=shape.rows as usize;let w=shape.wide();let p=shape.coefficients();
        self.mhc_planes(&vectors,&[t*w,p*w,p,3,t*w,t*p,t*shape.hidden as usize],4)?;
        let mut args=Prepare {input:vectors[0].device_address(),weights:vectors[1].device_address(),
            base:vectors[2].device_address(),scale:vectors[3].device_address(),normalized:vectors[4].device_address(),
            coefficients:vectors[5].device_address(),collapsed:vectors[6].device_address(),shape};
        self.mhc_launch("mhc_prepare",shape,0,&mut args)
    }
    /// `tail` extra blocks follow the arithmetic grid of the three CHAIN-SHADOW
    /// entries; every other entry refuses a tail.
    pub(super) fn mhc_launch<T>(&self,name:&str,shape:Shape,tail:u32,args:&mut T) -> Result<(),String> {
        if tail>crate::mhc_params::SHADOW_MAX_BLOCKS || (tail!=0 && !["mhc_boundary_rms","mhc_boundary_dots","mhc_boundary_finish"].contains(&name)) {
            return Err("mHC launch tail is not admitted for this entry".into());
        }
        let (blocks,threads)=match name {
            "mhc_boundary_rms"|"chain_shadow_prefetch"=>(shape.rows,256),
            #[cfg(feature="kernel-bench")]
            "chain_add_latency"=>(1,32),
            "mhc_residual_rows"=>((shape.rows as usize*shape.wide()).div_ceil(256) as u32,256),
            "mhc_project_tile"=>(crate::prompt_admission::mhc_tile_blocks(shape.rows),((crate::prompt_admission::MHC_TOKENS+crate::prompt_admission::MHC_PRODUCER_WARPS)*32) as u32),
            "mhc_boundary_dots"=>(shape.rows.checked_mul(u32::try_from(shape.coefficients()).map_err(|_|"mHC coefficient grid exceeds u32")?).ok_or("mHC coefficient grid exceeds u32")?,256),
            _=>(shape.rows,shape.prepare_threads()),
        };
        let blocks=blocks.checked_add(tail).ok_or("mHC grid with tail exceeds u32")?;
        crate::prompt_launches::record(name);
        // SAFETY: Only the checked mHC binders call this private helper. They
        // select the matching by-value ABI, validate every extent/context/alias,
        // and retain all owners through the ordered consumer. In the one-block
        // launches each block owns one token. In the split launches an RMS block
        // owns one token; a dots block owns one (token, coefficient) word and a
        // disjoint interval of the normalized plane. Warps are complete and
        // block barriers are uniform. CHAIN-SHADOW tail blocks (rms, dots and
        // finish only) and every chain_shadow_prefetch block write nothing,
        // reach no barrier, and prefetch only a ShadowRange built from a live
        // resident weight owner of this context. The
        // `chain_add_latency` probe is one warp whose binder passes a private,
        // checked 6-word plane and synchronizes before reading it. Launch
        // copies the live argument; error/drop cleanup drains referenced work.
        unsafe {self.launch_flash(name,blocks,threads,&mut [parameter(args)])}
    }
    pub(super) fn mhc_combine_owned(&self, shape: Shape, vectors: [&DeviceVector<'_>;4]) -> Result<(),String> {
        shape.validate()?;let t=shape.rows as usize;let w=shape.wide();let h=shape.hidden as usize;
        self.mhc_planes(&vectors,&[t*w,t*shape.coefficients(),t*h,t*w],3)?;
        let mut args=Combine {input:vectors[0].device_address(),coefficients:vectors[1].device_address(),
            block:vectors[2].device_address(),output:vectors[3].device_address(),rows:shape.rows,streams:shape.streams,hidden:shape.hidden};
        // SAFETY: Shape and all FP32 spans are checked and disjoint. Each owner
        // and the by-value launch argument remain live through stream completion.
        // The output uses private contiguous storage with the exact kernel extent.
        let launched=unsafe {self.launch_flash("mhc_bf16_combine",(t*w).div_ceil(128) as u32,128,&mut [parameter(&mut args)])};
        let completed=self.resident_completion();launched.and(completed)
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
        let parameters=[self.weights.clone(),self.base.clone(),self.scale.clone()];
        b.mhc_prepare_owned(s,[&input.vector,&self.weights,&self.base,&self.scale,&normalized,&coefficients,&collapsed.vector])?;
        Ok(MhcPass {backend:b,shape:s,input,normalized,coefficients,collapsed,_parameters:parameters})
    }
}
impl<'b,'a> ResidentMhc<'b,'a> {
    /// Independent token rows `[rows,N,D]`. Each token keeps its own RMS fold,
    /// coefficient dots and Sinkhorn iterations. Only the launch shape changes.
    pub fn begin_rows(&self, input: Bf16Vector<'a>, rows: usize) -> Result<MhcPass<'b,'a>,String> {
        let b=self.backend;let s=Shape {rows:u32::try_from(rows).map_err(|_|"mHC rows exceed the ABI")?,..self.shape};
        b.mhc_ready()?;s.validate()?;
        b.check_vector(&input.vector,rows*s.wide())?;
        let normalized=b.vector(rows*s.wide())?;let coefficients=b.vector(rows*s.coefficients())?;
        let collapsed=Bf16Vector {vector:b.vector(rows*s.hidden as usize)?};
        let parameters=[self.weights.clone(),self.base.clone(),self.scale.clone()];
        b.mhc_prepare_owned(s,[&input.vector,&self.weights,&self.base,&self.scale,&normalized,&coefficients,&collapsed.vector])?;
        Ok(MhcPass {backend:b,shape:s,input,normalized,coefficients,collapsed,_parameters:parameters})
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
