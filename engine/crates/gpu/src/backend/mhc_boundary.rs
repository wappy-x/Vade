//! Owned post/pre boundaries on one ordered stream.
use super::*;
use crate::{mhc_params::{Shape,Boundary,SHADOW_MAX_BLOCKS},norm_boundary::{NormShape,WeightType}};

/// CHAIN-SHADOW: one static, resident weight span that a boundary's tail blocks
/// pull toward L2. Built only from a checked resident owner of this backend; the
/// borrow ties the span to that owner for the call that launches it, and a
/// dropped device buffer waits for ordered completion before its memory leaves.
#[derive(Clone,Copy,Debug)]
pub struct ShadowRange<'v> {address:u64,bytes:u64,context:*const Context,_owner:PhantomData<&'v ()>}
impl ShadowRange<'_> {
    pub fn bytes(&self)->u64 {self.bytes}
}
/// Which span each chain launch prefetches, with how many tail blocks and which
/// L2 policy. `None` keeps that launch's grid exactly as before.
#[derive(Clone,Copy,Debug)]
pub struct Shadow<'v> {pub rms:Option<ShadowRange<'v>>,pub dots:Option<ShadowRange<'v>>,pub finish:Option<ShadowRange<'v>>,pub blocks:u32,pub evict_last:bool}
impl<'v> ShadowRange<'v> {
    /// Only resident owners of this module tree build a span, from their own checked extent.
    pub(super) fn checked(address:u64,owner_bytes:usize,offset:usize,bytes:usize,context:&Context)->Result<Self,String> {
        let end=offset.checked_add(bytes).ok_or("shadow range overflow")?;
        if bytes==0 || end>owner_bytes {return Err("shadow range lies outside its weight owner".into());}
        Ok(Self {address:address+offset as u64,bytes:bytes as u64,context:context as *const Context,_owner:PhantomData})
    }
}
impl<'a> Backend<'a> {
    /// The byte span `[offset, offset+bytes)` of a contiguous FP32 weight owner.
    pub fn shadow_vector_range<'v>(&self,weight:&'v DeviceVector<'_>,offset:usize,bytes:usize)->Result<ShadowRange<'v>,String> {
        if weight.is_segmented() {return Err("shadow range needs one contiguous weight".into());}
        self.check_vector(weight,weight.len())?;
        ShadowRange::checked(weight.device_address(),weight.len()*4,offset,bytes,self.context)
    }
    /// Time `adds` dependent FP32 adds in one thread: (result bits, SM cycles,
    /// global-timer ns, GPU event ns around the one-warp launch). Bench builds only:
    /// no production path can reach the probe kernel.
    #[cfg(feature="kernel-bench")]
    pub fn chain_add_latency(&self,start:f32,step:f32,adds:u32)->Result<(u32,u64,u64,u128),String> {
        self.mhc_ready()?;self.require_arm_kernels(&["chain_add_latency"])?;
        if adds>crate::mhc_params::PROBE_MAX_ADDS {return Err("add latency probe count exceeds PROBE_MAX_ADDS".into());}
        let output=self.vector(6)?;self.mhc_planes(&[&output],&[6],1)?;
        let mut args=crate::mhc_params::AddLatency {output:output.device_address(),start,step,adds,reserved:0};
        let shape=Shape {rows:1,streams:1,hidden:1,iterations:1,rms_eps:1.0,eps:1.0};
        self.synchronize()?;let begin=Event::record(&self.stream)?;
        let launched=self.mhc_launch("chain_add_latency",shape,0,&mut args);
        let end=Event::record(&self.stream)?;launched?;self.synchronize()?;let event_ns=begin.elapsed_ns(&end)?;
        let words=self.read_vector_values(&output)?;let word=|i:usize|u64::from(words[2*i].to_bits())|(u64::from(words[2*i+1].to_bits())<<32);
        Ok((word(0) as u32,word(1),word(2),event_ns))
    }
}
/// A GPU event on the ordered stream. Two marks time the work queued between
/// them once a synchronize has completed both.
pub struct GpuMark<'a>(Event<'a>);
impl GpuMark<'_> {
    pub fn ns_until(&self,later:&Self)->Result<u128,String> {self.0.elapsed_ns(&later.0)}
}
impl<'a> Backend<'a> {
    /// Record a mark after every launch queued so far, including recorded ones.
    pub fn gpu_mark(&self)->Result<GpuMark<'a>,String> {self.flush_resident_launches()?;Ok(GpuMark(Event::record(&self.stream)?))}
}
fn shadow_args(args:&mut Boundary,range:Option<ShadowRange<'_>>,evict_last:bool) {
    let (address,bytes)=range.map_or((0,0),|range|(range.address,range.bytes));
    args.prefetch=address;args.prefetch_bytes=bytes;args.prefetch_policy=u32::from(evict_last);
}

struct Parameters<'a> {weights:DeviceVector<'a>,base:DeviceVector<'a>,scale:DeviceVector<'a>,norm:DeviceVector<'a>}
pub struct ResidentMhcBoundary<'b,'a> {backend:&'b Backend<'a>,shape:Shape,norm:NormShape,parameters:Rc<Parameters<'a>>}
/// All inputs, parameter planes and prior passes survive queued execution.
pub struct MhcBoundaryPass<'b,'a> {
    backend:&'b Backend<'a>,shape:Shape,parameters:Rc<Parameters<'a>>,
    source:Rc<DeviceVector<'a>>,residual:Rc<DeviceVector<'a>>,mixing:DeviceVector<'a>,
    coefficients:DeviceVector<'a>,collapsed:DeviceVector<'a>,partials:DeviceVector<'a>,output:Rc<DeviceVector<'a>>,
    previous:Option<Box<MhcBoundaryPass<'b,'a>>>,
}
impl<'a> Backend<'a> {
    pub fn resident_mhc_boundary(&self,shape:Shape,norm:NormShape,weights:&[f32],base:&[f32],scale:&[f32],norm_weight:&[f32]) -> Result<ResidentMhcBoundary<'_,'a>,String> {
        self.mhc_ready()?;shape.validate()?;norm.check_weights(norm_weight)?;
        if norm.width()!=shape.hidden as usize || weights.len()!=shape.coefficients()*shape.wide()
            || base.len()!=shape.coefficients() || scale.len()!=3 || weights.iter().chain(base).chain(scale).any(|value| !value.is_finite()) {
            return Err("mHC boundary parameter geometry or value differs".into());
        }
        self.require_arm_kernels(&["mhc_boundary_project","mhc_boundary_rms","mhc_boundary_dots","mhc_boundary_finish","mhc_bf16_combine","mhc_project_tile","mhc_residual_rows"])?;
        let parameters=Rc::new(Parameters {weights:self.upload_vector(weights)?,base:self.upload_vector(base)?,
            scale:self.upload_vector(scale)?,norm:self.upload_vector(norm_weight)?});self.synchronize()?;
        Ok(ResidentMhcBoundary {backend:self,shape,norm,parameters})
    }
}
impl<'b,'a> ResidentMhcBoundary<'b,'a> {
    pub fn shape(&self)->Shape {self.shape}
    fn with_rows(&self,rows:usize)->Result<Self,String> {
        let shape=Shape {rows:u32::try_from(rows).map_err(|_|"mHC boundary row count exceeds u32")?,..self.shape};shape.validate()?;
        Ok(Self {backend:self.backend,shape,norm:self.norm,parameters:self.parameters.clone()})
    }
    pub fn begin_rows(&self,input:Rc<DeviceVector<'a>>,rows:usize)->Result<MhcBoundaryPass<'b,'a>,String> {self.with_rows(rows)?.begin(input)}
    pub fn begin_after_rows(&self,previous:MhcBoundaryPass<'b,'a>,block:Rc<DeviceVector<'a>>,residual:Rc<DeviceVector<'a>>,rows:usize)->Result<MhcBoundaryPass<'b,'a>,String> {
        self.with_rows(rows)?.begin_after(previous,block,residual)
    }
    pub fn begin(&self,input:Rc<DeviceVector<'a>>)->Result<MhcBoundaryPass<'b,'a>,String> {
        self.backend.mhc_ready()?;self.backend.check_vector(&input,self.shape.rows as usize*self.shape.wide())?;
        let residual=Rc::new(self.backend.vector(input.len())?);self.submit(input,residual,None,None)
    }
    pub fn begin_after(&self,previous:MhcBoundaryPass<'b,'a>,block:Rc<DeviceVector<'a>>,residual:Rc<DeviceVector<'a>>)->Result<MhcBoundaryPass<'b,'a>,String> {
        self.begin_after_shadowed(previous,block,residual,None)
    }
    /// `begin` with the next stage's weights pulled toward L2 while the chain runs.
    pub fn begin_shadowed(&self,input:Rc<DeviceVector<'a>>,shadow:Option<&Shadow<'_>>)->Result<MhcBoundaryPass<'b,'a>,String> {
        self.backend.mhc_ready()?;self.backend.check_vector(&input,self.shape.rows as usize*self.shape.wide())?;
        let residual=Rc::new(self.backend.vector(input.len())?);self.submit(input,residual,None,shadow)
    }
    /// `begin_after` with the next stage's weights pulled toward L2 while the chain runs.
    /// The arithmetic launches, their grids for arithmetic blocks, and every plane are unchanged.
    pub fn begin_after_shadowed(&self,previous:MhcBoundaryPass<'b,'a>,block:Rc<DeviceVector<'a>>,residual:Rc<DeviceVector<'a>>,shadow:Option<&Shadow<'_>>)->Result<MhcBoundaryPass<'b,'a>,String> {
        self.backend.mhc_ready()?;let shape=self.shape;
        if !ptr::eq(self.backend,previous.backend) || (shape.rows,shape.streams,shape.hidden)!=(previous.shape.rows,previous.shape.streams,previous.shape.hidden) {
            return Err("deferred mHC boundary context or shape differs".into());
        }
        self.backend.check_vector(&block,shape.rows as usize*shape.hidden as usize)?;
        self.backend.check_vector(&residual,shape.rows as usize*shape.wide())?;
        // This launch reads only the last pass. Older passes would stay live to the first real
        // consumer, about 21 GB across a 1024-row group. Queued drops wait for their launches.
        let mut previous=previous;previous.previous=None;
        self.submit(block,residual,Some(Box::new(previous)),shadow)
    }
    fn submit(&self,source:Rc<DeviceVector<'a>>,residual:Rc<DeviceVector<'a>>,previous:Option<Box<MhcBoundaryPass<'b,'a>>>,shadow:Option<&Shadow<'_>>)->Result<MhcBoundaryPass<'b,'a>,String> {
        let backend=self.backend;
        if let Some(shadow)=shadow {
            if shadow.blocks==0 || shadow.blocks>SHADOW_MAX_BLOCKS {return Err("shadow tail block count is outside 1..=64".into());}
            for range in [shadow.rms,shadow.dots,shadow.finish].into_iter().flatten() {
                if !ptr::eq(range.context,backend.context) {return Err("shadow range belongs to another context".into());}
            }
        }let shape=self.shape;let rows=shape.rows as usize;let wide=shape.wide();let count=shape.coefficients();let hidden=shape.hidden as usize;
        let pass=MhcBoundaryPass {backend,shape,parameters:self.parameters.clone(),source,residual,mixing:backend.vector(rows*wide)?,
            coefficients:backend.vector(rows*count)?,collapsed:backend.vector(rows*hidden)?,partials:backend.vector(rows*hidden.div_ceil(128))?,
            output:Rc::new(backend.vector(rows*hidden)?.into_resident_shared()?),previous};
        let mut vectors=vec![&*pass.source,&pass.parameters.weights,&pass.parameters.base,&pass.parameters.scale,&*pass.residual,&pass.mixing,
            &pass.coefficients,&pass.collapsed,&pass.parameters.norm,&pass.partials,&*pass.output];
        let mut lengths=vec![rows*if pass.previous.is_some() {hidden} else {wide},count*wide,count,3,rows*wide,rows*wide,rows*count,rows*hidden,hidden,rows*hidden.div_ceil(128),rows*hidden];
        if let Some(previous)=&pass.previous {vectors.extend([&*previous.residual,&previous.coefficients]);lengths.extend([rows*wide,rows*count]);}
        backend.mhc_planes(&vectors,&lengths,4)?;
        let mut args=Boundary {input:pass.source.device_address(),previous_input:pass.previous.as_ref().map_or(0,|previous|previous.residual.device_address()),
            previous_coefficients:pass.previous.as_ref().map_or(0,|previous|previous.coefficients.device_address()),residual:pass.residual.device_address(),
            weights:pass.parameters.weights.device_address(),base:pass.parameters.base.device_address(),scale:pass.parameters.scale.device_address(),
            mixing_normalized:pass.mixing.device_address(),coefficients:pass.coefficients.device_address(),collapsed:pass.collapsed.device_address(),
            norm_weight:pass.parameters.norm.device_address(),norm_partials:pass.partials.device_address(),output:pass.output.device_address(),shape,
            norm_epsilon:self.norm.epsilon(),norm_bf16:u32::from(self.norm.weight_type()==WeightType::Bf16),
            prefetch:0,prefetch_bytes:0,prefetch_policy:0,prefetch_reserved:0};
        // The split chains use separate SMs for independent coefficient rows.
        // Larger row groups keep the one-block project schedule; both preserve
        // the same ascending FP32 chains and use the same existing planes.
        // CHAIN-SHADOW applies to the split chains only (decode rows). Each tail
        // block prefetches; arithmetic blocks keep their indices and arguments.
        let tail=|range:Option<ShadowRange<'_>>|shadow.filter(|_|range.is_some()).map_or(0,|shadow|shadow.blocks);
        let evict_last=shadow.is_some_and(|shadow|shadow.evict_last);
        if shape.split_project() {
            let range=shadow.and_then(|shadow|shadow.rms);shadow_args(&mut args,range,evict_last);
            backend.mhc_launch("mhc_boundary_rms",shape,tail(range),&mut args)?;
            let range=shadow.and_then(|shadow|shadow.dots);shadow_args(&mut args,range,evict_last);
            backend.mhc_launch("mhc_boundary_dots",shape,tail(range),&mut args)?;
            let range=shadow.and_then(|shadow|shadow.finish);shadow_args(&mut args,range,evict_last);
            backend.mhc_launch("mhc_boundary_finish",shape,tail(range),&mut args)?;
            return Ok(pass);
        } else if crate::prompt_admission::mhc_tile_admitted(shape) {
            // Nine and more rows: eight tokens share one weight stream and one block (same chains, same planes).
            backend.mhc_launch("mhc_residual_rows",shape,0,&mut args)?;
            backend.mhc_launch("mhc_project_tile",shape,0,&mut args)?;
        } else {backend.mhc_launch("mhc_boundary_project",shape,0,&mut args)?;}
        backend.mhc_launch("mhc_boundary_finish",shape,0,&mut args)?;
        Ok(pass)
    }
}
impl<'a> MhcBoundaryPass<'_,'a> {
    pub fn previous(&self)->Option<&Self> {self.previous.as_deref()}
    pub fn shape(&self)->Shape {self.shape}
    pub fn residual(&self)->&DeviceVector<'a> {&self.residual}
    pub fn mixing_normalized(&self)->&DeviceVector<'a> {&self.mixing}
    pub fn coefficients(&self)->&DeviceVector<'a> {&self.coefficients}
    pub fn collapsed(&self)->&DeviceVector<'a> {&self.collapsed}
    pub fn normalized(&self)->Rc<DeviceVector<'a>> {self.output.clone()}
    /// A real residual consumer ends deferral. Preserve the original block cast
    /// and combine arithmetic, including completion before owners are dropped.
    pub fn materialize(self,block:&DeviceVector<'_>,output:&DeviceVector<'_>)->Result<(),String> {
        let backend=self.backend;let shape=self.shape;backend.mhc_ready()?;
        backend.check_vector(block,shape.rows as usize*shape.hidden as usize)?;
        backend.check_vector(output,shape.rows as usize*shape.wide())?;
        let block=backend.round_bf16_lanes(block)?;
        backend.mhc_combine_owned(shape,[&self.residual,&self.coefficients,block.vector(),output])
    }
}
