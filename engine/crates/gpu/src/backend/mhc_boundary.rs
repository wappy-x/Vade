//! Owned post/pre boundaries on one ordered stream.
use super::*;
use crate::{mhc_host::HostBoundary,mhc_host_service::{Kind,Queue},mhc_params::{Shape,Boundary,SHADOW_MAX_BLOCKS},norm_boundary::{NormShape,WeightType}};
use std::sync::Arc;

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
    pub fn chain_shadow_prefetch(&self,shadow:&Shadow<'_>)->Result<(),String> {
        if shadow.blocks==0 || shadow.blocks>SHADOW_MAX_BLOCKS {return Err("shadow prefetch block count is outside 1..=64".into());}
        let ranges=[shadow.rms,shadow.dots,shadow.finish];
        for range in ranges.into_iter().flatten() {
            if !ptr::eq(range.context,self.context) {return Err("shadow range belongs to another context".into());}
        }
        if ranges.iter().all(Option::is_none) {return Ok(());}
        self.mhc_ready()?;self.require_arm_kernels(&["chain_shadow_prefetch"])?;
        let shape=Shape {rows:shadow.blocks,streams:1,hidden:1,iterations:1,rms_eps:1.0,eps:1.0};
        for range in ranges.into_iter().flatten() {
            let mut args=crate::mhc_params::ShadowPrefetch {prefetch:range.address,prefetch_bytes:range.bytes,prefetch_policy:u32::from(shadow.evict_last),reserved:0};
            self.mhc_launch("chain_shadow_prefetch",shape,0,&mut args)?;
        }
        Ok(())
    }
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
/// The host lane of one-row boundaries: a local mailbox, the job queue of its service thread
/// (crate::mhc_host_service) and the cores that fold coefficient rows. The caller keeps the
/// mailbox and the running service alive while any boundary made with this link can launch.
#[derive(Clone)]
pub struct MhcHostLink<'b,'a> {mailbox:&'b TpMailbox<'a>,queue:Arc<Queue>,cores:usize}
impl<'b,'a> MhcHostLink<'b,'a> {
    pub fn new(mailbox:&'b TpMailbox<'a>,queue:Arc<Queue>,cores:usize)->Self {Self {mailbox,queue,cores}}
    pub fn queue(&self)->&Arc<Queue> {&self.queue}
    pub fn require_success(&self)->Result<(),String> {
        self.mailbox.require_success()?;if self.queue.failed() {return Err("host mHC service failed".into());}Ok(())
    }
}
/// The longest a take waits for the host lane (ns). The service answers in tens of microseconds.
const HOST_TAKE_LIMIT_NS:u64=10_000_000_000;
/// A hosted pass: its residual and coefficients live in the service under `take` until the pass drops.
struct Hosted<'b,'a> {link:MhcHostLink<'b,'a>,boundary:Arc<HostBoundary>,take:u64}
impl Drop for Hosted<'_,'_> {fn drop(&mut self) {self.link.queue.release(self.take);}}
pub struct ResidentMhcBoundary<'b,'a> {backend:&'b Backend<'a>,shape:Shape,norm:NormShape,parameters:Rc<Parameters<'a>>,host:Option<(MhcHostLink<'b,'a>,Arc<HostBoundary>)>}
/// All inputs, parameter planes and prior passes survive queued execution.
pub struct MhcBoundaryPass<'b,'a> {
    backend:&'b Backend<'a>,shape:Shape,parameters:Rc<Parameters<'a>>,
    source:Rc<DeviceVector<'a>>,residual:Rc<DeviceVector<'a>>,mixing:DeviceVector<'a>,
    coefficients:DeviceVector<'a>,collapsed:DeviceVector<'a>,partials:DeviceVector<'a>,output:Rc<DeviceVector<'a>>,
    previous:Option<Box<MhcBoundaryPass<'b,'a>>>,
    /// Some: the host lane computed this pass. Only `output` is on the device until `deliver`.
    hosted:Option<Hosted<'b,'a>>,
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
        Ok(ResidentMhcBoundary {backend:self,shape,norm,parameters,host:None})
    }
    /// The same boundary with the host lane for its one-row passes: the same planes, bit for bit
    /// (crate::mhc_host), computed by CPU cores between a put and a take on the local mailbox.
    #[allow(clippy::too_many_arguments)]
    pub fn resident_mhc_boundary_hosted<'b>(&'b self,shape:Shape,norm:NormShape,weights:&[f32],base:&[f32],scale:&[f32],norm_weight:&[f32],link:&MhcHostLink<'b,'a>) -> Result<ResidentMhcBoundary<'b,'a>,String> {
        let mut boundary=self.resident_mhc_boundary(shape,norm,weights,base,scale,norm_weight)?;
        let layout=link.mailbox.layout();let local=crate::tp_mailbox_host::HEADER+2*crate::tp_mailbox_host::TP_MAILBOX_SLOTS*layout.slot_words;
        if link.mailbox.slot_bytes()<shape.wide()*4 || layout.frame!=0 || layout.words()!=local {return Err("mHC host mailbox is smaller than one residual plane or has a routed layout".into());}
        let host=HostBoundary::new(Shape {rows:1,..shape},norm,weights,base,scale,norm_weight,link.cores)?;
        boundary.host=Some((link.clone(),Arc::new(host)));Ok(boundary)
    }
}
impl<'b,'a> ResidentMhcBoundary<'b,'a> {
    #[cfg(feature="kernel-bench")]
    pub fn finish_host_planes(&self,residual:DeviceVector<'a>,coefficients:DeviceVector<'a>,shadow:Option<&Shadow<'_>>)->Result<MhcBoundaryPass<'b,'a>,String> {
        let backend=self.backend;backend.mhc_ready()?;
        let shape=self.shape;let rows=shape.rows as usize;let wide=shape.wide();let count=shape.coefficients();let hidden=shape.hidden as usize;
        if let Some(shadow)=shadow {
            if shadow.blocks==0 || shadow.blocks>SHADOW_MAX_BLOCKS {return Err("shadow tail block count is outside 1..=64".into());}
            if shadow.finish.is_some_and(|range|!ptr::eq(range.context,backend.context)) {return Err("shadow range belongs to another context".into());}
        }
        let residual=Rc::new(residual);
        let pass=MhcBoundaryPass {backend,shape,parameters:self.parameters.clone(),source:residual.clone(),residual,mixing:backend.vector(rows*wide)?,
            coefficients,collapsed:backend.vector(rows*hidden)?,partials:backend.vector(rows*hidden.div_ceil(128))?,
            output:Rc::new(backend.vector(rows*hidden)?.into_resident_shared()?),previous:None,hosted:None};
        backend.mhc_planes(&[&pass.parameters.norm,&pass.residual,&pass.coefficients,&pass.collapsed,&pass.partials,&pass.output],
            &[hidden,rows*wide,rows*count,rows*hidden,rows*hidden.div_ceil(128),rows*hidden],2)?;
        let mut args=Boundary {input:pass.residual.device_address(),previous_input:0,previous_coefficients:0,residual:pass.residual.device_address(),
            weights:pass.parameters.weights.device_address(),base:pass.parameters.base.device_address(),scale:pass.parameters.scale.device_address(),
            mixing_normalized:pass.mixing.device_address(),coefficients:pass.coefficients.device_address(),collapsed:pass.collapsed.device_address(),
            norm_weight:pass.parameters.norm.device_address(),norm_partials:pass.partials.device_address(),output:pass.output.device_address(),shape,
            norm_epsilon:self.norm.epsilon(),norm_bf16:u32::from(self.norm.weight_type()==WeightType::Bf16),
            prefetch:0,prefetch_bytes:0,prefetch_policy:0,prefetch_reserved:0};
        let range=shadow.and_then(|shadow|shadow.finish);shadow_args(&mut args,range,shadow.is_some_and(|shadow|shadow.evict_last));
        backend.mhc_launch("mhc_boundary_finish",shape,shadow.filter(|_|range.is_some()).map_or(0,|shadow|shadow.blocks),&mut args)?;
        Ok(pass)
    }
    pub fn shape(&self)->Shape {self.shape}
    fn with_rows(&self,rows:usize)->Result<Self,String> {
        let shape=Shape {rows:u32::try_from(rows).map_err(|_|"mHC boundary row count exceeds u32")?,..self.shape};shape.validate()?;
        Ok(Self {backend:self.backend,shape,norm:self.norm,parameters:self.parameters.clone(),host:self.host.clone()})
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
            output:Rc::new(backend.vector(rows*hidden)?.into_resident_shared()?),previous,hosted:None};
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
        // Host lane: one-row split boundaries whose last pass is also in the service (or absent).
        // The put carries the source, the service computes every plane, the take brings the output.
        if let Some((link,host))=self.host.as_ref().filter(|_|rows==1 && shape.split_project() && pass.previous.as_ref().is_none_or(|previous|previous.hosted.is_some())) {
            let previous=pass.previous.as_ref().and_then(|previous|previous.hosted.as_ref()).map(|hosted|hosted.take);
            let words=if previous.is_some() {hidden} else {wide};
            let ticket=link.queue.push(host.clone(),Kind::Run {previous})?;
            let pass=MhcBoundaryPass {hosted:Some(Hosted {link:link.clone(),boundary:host.clone(),take:ticket.take}),..pass};
            backend.tp_mailbox_put(link.mailbox,&pass.source,1,words,words,0,ticket.put.ok_or("mHC host run has no put")?)?;
            // CHAIN-SHADOW: the same spans as the kernel chain's tails, from prefetch-only launches
            // that run while the host lane works (the take behind them waits for the answer).
            if let Some(shadow)=shadow {backend.chain_shadow_prefetch(shadow)?;}
            backend.tp_mailbox_take_shared(link.mailbox,&pass.output,1,hidden,hidden,0,0,ticket.take,HOST_TAKE_LIMIT_NS)?;
            return Ok(pass);
        }
        // A kernel pass after a hosted one reads the last residual and coefficients on the device.
        if let Some(previous)=&pass.previous {previous.deliver()?;}
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
    /// The job that computed this pass on the host lane, if it did (the twin gate reads its planes).
    pub fn hosted_take(&self)->Option<u64> {self.hosted.as_ref().map(|hosted|hosted.take)}
    /// Bring a hosted pass's residual and coefficients to their device planes (queued; two takes).
    /// A kernel pass has them already.
    pub fn deliver(&self)->Result<(),String> {
        let Some(hosted)=&self.hosted else {return Ok(());};let (backend,shape)=(self.backend,self.shape);
        let (wide,count)=(shape.wide(),shape.coefficients());
        let ticket=hosted.link.queue.push(hosted.boundary.clone(),Kind::Residual {of:hosted.take})?;
        backend.tp_mailbox_take_shared(hosted.link.mailbox,&self.residual,1,wide,wide,0,0,ticket.take,HOST_TAKE_LIMIT_NS)?;
        let ticket=hosted.link.queue.push(hosted.boundary.clone(),Kind::Coefficients {of:hosted.take})?;
        backend.tp_mailbox_take_shared(hosted.link.mailbox,&self.coefficients,1,count,count,0,0,ticket.take,HOST_TAKE_LIMIT_NS)
    }
    /// A real residual consumer ends deferral. Preserve the original block cast
    /// and combine arithmetic, including completion before owners are dropped.
    pub fn materialize(self,block:&DeviceVector<'_>,output:&DeviceVector<'_>)->Result<(),String> {
        let backend=self.backend;let shape=self.shape;backend.mhc_ready()?;
        backend.check_vector(block,shape.rows as usize*shape.hidden as usize)?;
        backend.check_vector(output,shape.rows as usize*shape.wide())?;
        self.deliver()?;
        let block=backend.round_bf16_lanes(block)?;
        backend.mhc_combine_owned(shape,[&self.residual,&self.coefficients,block.vector(),output])
    }
}
