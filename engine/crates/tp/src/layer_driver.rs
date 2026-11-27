//! Checked preparation for a deferred layer. This module cannot launch a layer.
//! Header admission, primitive arguments, and source equivalence are separate checks.
use crate::{Result, tensor_layout::GatherLayout};
use engine_model::plan::{TextDescription, admission::{HeaderAdmission, CheckpointAdmission, WeightView, ComponentView, UploadWeight},
    layer::{LayerPlan, Topology, Split, Kind, cast::{Contract, Dtype}},
    state::{StatePlan, ByteLayout}, storage::WeightStorage, tensor_split::{Ownership, ProjectionRows}};
use engine_model::safetensors::Dtype as StoredDtype;
use spark_gpu_probe::mhc_params::{Shape, Prepare};
use engine_fabric::exchange::Exchange;

fn need(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(why.into()) } }
fn model<T>(result: engine_model::Result<T>) -> Result<T> { result.map_err(|e| e.to_string()) }
fn number(n: u64) -> Result<u32> { u32::try_from(n).map_err(|_| "layer scalar exceeds the kernel ABI".into()) }

/// The header owns the physical dtype and component axes. Row splits retain full K.
#[derive(Debug)]
pub struct WeightBinding<'a> { view: &'a WeightView, ownership: Option<Ownership>, rows: Option<ProjectionRows> }
impl WeightBinding<'_> {
    pub fn view(&self) -> &WeightView { self.view }
    pub fn ownership(&self) -> Option<&Ownership> { self.ownership.as_ref() }
    pub fn rows(&self) -> Option<&ProjectionRows> { self.rows.as_ref() }
}
#[derive(Debug)]
pub struct GatherBinding { step: usize, name: String, dtype: Dtype, layout: GatherLayout }
impl GatherBinding {
    pub fn step(&self) -> usize { self.step }
    pub fn name(&self) -> &str { &self.name }
    pub fn dtype(&self) -> Dtype { self.dtype }
    pub fn layout(&self) -> &GatherLayout { &self.layout }
}

/// Prepare operands for the canonical decoder program and its mHC boundaries.
/// This is a header-backed boundary, not executable model admission.
/// Borrows keep the checked headers and state plan alive through argument preparation.
pub struct LayerBoundary<'a> {
    program: LayerPlan, casts: Contract, state: &'a StatePlan,
    weights: Vec<Vec<WeightBinding<'a>>>, gathers: Vec<GatherBinding>, scratch: ByteLayout,
}
impl<'a> LayerBoundary<'a> {
    pub fn new(d: &TextDescription, headers: &'a HeaderAdmission, state: &'a StatePlan,
        layer: usize, context: u64, topology: Topology, wire_limit: usize) -> Result<Self> {
        let program=model(LayerPlan::new(d,layer,context,topology))?;
        need(context<=state.capacity() && topology==state.topology(), "layer and state context or topology differ")?;
        // Compare the whole source-derived plan. Equal byte counts do not establish identity.
        let expected=model(StatePlan::new(d,topology,state.capacity(),state.working_bytes()))?;
        need(*state==expected,"state belongs to a different operation program")?;
        Self::bind(program,headers,state,topology,wire_limit)
    }
    /// The boundary of every layer, one at a time in layer order. The first layer goes through `new`, which
    /// compares the state plan with the plan of this description. That comparison reads no layer and builds
    /// the whole state plan again (about 60 ms on a Spark), so the other layers do not repeat it: every
    /// boundary is the one `new` gives. A service start built all 45 layers twice this way.
    /// The caller must take the items from the first one and stop at the first error: the two checks of `new`
    /// are in the first item only. A caller that skips items, or wants one layer, uses `new`.
    pub fn each(d: &'a TextDescription, headers: &'a HeaderAdmission, state: &'a StatePlan,
        context: u64, topology: Topology, wire_limit: usize) -> impl Iterator<Item=Result<Self>> {
        let count=d.layers.len();
        (0..count.min(1)).map(move |layer| Self::new(d,headers,state,layer,context,topology,wire_limit))
            .chain((1..count).map(move |layer| Self::bind(model(LayerPlan::new(d,layer,context,topology))?,headers,state,topology,wire_limit)))
    }
    fn bind(program: LayerPlan, headers: &'a HeaderAdmission, state: &'a StatePlan, topology: Topology, wire_limit: usize) -> Result<Self> {
        let casts=model(program.bf16_cast_contract())?;
        let mut weights=Vec::new(); let mut gathers=Vec::new();
        for (index,step) in program.steps.iter().enumerate() {
            let mut bound=Vec::new();
            for weight in &step.weights {
                let view=headers.weights().get(&weight.name).ok_or_else(|| format!("missing admitted layer weight {}",weight.name))?;
                let logical=view.logical();
                need(logical.shape==weight.shape && logical.storage==weight.storage,"layer weight differs from admitted headers")?;
                let ownership=if let Split::Axis {axis,unit}=weight.split {
                    Some(model(Ownership::equal(&weight.shape,axis,unit,topology.parts))?)
                } else { None };
                let rows=if let Some(owner)=ownership.as_ref().filter(|_| weight.shape.len()==2) {
                    Some(model(ProjectionRows::new(owner,topology.rank,weight.storage))?)
                } else {None};
                bound.push(WeightBinding {view,ownership,rows});
            }
            weights.push(bound);
            // Preserve operation order and each operation's declared gather order.
            for name in &step.gather {
                let output=step.outputs.iter().position(|b| &b.name==name).ok_or("gather output is absent")?;
                let Split::Axis {axis,unit}=step.output_splits[output] else { return Err("gather requires a tensor interval".into()); };
                let dtype=casts.steps[index].outputs[output].dtype;
                let bytes=match dtype { Dtype::Bf16=>2,Dtype::F32|Dtype::U32=>4,
                    Dtype::Unresolved=>return Err(format!("{name}: physical gather dtype is unresolved")) };
                let ownership=model(Ownership::equal(&step.outputs[output].shape,axis,unit,topology.parts))?;
                let layout=GatherLayout::new(ownership,bytes,wire_limit)?;
                gathers.push(GatherBinding {step:index,name:name.clone(),dtype,layout});
            }
        }
        let scratch=model(ByteLayout::for_program(&program))?;
        Ok(Self {program,casts,state,weights,gathers,scratch})
    }
    pub fn program(&self) -> &LayerPlan { &self.program }
    pub fn casts(&self) -> &Contract { &self.casts }
    pub fn state(&self) -> &StatePlan { self.state }
    pub fn weights(&self, step: usize) -> Result<&[WeightBinding<'a>]> {
        self.weights.get(step).map(Vec::as_slice).ok_or_else(|| "layer step is outside the program".into())
    }
    pub fn gathers(&self) -> &[GatherBinding] { &self.gathers }
    /// These are logical host extents. They never authorize a physical cache dtype.
    pub fn scratch(&self) -> &ByteLayout { &self.scratch }
    pub fn check_scratch_offsets(&self, offsets: &[usize], capacity: usize) -> Result<()> {
        model(self.scratch.check_offsets(offsets,capacity))
    }
    /// No caller-provided capability flag can open this unresolved source contract.
    pub fn require_executable(&self) -> Result<()> {
        model(self.casts.require_complete())
    }
    /// Reuse the existing rank-ordered byte gather. This is a host copy adapter.
    /// Device stream completion and distributed group identity are not established here.
    pub fn gather<E: Exchange>(&self, step: usize, name: &str, exchange: &mut E, local: &[u8], global: &mut [u8]) -> Result<()> {
        let binding=self.gathers.iter().find(|g| g.step==step && g.name==name).ok_or("gather is not declared at this step")?;
        binding.layout.gather(exchange,self.program.topology.rank,local,global)
    }
    /// Bind the existing primitive ABI only. Other entries return named errors.
    /// A target revision or entry name is not evidence that its ABI is bound.
    pub fn mhc_shape(&self, step: usize) -> Result<Shape> {
        let operation=self.program.steps.get(step).ok_or("layer step is outside the program")?;
        if operation.kind!=Kind::MhcPrepare || operation.target.entries!=["mhc_prepare"] {
            return Err(format!("{}: missing checked device ABI binding for {:?} ({})",operation.id,operation.kind,operation.target.entries.join(", ")));
        }
        let s=&self.program.settings;
        let shape=Shape {rows:1,streams:number(s.residual.streams)?,hidden:number(s.hidden)?,
            iterations:number(s.residual.iterations)?,rms_eps:s.epsilon as f32,eps:s.residual.epsilon as f32};
        shape.validate().map_err(str::to_string)?; Ok(shape)
    }
    /// Read admitted components through a supplied reader. No device allocation occurs.
    pub fn prepare_mhc(&self, step: usize, mut read: impl FnMut(&ComponentView)->engine_model::Result<Vec<u8>>) -> Result<MhcOperands> {
        self.pack_mhc(step,|view| model(view.upload(&mut read)))
    }
    /// Use the retained files that supplied checkpoint admission. Do not reopen
    /// paths or duplicate the loader. Matching headers do not prove payload identity.
    pub fn prepare_mhc_checkpoint(&self, step: usize, checkpoint: &CheckpointAdmission) -> Result<MhcOperands> {
        self.pack_mhc(step,|view| {
            let name=&view.logical().name;
            need(checkpoint.headers().weights().get(name)==Some(view),"mHC checkpoint headers differ from the bound weight")?;
            model(checkpoint.upload(name))
        })
    }
    fn pack_mhc(&self, step: usize, mut load: impl FnMut(&WeightView)->Result<UploadWeight>) -> Result<MhcOperands> {
        let shape=self.mhc_shape(step)?;
        let weights=self.weights(step)?;
        need(weights.len()==3,"mHC requires function, base, and scale weights")?;
        let mut operands=Vec::new(); let mut dtypes=Vec::new();
        for weight in weights {
            need(weight.view.logical().storage==WeightStorage::Plain,"mHC parameters must use plain storage")?;
            let upload=load(weight.view)?;
            let mut values=Vec::new();
            match upload.dtype {
                StoredDtype::Bf16 => { for bytes in upload.weight.as_chunks::<2>().0 {
                    values.push(engine_model::flash::nvfp4::bf16_to_f32(u16::from_le_bytes(*bytes)));
                } },
                StoredDtype::F32 => { for bytes in upload.weight.as_chunks::<4>().0 {
                    values.push(f32::from_le_bytes(*bytes));
                } },
                _=>return Err("mHC parameter has no lossless FP32 operand conversion".into()),
            }
            dtypes.push(upload.dtype); operands.push(values);
        }
        let scale=operands.pop().unwrap(); let base=operands.pop().unwrap(); let weights=operands.pop().unwrap();
        need(weights.len()==shape.coefficients()*shape.wide() && base.len()==shape.coefficients() && scale.len()==3,
            "mHC packed operands differ from the shared ABI")?;
        Ok(MhcOperands {shape,weights,base,scale,stored_dtypes:dtypes.try_into().unwrap(),topology:self.program.topology})
    }
}

/// Exact header bytes widened to row-major FP32. No scale or reduction changes.
pub struct MhcOperands { shape: Shape, weights: Vec<f32>, base: Vec<f32>, scale: Vec<f32>, stored_dtypes: [StoredDtype;3], topology: Topology }
impl MhcOperands {
    pub fn shape(&self) -> Shape { self.shape }
    pub fn weights(&self) -> &[f32] { &self.weights }
    pub fn base(&self) -> &[f32] { &self.base }
    pub fn scale(&self) -> &[f32] { &self.scale }
    pub fn stored_dtypes(&self) -> [StoredDtype;3] { self.stored_dtypes }
    /// Retain real device owners on one backend. This admits the mHC primitive
    /// only. The full layer's source and cast refusal still applies.
    #[cfg(target_os = "linux")]
    pub fn upload<'b,'a>(&self, backend: &'b spark_gpu_probe::driver::Backend<'a>) -> Result<spark_gpu_probe::driver::ResidentMhc<'b,'a>> {
        backend.resident_mhc(self.shape,&self.weights,&self.base,&self.scale)
    }
    /// Check the full argument layout before a future resident-buffer adapter.
    /// Addresses describe storage only. This does not prove allocation ownership,
    /// uploaded contents, readiness, or completion. It cannot launch a kernel.
    /// The seventh span is the fused BF16 collapse output in FP32 lanes.
    pub fn arguments(&self, spans: [PhysicalSpan;7]) -> Result<Prepare> {
        let s=self.shape; let t=s.rows as usize;
        let lengths=[t*s.wide(),self.weights.len(),self.base.len(),self.scale.len(),t*s.wide(),t*s.coefficients(),t*s.hidden as usize];
        let mut ends=Vec::new();
        for (i,(span,count)) in spans.iter().zip(lengths).enumerate() {
            need(span.dtype==Dtype::F32 && span.packing==Packing::RowMajor,
                "mHC arguments require widened row-major FP32 planes")?;
            need(span.rank==self.topology.rank && span.context==spans[0].context && span.context!=0,
                "mHC argument has a different context or rank")?;
            need(span.address!=0 && span.address%4==0 && span.words==count,"mHC argument alignment or extent differs")?;
            let bytes=u64::try_from(count).ok().and_then(|n| n.checked_mul(4)).ok_or("mHC argument extent overflow")?;
            let end=span.address.checked_add(bytes).ok_or("mHC argument address overflow")?;
            for (other,&other_end) in spans[..i].iter().zip(&ends) {
                need(span.address>=other_end || other.address>=end,"mHC arguments overlap")?;
            }
            ends.push(end);
        }
        Ok(Prepare {input:spans[0].address,weights:spans[1].address,base:spans[2].address,scale:spans[3].address,
            normalized:spans[4].address,coefficients:spans[5].address,collapsed:spans[6].address,shape:s})
    }
}
/// Physical metadata supplied by the caller. The future backend must verify
/// the allocation owner and uploaded contents before it can use these addresses.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalSpan { pub address: u64, pub words: usize, pub context: u64, pub rank: u32, pub dtype: Dtype, pub packing: Packing }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packing { RowMajor, Bf16Fragments }
