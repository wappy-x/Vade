//! Ordered copy gathers at the operation boundaries of a described decoder.
//! The pair uses the existing mailbox wire. No partial dot products cross it.
use crate::{Result, tensor_layout::{GatherLayout, MailboxRows}, transport::Ring};
use engine_model::plan::{layer::{LayerPlan, Split, Element}, tensor_split::{Ownership, ProjectionRows}};

pub const PAIR_RANKS: usize = 2;
const AGREEMENT_BYTES: usize = 48;
const MANIFEST_LIMIT: usize = 8 << 20;
fn need(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(why.into()) } }
fn model<T>(value: engine_model::Result<T>) -> Result<T> { value.map_err(|e| e.to_string()) }

/// The physical dtype must come from the operator binding. This does not select
/// a checkpoint cast. Integer selection buffers cannot use a floating dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType { Bf16, F32, U32 }
impl WireType { pub fn bytes(self) -> usize { match self { Self::Bf16=>2,Self::F32|Self::U32=>4 } } }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gather { step: usize, output: usize, name: String, dtype: WireType, layout: GatherLayout, rows: Vec<MailboxRows> }
impl Gather {
    pub fn step(&self) -> usize { self.step }
    pub fn output(&self) -> usize { self.output }
    pub fn name(&self) -> &str { &self.name }
    pub fn dtype(&self) -> WireType { self.dtype }
    pub fn layout(&self) -> &GatherLayout { &self.layout }
    pub fn rows(&self, rank: usize) -> Result<&MailboxRows> { self.rows.get(rank).ok_or_else(|| "gather rank is outside the pair".into()) }
}

/// A checked operation-boundary adapter. It keeps the original program and
/// weight row coordinates. It does not authorize an operator or admit a model.
#[derive(Debug, Clone)]
pub struct PairPlan { program: LayerPlan, gathers: Vec<Gather>, manifest: Vec<u8>, slot_bytes: usize }
impl PairPlan {
    /// Types follow step order, then each step's declared gather order.
    /// `byte_limit` bounds each global tensor. The session owns aggregate limits.
    pub fn new(program: LayerPlan, types: &[WireType], byte_limit: usize) -> Result<Self> {
        model(program.validate())?;
        need(program.topology.parts as usize==PAIR_RANKS,"layer transport requires the pair topology")?;
        need(types.len()==program.steps.iter().map(|s| s.gather.len()).sum::<usize>(),"physical gather dtype count differs from the program")?;
        let mut gathers=Vec::new(); let mut slot_bytes=4;
        for (step,operation) in program.steps.iter().enumerate() {
            // Projected weight rows retain the original shape and scale origin.
            for weight in &operation.weights { if let Split::Axis {axis,unit}=weight.split {
                let owner=model(Ownership::equal(&weight.shape,axis,unit,program.topology.parts))?;
                if weight.shape.len()==2 { model(ProjectionRows::new(&owner,program.topology.rank,weight.storage))?; }
            } }
            let mut seen=std::collections::BTreeSet::new();
            for name in &operation.gather {
                need(seen.insert(name),"duplicate operation gather")?;
                let output=operation.outputs.iter().position(|b| &b.name==name).ok_or("gather output is absent")?;
                let buffer=&operation.outputs[output];
                let dtype=types[gathers.len()];
                need(matches!((buffer.element,dtype),(Element::U32,WireType::U32)|(Element::F32,WireType::F32|WireType::Bf16)),"physical gather dtype differs from the logical element")?;
                let Split::Axis {axis,unit}=operation.output_splits[output] else { return Err("gather output has no tensor interval".into()); };
                let layout=GatherLayout::new(model(Ownership::equal(&buffer.shape,axis,unit,program.topology.parts))?,dtype.bytes(),byte_limit)?;
                let rows=(0..PAIR_RANKS).map(|r| layout.mailbox(r as u32,byte_limit)).collect::<Result<Vec<_>>>()?;
                slot_bytes=slot_bytes.max(rows[0].packet_bytes());
                gathers.push(Gather {step,output,name:name.clone(),dtype,layout,rows});
            }
        }
        // Debug is a source-versioned exact serialization, not a hash. Normalize
        // only rank. Include scalar bits as well as readable operation data.
        let mut canonical=program.clone(); canonical.topology.rank=0;
        let s=&program.settings;
        let mut bits=vec![s.epsilon.to_bits(),s.swiglu_limit.to_bits(),s.residual.epsilon.to_bits(),s.routing.scale.to_bits(),u64::from(s.index_key_epsilon.to_bits())];
        if let engine_model::plan::description::TokenMixer::Kda(k)=s.decoder.mixer { bits.push(k.gate_lower_bound.to_bits()); }
        let manifest=format!("vade-layer-copy-v1\n{canonical:?}\n{types:?}\n{bits:?}\n").into_bytes();
        need(manifest.len()<=MANIFEST_LIMIT,"layer transport manifest exceeds its limit")?;
        Ok(Self {program,gathers,manifest,slot_bytes})
    }
    pub fn program(&self) -> &LayerPlan { &self.program }
    pub fn gathers(&self) -> &[Gather] { &self.gathers }
    pub fn step_gathers(&self, step: usize) -> impl Iterator<Item=&Gather> { self.gathers.iter().filter(move |g| g.step==step) }
    pub fn slot_bytes(&self) -> usize { self.slot_bytes }
    pub fn manifest(&self) -> &[u8] { &self.manifest }
}

/// Control and persistent message counts. The ring is dedicated to this group.
/// A failed agreement or an abandoned layer makes the group unusable.
pub struct PairGroup { ring: Ring, sequence: u64, calls: u64, failed: bool }
impl PairGroup {
    pub fn new(mut ring: Ring) -> Result<Self> {
        if ring.world()!=PAIR_RANKS { ring.abort(); return Err("layer transport needs a pair control group".into()); }
        Ok(Self {ring,sequence:0,calls:0,failed:false})
    }
    pub fn rank(&self) -> usize { self.ring.rank() }
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn failed(&self) -> bool { self.failed }
    fn fail(&mut self) { self.failed=true; self.ring.abort(); }
    /// Both ranks must call with the same request/lane, causal token, and plan.
    /// Fixed-size agreement precedes variable-size data, so a mismatch cannot
    /// turn one rank's manifest into the other rank's next message.
    pub fn agree(&mut self, plan: &PairPlan, request: u64, token: u64) -> Result<()> {
        let result=(|| {
            need(!self.failed,"layer transport failed earlier")?;
            need(plan.program.topology.rank as usize==self.rank(),"layer transport rank differs from the program")?;
            let fields=[plan.manifest.len() as u64,request,token,self.sequence,self.calls,self.rank() as u64];
            let mut wire=vec![0;AGREEMENT_BYTES*PAIR_RANKS];
            for (dst,value) in wire[self.rank()*AGREEMENT_BYTES..(self.rank()+1)*AGREEMENT_BYTES].as_chunks_mut::<8>().0.iter_mut().zip(fields) { dst.copy_from_slice(&value.to_le_bytes()); }
            self.ring.all_gather(&mut wire,AGREEMENT_BYTES)?;
            for (rank,packet) in wire.as_chunks::<AGREEMENT_BYTES>().0.iter().enumerate() {
                let got=packet.as_chunks::<8>().0.iter().map(|v| u64::from_le_bytes(*v)).collect::<Vec<_>>();
                need(got[..5]==fields[..5] && got[5]==rank as u64,"pair layer request, token, sequence, or manifest length differs")?;
            }
            let mut wire=vec![0;plan.manifest.len()*PAIR_RANKS];
            let at=self.rank()*plan.manifest.len(); wire[at..at+plan.manifest.len()].copy_from_slice(&plan.manifest);
            self.ring.all_gather(&mut wire,plan.manifest.len())?;
            need(wire.chunks_exact(plan.manifest.len()).all(|v| v==plan.manifest),"pair operation programs or physical gather types differ")?;
            self.calls=self.calls.checked_add(1).ok_or("pair layer call count overflow")?;
            Ok(())
        })();
        if result.is_err() { self.fail(); } result
    }
    #[cfg(any(target_os="linux",test))]
    fn next_sequence(&mut self) -> Result<u64> {
        let next=self.sequence.checked_add(1).ok_or("pair mailbox sequence overflow")?;
        // Start a new group before the low device sequence can wrap to zero.
        need(next<=u32::MAX as u64,"pair mailbox sequence exhausted; create a new group")?;
        self.sequence=next; Ok(next)
    }
    pub fn complete(&mut self) -> Result<()> {
        need(!self.failed,"pair transport failed before completion")?;
        let result=self.ring.barrier(); if result.is_err() { self.fail(); } result
    }
}

#[derive(Default)]
#[cfg(any(target_os="linux",test))]
struct Order { next: usize, complete: bool }
#[cfg(any(target_os="linux",test))]
impl Order {
    fn check(&self, plan: &PairPlan, step: usize, outputs: usize) -> Result<()> {
        need(!self.complete && step==self.next && step<plan.program.steps.len(),"layer operations are out of transport order")?;
        need(outputs==plan.step_gathers(step).count(),"layer gather outputs are missing or extra")
    }
    fn finish(&self, plan: &PairPlan) -> Result<()> { need(self.next==plan.program.steps.len(),"layer ended before every operation boundary") }
}

#[cfg(target_os="linux")]
use spark_gpu_probe::driver::{Backend, DeviceVector, TpMailbox};

/// The mailbox is fixed for the group lifetime. Its existing wire must be
/// attached before begin. The backend owns every vector and the mailbox.
#[cfg(target_os="linux")]
pub struct DevicePair<'b,'a> { group: PairGroup, backend: &'b Backend<'a>, mailbox: &'b TpMailbox<'a>, limit_ns: u64 }
#[cfg(target_os="linux")]
impl<'b,'a> DevicePair<'b,'a> {
    pub fn new(group: PairGroup, backend: &'b Backend<'a>, mailbox: &'b TpMailbox<'a>, limit_ns: u64) -> Result<Self> {
        let layout=mailbox.layout();
        need(layout.world==PAIR_RANKS && layout.frame==0 && layout.slots==spark_gpu_probe::tp_mailbox_host::TP_MAILBOX_SLOTS,
            "pair mailbox layout differs from the existing wire")?;
        need(limit_ns>0 && limit_ns<1<<47 && group.sequence==0,"pair mailbox is reused or has an invalid timeout")?;
        mailbox.claim_copy_group()?;
        Ok(Self {group,backend,mailbox,limit_ns})
    }
    pub fn sequence(&self) -> u64 { self.group.sequence() }
    pub fn failed(&self) -> bool { self.group.failed() || self.mailbox.failed() }
    pub fn begin<'s,'p>(&'s mut self, plan: &'p PairPlan, request: u64, token: u64) -> Result<DeviceLayer<'s,'p,'b,'a>> {
        let result=(|| {
            need(!self.failed() && self.mailbox.slot_bytes()>=plan.slot_bytes,"pair mailbox is failed or smaller than the layer plan")?;
            self.group.agree(plan,request,token)
        })();
        if let Err(error)=result { self.fail(); return Err(error); }
        Ok(DeviceLayer {pair:self,plan,order:Order::default()})
    }
    fn fail(&mut self) { self.mailbox.fail(); self.group.fail(); }
}

/// Call after each operator, including operators without gathers. Supply only
/// its declared gather outputs, in declaration order, as global row-major
/// storage. The producer writes its own interval at the original global offset.
#[cfg(target_os="linux")]
pub struct DeviceLayer<'s,'p,'b,'a> { pair: &'s mut DevicePair<'b,'a>, plan: &'p PairPlan, order: Order }
#[cfg(target_os="linux")]
impl DeviceLayer<'_,'_,'_,'_> {
    pub fn after_step(&mut self, step: usize, outputs: &mut [&mut DeviceVector<'_>]) -> Result<()> {
        let result=(|| {
            self.order.check(self.plan,step,outputs.len())?;
            need(!self.pair.failed(),"pair transport failed before the operation boundary")?;
            // Check every shape before the first put. The backend also checks
            // each allocation's context when it queues its copy.
            for (g,vector) in self.plan.step_gathers(step).zip(outputs.iter()) {
                need(vector.len()==g.layout.bytes()/4 && !vector.is_segmented() && vector.writable_from(0),"pair gather requires complete writable global storage")?;
            }
            for (g,vector) in self.plan.step_gathers(step).zip(outputs.iter_mut()) {
                let rank=self.pair.group.rank();
                let seq=self.pair.group.next_sequence()?;
                g.rows[rank].put(self.pair.backend,self.pair.mailbox,vector,seq)?;
                // Global rank order fixes placement. There is no float sum.
                for (origin,rows) in g.rows.iter().enumerate() { if origin!=rank {
                    rows.take(self.pair.backend,self.pair.mailbox,vector,rank,seq,self.pair.limit_ns)?;
                } }
                self.pair.backend.synchronize()?;
                need(!self.pair.mailbox.failed(),"pair gather timed out or its wire failed")?;
            }
            self.order.next+=1; Ok(())
        })();
        if result.is_err() {
            self.pair.fail();
            // Even a partial launch must finish before its borrowed vectors go.
            let drained=self.pair.backend.synchronize(); return result.and(drained);
        }
        result
    }
    pub fn finish(mut self) -> Result<()> {
        self.order.finish(self.plan)?;
        self.pair.backend.synchronize()?;
        need(!self.pair.failed(),"pair failed before layer completion")?;
        self.pair.group.complete()?;
        self.order.complete=true; Ok(())
    }
}
#[cfg(target_os="linux")]
impl Drop for DeviceLayer<'_,'_,'_,'_> {
    fn drop(&mut self) { if !self.order.complete { self.pair.fail(); let _=self.pair.backend.synchronize(); } }
}

#[cfg(test)]
#[path = "layer_transport_tests.rs"]
mod tests;
