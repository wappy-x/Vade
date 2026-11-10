//! Checked byte storage for the deferred layer program. No device allocation or launch.
//! Bytes follow the program's element types and local row-major shapes.
use super::{TextDescription, description::TokenMixer, layer::{Buffer, Element, LayerPlan, Split, Topology}};
use crate::{Result, invalid};
use std::{collections::BTreeSet, ops::Range, sync::Arc};

const ALIGN: usize = std::mem::align_of::<u32>();
fn need(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(invalid(why)) } }
fn add(a: usize,b: usize) -> Result<usize> {
    a.checked_add(b).filter(|&n| n <= isize::MAX as usize).ok_or_else(|| invalid("state byte sum overflow"))
}
fn mul(a: usize,b: usize) -> Result<usize> {
    a.checked_mul(b).filter(|&n| n <= isize::MAX as usize).ok_or_else(|| invalid("state byte product overflow"))
}
fn number(n: u64) -> Result<usize> { usize::try_from(n).map_err(|_| invalid("state dimension exceeds usize")) }
fn aligned(n: usize) -> Result<usize> { Ok(add(n,ALIGN-1)? / ALIGN * ALIGN) }
fn overlap(a: &Range<usize>,b: &Range<usize>) -> bool { !a.is_empty() && !b.is_empty() && a.start < b.end && b.start < a.end }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocation {
    pub name: String, pub shape: Vec<u64>, pub element: Element, pub bytes: Range<usize>,
    /// Inclusive steps from LayerPlan::lifetimes. Zero is the external input boundary.
    pub live: (usize,usize),
}
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ByteLayout { entries: Vec<Allocation>, bytes: usize }
impl ByteLayout {
    pub fn entries(&self) -> &[Allocation] { &self.entries }
    pub fn bytes(&self) -> usize { self.bytes }
    fn push(&mut self, buffer: Buffer, live: (usize,usize)) -> Result<()> {
        need(!self.entries.iter().any(|a| a.name==buffer.name),"duplicate storage name")?;
        let size=mul(buffer.elements()?,std::mem::size_of::<u32>())?;
        let start=aligned(self.bytes)?; let end=add(start,size)?;
        self.entries.push(Allocation { name:buffer.name,shape:buffer.shape,element:buffer.element,bytes:start..end,live });
        self.bytes=end; Ok(())
    }
    /// Check proposed offsets before a caller binds an arena. An offset is relative
    /// to an aligned base. Empty planes own no bytes. Live aliases are forbidden.
    pub fn check_offsets(&self, offsets: &[usize], capacity: usize) -> Result<()> {
        need(capacity <= isize::MAX as usize && offsets.len()==self.entries.len(),"invalid storage capacity or offset count")?;
        let mut spans=Vec::with_capacity(offsets.len());
        for (a,&start) in self.entries.iter().zip(offsets) {
            need(start%ALIGN==0,"unaligned storage offset")?;
            let end=add(start,a.bytes.len())?;
            need(end<=capacity,"storage exceeds arena capacity")?;
            let span=start..end;
            for (b,other) in self.entries.iter().zip(&spans) {
                let live=a.live.0<=b.live.1 && b.live.0<=a.live.1;
                need(!live || !overlap(&span,other),"live storage overlaps")?;
            }
            spans.push(span);
        }
        Ok(())
    }
    /// Reserve each declared local buffer once. Gathered outputs reserve their
    /// full shape. Step scratch stays replicated because the program has no split
    /// for it. This conservative layout does not infer a kernel packing ABI.
    pub fn for_program(program: &LayerPlan) -> Result<Self> {
        Self::retaining(program,&[])
    }
    fn retaining(program: &LayerPlan,retained: &[&str]) -> Result<Self> {
        program.validate()?;
        let mut life=program.lifetimes(); let splits=program.placements(); let mut out=Self::default();
        for &name in retained {
            let span=life.get_mut(name).ok_or_else(|| invalid("retained state output is absent"))?;
            span.1=program.steps.len();
        }
        for b in program.inputs.iter().chain(program.steps.iter().flat_map(|s| &s.outputs)) {
            let (first,last)=life[&b.name];
            out.push(local(program,b,splits[&b.name])?,(first.map_or(0,|v| v+1),last+1))?;
        }
        for (step,s) in program.steps.iter().enumerate() { for b in &s.scratch { out.push(b.clone(),(step+1,step+1))?; } }
        out.check_offsets(&out.entries.iter().map(|a| a.bytes.start).collect::<Vec<_>>(),out.bytes)?; Ok(out)
    }
}
fn local(p: &LayerPlan,b: &Buffer,split: Split) -> Result<Buffer> {
    Ok(p.topology.interval(&b.shape,split)?.map_or_else(|| b.clone(),|(axis,_,width)| {
        let mut out=b.clone(); out.shape[axis]=width; out
    }))
}
fn buffer<'a>(p: &'a LayerPlan,name: &str) -> Result<&'a Buffer> {
    p.inputs.iter().chain(p.steps.iter().flat_map(|s| &s.outputs)).find(|b| b.name==name)
        .ok_or_else(|| invalid("state plane is absent from layer program"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlaneKind { KdaState, KdaHistory, Latent, IndexKeys, IndexGates, Pools }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatePlane {
    pub layer: usize, pub kind: PlaneKind, pub allocation: Allocation,
    /// A fixed KDA plane has no row divisor. Cache rows use one. Pools use
    /// the configured group length. No completed index group is discarded.
    row_divisor: Option<u64>, capacity: u64,
}
impl StatePlane {
    pub fn used_bytes(&self, context: u64) -> Result<usize> {
        need(context<=self.capacity,"state context exceeds token capacity")?;
        if let Some(divisor)=self.row_divisor {
            let rows=context/divisor;
            need(rows<=self.allocation.shape[0],"state rows exceed plane capacity")?;
            let row=Buffer::new("row",&self.allocation.shape[1..]).elements()?;
            mul(mul(number(rows)?,row)?,std::mem::size_of::<u32>())
        } else { Ok(self.allocation.bytes.len()) }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct LayerStorage { pub program: LayerPlan, pub scratch: ByteLayout }
#[derive(Debug, Clone, PartialEq)]
pub struct StatePlan {
    capacity: u64, topology: Topology, layers: Vec<LayerStorage>,
    planes: Vec<StatePlane>, state: ByteLayout, scratch_bytes: usize, working: ByteLayout,
}
impl StatePlan {
    /// Admission covers a committed bank, a separate append bank, and one scratch
    /// arena for an ordered lane. Weights and saved cache owners need separate
    /// budgets. No activation or model payload is allocated by this constructor.
    pub fn new(d: &TextDescription,topology: Topology,capacity: u64,byte_limit: usize) -> Result<Self> {
        topology.validate()?;
        need(capacity>0 && capacity<=d.max_positions && capacity<=u32::MAX as u64,"invalid state token capacity")?;
        need(byte_limit>0 && byte_limit<=isize::MAX as usize && !d.layers.is_empty(),"invalid state byte capacity or empty decoder")?;
        let mut out=Self { capacity,topology,layers:Vec::new(),planes:Vec::new(),state:ByteLayout::default(),scratch_bytes:0,working:ByteLayout::default() };
        for layer in 0..d.layers.len() {
            let program=LayerPlan::new(d,layer,capacity,topology)?;
            match program.settings.decoder.mixer {
                TokenMixer::Kda(_) => {
                    out.plane(&program,PlaneKind::KdaState,"next_state",None)?;
                    out.plane(&program,PlaneKind::KdaHistory,"next_history",None)?;
                },
                TokenMixer::LatentAttention(a) => {
                    for (kind,name) in [(PlaneKind::Latent,"next_latent_cache"),(PlaneKind::IndexKeys,"next_index_keys"),(PlaneKind::IndexGates,"next_index_gates")] {
                        out.plane(&program,kind,name,Some(1))?;
                    }
                    out.plane(&program,PlaneKind::Pools,"pool_keys",Some(a.indexer.pool))?;
                },
            }
            // The session also retains complete pools. Hold their output until
            // state copy-out ends, even though its last arithmetic read is earlier.
            let retained: &[&str]=if matches!(program.settings.decoder.mixer,TokenMixer::LatentAttention(_)) { &["pool_keys"] } else { &[] };
            let scratch=ByteLayout::retaining(&program,retained)?;
            out.scratch_bytes=out.scratch_bytes.max(scratch.bytes());
            out.layers.push(LayerStorage { program,scratch });
        }
        for (name,bytes) in [("committed",out.state.bytes()),("append",out.state.bytes()),("scratch",out.scratch_bytes)] {
            let words=u64::try_from(bytes/std::mem::size_of::<u32>()).map_err(|_| invalid("state byte extent exceeds u64"))?;
            let mut b=Buffer::new(name,&[words]); b.element=Element::U32; out.working.push(b,(0,usize::MAX))?;
        }
        need(out.working.bytes()<=byte_limit,"state and scratch exceed byte capacity")?;
        out.state.check_offsets(&out.state.entries.iter().map(|a| a.bytes.start).collect::<Vec<_>>(),out.state.bytes())?;
        Ok(out)
    }
    fn plane(&mut self,p: &LayerPlan,kind: PlaneKind,name: &str,row_divisor: Option<u64>) -> Result<()> {
        let mut b=local(p,buffer(p,name)?,p.placements()[name])?;
        b.name=format!("{}.{}",p.layer,name);
        self.state.push(b,(0,usize::MAX))?;
        self.planes.push(StatePlane { layer:p.layer,kind,allocation:self.state.entries.last().unwrap().clone(),row_divisor,capacity:self.capacity }); Ok(())
    }
    pub fn capacity(&self) -> u64 { self.capacity }
    pub fn topology(&self) -> Topology { self.topology }
    pub fn layers(&self) -> &[LayerStorage] { &self.layers }
    pub fn planes(&self) -> &[StatePlane] { &self.planes }
    pub fn state_layout(&self) -> &ByteLayout { &self.state }
    pub fn scratch_bytes(&self) -> usize { self.scratch_bytes }
    pub fn working_layout(&self) -> &ByteLayout { &self.working }
    pub fn working_bytes(&self) -> usize { self.working.bytes() }
    /// All active lanes need their own working storage. Saved boundaries own a
    /// separate bank each. The caller supplies the aggregate host byte budget.
    pub fn check_owners(&self,lanes: usize,saved: usize,byte_limit: usize) -> Result<usize> {
        need(lanes>0 && byte_limit<=isize::MAX as usize,"invalid owner count or byte capacity")?;
        let total=add(mul(self.working.bytes(),lanes)?,mul(self.state.bytes(),saved)?)?;
        need(total<=byte_limit,"state owners exceed byte capacity")?; Ok(total)
    }
    pub fn plane_at(&self,layer: usize,kind: PlaneKind) -> Result<&StatePlane> {
        self.planes.iter().find(|p| p.layer==layer && p.kind==kind).ok_or_else(|| invalid("state plane is absent"))
    }
    /// Read-only views into the raw key and gate owners. Completed raw rows stay
    /// retained. The tail is not another overlapping allocation or a packed ABI.
    pub fn index_tail(&self,layer: usize,context: u64) -> Result<IndexTail> {
        need(context<=self.capacity,"tail context exceeds capacity")?;
        let p=self.layers.get(layer).ok_or_else(|| invalid("tail layer is absent"))?;
        let TokenMixer::LatentAttention(a)=p.program.settings.decoder.mixer else { return Err(invalid("tail requires an indexer")); };
        let pool=a.indexer.pool; let complete_pools=context/pool; let complete_rows=complete_pools*pool;
        let keys=self.plane_at(layer,PlaneKind::IndexKeys)?; let gates=self.plane_at(layer,PlaneKind::IndexGates)?;
        Ok(IndexTail { complete_pools,rows:context-complete_rows,
            keys:add(keys.allocation.bytes.start,keys.used_bytes(complete_rows)?)?..add(keys.allocation.bytes.start,keys.used_bytes(context)?)?,
            gates:add(gates.allocation.bytes.start,gates.used_bytes(complete_rows)?)?..add(gates.allocation.bytes.start,gates.used_bytes(context)?)? })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexTail { pub complete_pools: u64, pub rows: u64, pub keys: Range<usize>, pub gates: Range<usize> }

fn zeros(count: usize) -> Result<Vec<u8>> {
    need(count<=isize::MAX as usize,"host state capacity exceeds isize")?;
    let mut out=Vec::new(); out.try_reserve_exact(count).map_err(|_| invalid("cannot reserve host state bytes"))?;
    out.resize(count,0); Ok(out)
}
fn copied(bytes: &[u8]) -> Result<Vec<u8>> { let mut out=zeros(bytes.len())?; out.copy_from_slice(bytes); Ok(out) }

/// A host owner for opaque state bytes. It does not compute model state or
/// prove kernel results. Each lane owns its bank. No scratch survives an append.
pub struct SessionState { plan: Arc<StatePlan>, context: u64, bytes: Vec<u8> }
/// A saved boundary owns its bytes. It never borrows a live lane or scratch.
pub struct SavedState { plan: Arc<StatePlan>, context: u64, bytes: Vec<u8> }
impl SavedState { pub fn context(&self) -> u64 { self.context } }
impl SessionState {
    pub fn new(plan: Arc<StatePlan>) -> Result<Self> { let bytes=zeros(plan.state.bytes())?; Ok(Self { plan,context:0,bytes }) }
    pub fn plan(&self) -> &StatePlan { &self.plan }
    pub fn context(&self) -> u64 { self.context }
    pub fn read(&self,layer: usize,kind: PlaneKind) -> Result<&[u8]> {
        let p=self.plan.plane_at(layer,kind)?; let start=p.allocation.bytes.start;
        Ok(&self.bytes[start..add(start,p.used_bytes(self.context)?)?])
    }
    /// Start exactly one causal token. Dropping an append, including an error
    /// path, drops only its private bank. Committed bytes remain unchanged.
    pub fn begin_append(&mut self) -> Result<Append<'_>> {
        need(self.context<self.plan.capacity,"append exceeds token capacity")?;
        let bytes=copied(&self.bytes)?; Ok(Append { owner:self,bytes,written:BTreeSet::new() })
    }
    pub fn save(&self,cache_byte_limit: usize) -> Result<SavedState> {
        need(self.bytes.len()<=cache_byte_limit,"saved state exceeds cache byte capacity")?;
        Ok(SavedState { plan:self.plan.clone(),context:self.context,bytes:copied(&self.bytes)? })
    }
    /// Restore uses a separate copy. Allocation failure leaves the lane intact.
    pub fn restore(&mut self,saved: &SavedState) -> Result<()> {
        need(*self.plan==*saved.plan,"saved state has a different layout or program")?;
        let bytes=copied(&saved.bytes)?; self.bytes=bytes; self.context=saved.context; Ok(())
    }
    /// A KDA state cannot be recovered by shortening a row count. The caller must
    /// supply a saved boundary. This also restores the exact indexer tail/pools.
    pub fn truncate(&mut self,saved: &SavedState) -> Result<()> {
        need(saved.context<=self.context,"truncate boundary is after the current context")?;
        self.restore(saved)
    }
}
pub struct Append<'a> { owner: &'a mut SessionState, bytes: Vec<u8>, written: BTreeSet<(usize,PlaneKind)> }
impl Append<'_> {
    pub fn context(&self) -> u64 { self.owner.context+1 }
    pub fn rollback(self) {}
    /// Supply the full used plane returned by the token program, including
    /// completed pools. Unused capacity is cleared before the bank can commit.
    pub fn write(&mut self,layer: usize,kind: PlaneKind,bytes: &[u8]) -> Result<()> {
        let p=self.owner.plan.plane_at(layer,kind)?; let used=p.used_bytes(self.context())?;
        need(bytes.len()==used,"append plane byte count differs from used extent")?;
        let span=p.allocation.bytes.clone(); self.bytes[span.clone()].fill(0);
        self.bytes[span.start..add(span.start,used)?].copy_from_slice(bytes);
        self.written.insert((layer,kind)); Ok(())
    }
    /// Call only after the complete token program and its ordered work finish.
    /// A future device owner must wait for launch and transport completion first.
    pub fn commit(self) -> Result<()> {
        need(self.owner.plan.planes.iter().all(|p| self.written.contains(&(p.layer,p.kind))),"append is missing a state plane")?;
        self.owner.context+=1; self.owner.bytes=self.bytes; Ok(())
    }
}
