//! Resident byte owners for an ordered decoder lane. Storage does not admit model arithmetic.
use crate::Result;
use engine_model::plan::{description::TokenMixer, layer::cast::Dtype, state::{StatePlan, PlaneKind}};
use std::{cell::{Cell, RefCell}, collections::BTreeSet, rc::Rc};

#[cfg(target_os="linux")]
#[path="session_state/device.rs"]
pub mod device;

fn need(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(why.into()) } }
fn add(a: usize,b: usize) -> Result<usize> {
    a.checked_add(b).filter(|&n| n<=isize::MAX as usize).ok_or_else(|| "session byte sum overflow".into())
}
fn mul(a: usize,b: usize) -> Result<usize> {
    a.checked_mul(b).filter(|&n| n<=isize::MAX as usize).ok_or_else(|| "session byte product overflow".into())
}
fn extent(n: usize) -> Result<usize> {
    let bytes=add(n.max(1),63)?/64*64;
    need(bytes/4<=u32::MAX as usize,"session plane exceeds the device vector ABI")?; Ok(bytes)
}

/// A plane's bytes are opaque. Its producer must supply the qualified storage ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane { pub layer: usize, pub kind: PlaneKind, pub length: Length }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Length { Fixed(usize), Rows { row_bytes: usize, tokens_per_row: u64 } }
#[derive(Debug, Clone, Copy)]
pub struct CacheType { pub layer: usize, pub kind: PlaneKind, pub dtype: Dtype }
impl Plane {
    pub fn used_bytes(&self, context: u64) -> Result<usize> {
        match self.length {
            Length::Fixed(bytes)=>{ need(bytes<=isize::MAX as usize,"session plane exceeds address range")?; Ok(bytes) },
            Length::Rows {row_bytes,tokens_per_row}=>{
                need(row_bytes>0 && tokens_per_row>0,"invalid session row geometry")?;
                let rows=usize::try_from(context/tokens_per_row).map_err(|_| "session row count exceeds usize")?;
                mul(row_bytes,rows)
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout { capacity: u64, planes: Vec<Plane>, sizes: Vec<usize>, scratch: usize, bank: usize }
impl Layout {
    /// Include all physical packing, status, stream, and launch scratch in scratch_bytes.
    /// This checks storage only. It cannot certify a checkpoint's dtype or cast contract.
    pub fn new(capacity: u64,planes: Vec<Plane>,scratch_bytes: usize) -> Result<Self> {
        need(capacity>0 && capacity<=u32::MAX as u64 && !planes.is_empty(),"invalid session capacity or empty state")?;
        let mut keys=BTreeSet::new(); let mut sizes=Vec::new(); let mut bank=0;
        for p in &planes {
            need(keys.insert((p.layer,p.kind)),"duplicate session state plane")?;
            let bytes=extent(p.used_bytes(capacity)?)?; bank=add(bank,bytes)?; sizes.push(bytes);
        }
        Ok(Self {capacity,planes,sizes,scratch:extent(scratch_bytes)?,bank})
    }
    /// Storage for the existing FP32 logical reference. Never use these widths
    /// as a checkpoint cache ABI. Physical KDA history has a different packing.
    pub fn logical_reference(plan: &StatePlan) -> Result<Self> {
        let mut planes=Vec::new();
        for p in plan.planes() {
            let length=match p.kind {
                PlaneKind::KdaState|PlaneKind::KdaHistory=>Length::Fixed(p.allocation.bytes.len()),
                _=>{
                    let every=if p.kind==PlaneKind::Pools {
                        let TokenMixer::LatentAttention(a)=plan.layers()[p.layer].program.settings.decoder.mixer else {
                            return Err("pool plane requires an indexer".into());
                        }; a.indexer.pool
                    } else {1};
                    let row=p.allocation.shape[1..].iter().try_fold(4usize,|n,&v| {
                        mul(n,usize::try_from(v).map_err(|_| "session dimension exceeds usize")?)
                    })?;
                    Length::Rows {row_bytes:row,tokens_per_row:every}
                },
            };
            planes.push(Plane {layer:p.layer,kind:p.kind,length});
        }
        Self::new(plan.capacity(),planes,plan.scratch_bytes())
    }
    /// Conditional physical capacity from explicit producer dtypes. This is not
    /// execution admission. In particular, no unresolved dtype gets a default.
    /// Preserve the full oldest KDA history slot, unlike the logical past plane.
    /// The driver supplies additional physical packing and transport scratch.
    pub fn physical_storage(plan: &StatePlan,types: &[CacheType],abi_scratch_bytes: usize) -> Result<Self> {
        need(types.len()==plan.planes().len(),"physical cache type count differs from state plan")?;
        let logical=Self::logical_reference(plan)?; let mut keys=BTreeSet::new();
        for ty in types {need(keys.insert((ty.layer,ty.kind)),"duplicate physical cache type")?;}
        let mut planes=Vec::new();
        for (p,logical) in plan.planes().iter().zip(logical.planes) {
            let ty=types.iter().find(|t| t.layer==p.layer && t.kind==p.kind).ok_or("physical cache type is absent")?;
            let width=match ty.dtype {Dtype::Bf16=>2,Dtype::F32=>4,_=>return Err("physical cache dtype is unresolved or not floating point".into())};
            need(p.kind!=PlaneKind::KdaState || ty.dtype==Dtype::F32,"KDA recurrent state requires FP32 storage")?;
            let length=if p.kind==PlaneKind::KdaHistory {
                let TokenMixer::Kda(k)=plan.layers()[p.layer].program.settings.decoder.mixer else {return Err("history plane requires KDA".into());};
                let mut shape=p.allocation.shape.clone();
                need(shape.len()==3,"invalid logical KDA history shape")?; shape[1]=k.conv;
                let bytes=shape.iter().try_fold(width,|n,&v| mul(n,usize::try_from(v).map_err(|_| "history dimension exceeds usize")?))?;
                Length::Fixed(bytes)
            } else {match logical.length {
                Length::Fixed(bytes)=>Length::Fixed(mul(bytes/4,width)?),
                Length::Rows {row_bytes,tokens_per_row}=>Length::Rows {row_bytes:mul(row_bytes/4,width)?,tokens_per_row},
            }};
            planes.push(Plane {layer:p.layer,kind:p.kind,length});
        }
        Self::new(plan.capacity(),planes,add(plan.scratch_bytes(),abi_scratch_bytes)?)
    }
    /// Fail before allocation while the source-derived physical contract is incomplete.
    pub fn checkpoint(plan: &StatePlan) -> Result<Self> {
        for layer in plan.layers() {
            layer.program.bf16_cast_contract().map_err(|e| e.to_string())?.require_complete().map_err(|e| e.to_string())?;
        }
        Err("checkpoint session storage needs checked physical ABI descriptors".into())
    }
    pub fn capacity(&self) -> u64 { self.capacity }
    pub fn planes(&self) -> &[Plane] { &self.planes }
    pub fn bank_bytes(&self) -> usize { self.bank }
    pub fn scratch_bytes(&self) -> usize { self.scratch }
    pub fn lane_bytes(&self) -> Result<usize> { add(mul(self.bank,2)?,self.scratch) }
    pub fn plane(&self,layer: usize,kind: PlaneKind) -> Result<usize> {
        self.planes.iter().position(|p| p.layer==layer && p.kind==kind).ok_or_else(|| "session state plane is absent".into())
    }
}

/// Rank-local admission includes existing weight arenas and runtime allocations.
/// Runtime includes transport staging and backend overhead outside these owners.
#[derive(Debug, Clone, Copy)]
pub struct Budget { pub weights: usize, pub runtime: usize, pub reserve: usize, pub lanes: usize, pub saved: usize, pub limit: usize }
impl Budget {
    pub fn required(&self,layout: &Layout) -> Result<usize> {
        need(self.lanes>0 && self.limit<=isize::MAX as usize,"invalid session budget")?;
        let fixed=add(add(self.weights,self.runtime)?,self.reserve)?;
        let working=mul(self.lanes,layout.lane_bytes()?)?;
        let total=add(add(fixed,working)?,mul(self.saved,layout.bank_bytes())?)?;
        need(total<=self.limit,"weights, runtime, and session owners exceed rank capacity")?; Ok(total)
    }
}

/// All calls use one ordered compute queue. copy copies the complete equal-size
/// buffers. write copies a prefix without conversion. wait drains partial work
/// on error too. Buffers must not have independent asynchronous users.
pub trait Store {
    type Buffer;
    fn allocate(&self,bytes: usize) -> Result<Self::Buffer>;
    fn size(&self,buffer: &Self::Buffer) -> usize;
    fn zero(&self,buffer: &mut Self::Buffer) -> Result<()>;
    fn copy(&self,source: &Self::Buffer,destination: &mut Self::Buffer) -> Result<()>;
    fn write(&self,buffer: &mut Self::Buffer,bytes: &[u8]) -> Result<()>;
    fn read(&self,buffer: &Self::Buffer) -> Result<Vec<u8>>;
    fn wait(&self) -> Result<()>;
}
/// Drain all non-compute users, including partial transport submissions. Return
/// success only when no user can access any borrowed session buffer again.
pub trait Completion { fn wait(&mut self) -> Result<()>; }
/// Use only when the callback does not submit work outside the store queue.
pub struct ComputeOnly;
impl Completion for ComputeOnly { fn wait(&mut self) -> Result<()> { Ok(()) } }

struct Inner<S: Store> {
    store: S, layout: Layout, budget: Budget, lanes: Cell<usize>, saved: Cell<usize>, failure: RefCell<Option<String>>,
}
impl<S: Store> Inner<S> {
    fn healthy(&self) -> Result<()> { self.failure.borrow().as_ref().map_or(Ok(()),|e| Err(format!("session pool is retired: {e}"))) }
    fn retire(&self,error: &str) { if self.failure.borrow().is_none() { *self.failure.borrow_mut()=Some(error.into()); } }
    fn wait(&self) -> Result<()> {
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.store.wait()))
            .unwrap_or_else(|_| Err("compute completion panicked".into()));
        if let Err(e)=&result { self.retire(e); } result
    }
    fn ordered<T>(&self,submit: impl FnOnce()->Result<T>) -> Result<T> {
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(submit));
        let waited=self.wait();
        let value=match result {Ok(result)=>result?,Err(payload)=>std::panic::resume_unwind(payload)};
        waited?; Ok(value)
    }
}
/// A pool is one admitted rank and checkpoint instance. Saved banks cannot cross
/// pools, even when their byte shapes match. Credits live as long as their owners.
pub struct Pool<S: Store> { inner: Rc<Inner<S>> }
impl<S: Store> Pool<S> {
    pub fn new(store: S,layout: Layout,budget: Budget) -> Result<Self> {
        budget.required(&layout)?;
        Ok(Self {inner:Rc::new(Inner {store,layout,budget,lanes:Cell::new(0),saved:Cell::new(0),failure:RefCell::new(None)})})
    }
    pub fn layout(&self) -> &Layout { &self.inner.layout }
    pub fn used_slots(&self) -> (usize,usize) { (self.inner.lanes.get(),self.inner.saved.get()) }
    pub fn failure(&self) -> Option<String> { self.inner.failure.borrow().clone() }
    pub fn session(&self) -> Result<Session<S>> {
        let mut sizes=self.inner.layout.sizes.clone(); sizes.extend_from_within(..); sizes.push(self.inner.layout.scratch);
        let owners=Owners::new(self.inner.clone(),false,&sizes)?;
        Ok(Session {owners,context:0})
    }
}

struct Owners<S: Store> { pool: Rc<Inner<S>>, saved: bool, buffers: Vec<S::Buffer> }
impl<S: Store> Owners<S> {
    fn new(pool: Rc<Inner<S>>,saved: bool,sizes: &[usize]) -> Result<Self> {
        pool.healthy()?;
        let (count,limit)=if saved {(&pool.saved,pool.budget.saved)} else {(&pool.lanes,pool.budget.lanes)};
        need(count.get()<limit,"session owner credits are exhausted")?; count.set(count.get()+1);
        let mut out=Self {pool,saved,buffers:Vec::new()};
        out.buffers.try_reserve_exact(sizes.len()).map_err(|_| "cannot reserve session owner table")?;
        for &bytes in sizes {
            let buffer=out.pool.store.allocate(bytes)?;
            out.buffers.push(buffer);
            let b=out.buffers.last_mut().unwrap();
            need(out.pool.store.size(b)==bytes,"store allocation differs from admitted extent")?;
            out.pool.store.zero(b)?;
        }
        out.pool.wait()?; Ok(out)
    }
}
impl<S: Store> Drop for Owners<S> {
    fn drop(&mut self) {
        // A failed completion cannot authorize a free. Keep the allocation and
        // its credit until the enclosing device context is retired by its owner.
        if self.pool.healthy().is_err() || self.pool.wait().is_err() {
            std::mem::forget(std::mem::take(&mut self.buffers)); return;
        }
        self.buffers.clear();
        let count=if self.saved {&self.pool.saved} else {&self.pool.lanes}; count.set(count.get()-1);
    }
}

pub struct Session<S: Store> { owners: Owners<S>, context: u64 }
pub struct Saved<S: Store> { owners: Owners<S>, context: u64 }
impl<S: Store> Saved<S> { pub fn context(&self) -> u64 { self.context } }
impl<S: Store> Session<S> {
    pub fn context(&self) -> u64 { self.context }
    pub fn layout(&self) -> &Layout { &self.owners.pool.layout }
    pub fn read(&self,plane: usize) -> Result<Vec<u8>> {
        let pool=&self.owners.pool; pool.healthy()?;
        let p=pool.layout.planes.get(plane).ok_or("session plane index is out of range")?;
        pool.wait()?;
        let mut bytes=pool.ordered(|| pool.store.read(&self.owners.buffers[plane]))?;
        need(bytes.len()==pool.layout.sizes[plane],"store read differs from admitted extent")?;
        bytes.truncate(p.used_bytes(self.context)?); Ok(bytes)
    }
    pub fn begin_append(&mut self) -> Result<Append<'_,S>> {
        let pool=&self.owners.pool; pool.healthy()?;
        need(self.context<pool.layout.capacity,"session append exceeds token capacity")?;
        let count=pool.layout.planes.len();
        let (committed,work)=self.owners.buffers.split_at_mut(count);
        let (staged,scratch)=work.split_at_mut(count);
        pool.ordered(|| {
            for (a,b) in committed.iter().zip(staged) { pool.store.copy(a,b)?; }
            pool.store.zero(&mut scratch[0])
        })?;
        Ok(Append {session:self,written:BTreeSet::new(),failed:false})
    }
    pub fn save(&self) -> Result<Saved<S>> {
        let pool=&self.owners.pool; pool.healthy()?;
        let mut owners=Owners::new(pool.clone(),true,&pool.layout.sizes)?;
        pool.ordered(|| {
            for (a,b) in self.owners.buffers.iter().zip(&mut owners.buffers) { pool.store.copy(a,b)?; } Ok(())
        })?;
        Ok(Saved {owners,context:self.context})
    }
    /// Restore into the append bank. A failed copy cannot replace committed state.
    pub fn restore(&mut self,saved: &Saved<S>) -> Result<()> {
        let pool=&self.owners.pool; pool.healthy()?;
        need(Rc::ptr_eq(pool,&saved.owners.pool),"saved state belongs to a different checkpoint or rank pool")?;
        let count=pool.layout.planes.len();
        pool.ordered(|| {
            for (a,b) in saved.owners.buffers.iter().zip(&mut self.owners.buffers[count..2*count]) { pool.store.copy(a,b)?; } Ok(())
        })?;
        self.swap_banks(); self.context=saved.context; Ok(())
    }
    /// Recurrent state requires a saved boundary. A token count is insufficient.
    pub fn truncate(&mut self,saved: &Saved<S>) -> Result<()> {
        need(saved.context<=self.context,"truncate boundary follows the current context")?; self.restore(saved)
    }
    fn swap_banks(&mut self) {
        let n=self.layout().planes.len(); for i in 0..n { self.owners.buffers.swap(i,i+n); }
    }
}

/// Buffers remain owned by the session. The callback must not retain addresses or
/// start users outside the compute queue and supplied completion object.
pub struct Work<'a,B> { pub committed: &'a [B], pub staged: &'a mut [B], pub scratch: &'a mut B }
pub struct Append<'a,S: Store> { session: &'a mut Session<S>, written: BTreeSet<usize>, failed: bool }
impl<S: Store> Append<'_,S> {
    pub fn context(&self) -> u64 { self.session.context+1 }
    pub fn layout(&self) -> &Layout { self.session.layout() }
    /// Write a complete used plane. Zero capacity padding for stable snapshots.
    pub fn write(&mut self,plane: usize,bytes: &[u8]) -> Result<()> {
        need(!self.failed,"session append has failed")?;
        // A rejected replacement must not publish an earlier complete set.
        self.failed=true;
        let pool=&self.session.owners.pool; pool.healthy()?;
        let p=pool.layout.planes.get(plane).ok_or("session plane index is out of range")?;
        need(bytes.len()==p.used_bytes(self.context())?,"session write differs from used plane extent")?;
        let index=plane+pool.layout.planes.len();
        let buffer=&mut self.session.owners.buffers[index];
        // A replacement write can panic after earlier writes filled every plane.
        // Invalidate the append before submission, then drain before unwinding.
        pool.ordered(|| pool.store.zero(buffer).and_then(|()| pool.store.write(buffer,bytes)))?;
        self.written.insert(plane); self.failed=false; Ok(())
    }
    /// Run an ordered producer and drain all users on success, error, or panic.
    /// The returned indices name fully written planes, including zero padding.
    /// Compute must finish before a host transport consumes its output. That
    /// producer fence belongs inside the callback; this fence retires all users.
    pub fn run<C: Completion>(&mut self,completion: &mut C,
        execute: impl FnOnce(&S,Work<'_,S::Buffer>)->Result<Vec<usize>>) -> Result<()> {
        need(!self.failed,"session append has failed")?;
        let pool=&self.session.owners.pool; pool.healthy()?;
        let count=pool.layout.planes.len();
        let (committed,work)=self.session.owners.buffers.split_at_mut(count);
        let (staged,scratch)=work.split_at_mut(count);
        // Mark before invoking external code. Unwind must never publish a bank.
        self.failed=true;
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(&pool.store,Work {committed,staged,scratch:&mut scratch[0]})));
        // Transport may enqueue its final copy on the compute queue. Drain it
        // first, then drain compute, even when either wait returns an error.
        let transport=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| completion.wait()))
            .unwrap_or_else(|_| Err("transport completion panicked".into()));
        if let Err(e)=&transport { pool.retire(e); }
        let compute=pool.wait();
        let written=match result { Ok(r)=>r,Err(payload)=>std::panic::resume_unwind(payload) };
        let written=written?; transport?; compute?;
        need(written.iter().all(|&i| i<count),"producer returned an invalid state plane")?;
        self.written.extend(written); self.failed=false; Ok(())
    }
    pub fn commit(mut self) -> Result<()> {
        need(!self.failed,"session append has failed")?;
        let pool=&self.session.owners.pool; pool.healthy()?;
        need(self.written.len()==pool.layout.planes.len(),"session append is missing state planes")?;
        pool.wait()?;
        self.session.swap_banks(); self.session.context+=1;
        self.failed=false; Ok(())
    }
    pub fn rollback(self) {}
}
