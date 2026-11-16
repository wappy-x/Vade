//! Column-tile matrices of a weight load, with no plane copy and no wait a matrix on the upload thread.
//!
//! The one-buffer upload copies every plane on the upload thread into the shared page-locked staging buffer, makes two
//! device allocations a matrix and waits for the stream after every matrix. Here a matrix takes one page-locked slab
//! of a small ring. A copy thread (`load_rows::with_stagers`) fills the slab. The upload thread only queues the two
//! device copies from the slab and the repack kernel, in name order. The device bytes are the same: the same planes
//! reach the same scratch vectors, and the same kernel writes the final vectors.
//!
//! Order on the device: the copies of matrix k, its repack and the copies of matrix k+1 are on one ordered stream, so
//! the repack of k reads the scratch before k+1 overwrites it. Order on the host: a slab has its own event
//! (`PromptBuffer`); a copy thread writes a slab only after the device has read its last matrix.
use super::*;
use load_rows::{SlabRing,SLABS,DEVICE_SLABS,SLAB_BYTES_MAX};
use spark_gpu_probe::driver::PromptBuffer;
use std::{collections::VecDeque,time::Instant};

/// The two row-major scratch vectors of a load (codes, scales).
pub(super) type Scratch<'d>=Option<(DeviceVector<'d>,DeviceVector<'d>)>;
/// A matrix whose planes are in slab `slab` (codes, then scales; padded byte counts), not yet queued for the device.
struct Staged {name:String,rows:Rows,cols:usize,shape:Shape,slab:usize,codes:usize,scales:usize}
#[derive(Default)]
pub(super) struct TileStream<'d> {slabs:Vec<PromptBuffer<'d>>,slab_bytes:usize,ring:Option<SlabRing>,staged:VecDeque<Staged>,
    /// The largest plane that took a slab. The shared staging buffer ends the load as large as the one-buffer upload leaves it.
    largest_plane:usize,
    /// Printed in RESIDENT LOAD SLABS: matrices staged, final-vector allocations, and the upload thread's nanoseconds in
    /// the wait for a free slab, the wait for a filled slab, the allocations, and the copy and repack calls.
    pub(super) matrices:u64,pub(super) allocations:u64,pub(super) slab_wait_ns:u128,pub(super) pack_wait_ns:u128,pub(super) vector_ns:u128,pub(super) queue_ns:u128,
    /// The largest ring of the load: slabs and bytes a slab.
    pub(super) peak:(usize,usize)}
/// True when a column-tile matrix of these plane bytes takes a slab.
pub(super) fn admits(codes:usize,scales:usize)->bool {
    codes>0 && scales>0 && codes.checked_next_multiple_of(4).zip(scales.checked_next_multiple_of(4)).and_then(|(codes,scales)|codes.checked_add(scales)).is_some_and(|bytes|bytes<=SLAB_BYTES_MAX)
}
impl<'d> TileStream<'d> {
    /// Takes the planes of one column-tile matrix: `u.weight[..codes]` and `u.scales[scale_start..scale_start+scales]`.
    /// A copy thread writes them to a slab. The matrix enters `matrices` when a later call queues it for the device.
    #[allow(clippy::too_many_arguments)] // one matrix: its place, its shape and its planes
    pub(super) fn stage<'b>(&mut self,b:&'b Backend<'d>,scratch:&mut Scratch<'d>,matrices:&mut BTreeMap<String,Matrix<'b,'d>>,name:&str,rows:Rows,cols:usize,shape:Shape,
        u:engine_model::plan::admission::UploadWeight,codes:usize,scale_start:usize,scales:usize)->Result<()> {
        let (code_bytes,scale_bytes)=(codes.div_ceil(4)*4,scales.div_ceil(4)*4);let need=code_bytes+scale_bytes;
        if need>self.slab_bytes {
            // A larger matrix than any before: the staged matrices leave their slabs, then larger slabs take their place.
            self.drain(b,scratch,matrices)?;self.slabs.clear();self.slab_bytes=need;
            if let Some(ring)=self.ring.as_mut() {ring.restart()?;}
        }
        if self.ring.is_none() {self.ring=Some(SlabRing::new(SLABS,DEVICE_SLABS)?);}
        // The oldest matrix goes to the device while the ring keeps only the slabs the device may still read.
        while self.ring.as_ref().is_some_and(SlabRing::full) {self.queue_oldest(b,scratch,matrices)?;}
        let slab=self.ring.as_mut().ok_or("resident slab ring is absent")?.take()?;
        if slab==self.slabs.len() {self.slabs.push(b.prompt_buffer(self.slab_bytes)?);
            if self.slabs.len()*self.slab_bytes>self.peak.0*self.peak.1 {self.peak=(self.slabs.len(),self.slab_bytes);}
        }
        let wait=Instant::now();
        // `start_pack` waits for the slab's event: the device has read the matrix that held this slab before.
        self.slabs.get_mut(slab).ok_or("resident slab is absent")?.start_pack(false,move |bytes| {
            let code=u.weight.get(..codes).ok_or("resident code plane is short")?;
            let scale=scale_start.checked_add(scales).and_then(|end|u.scales.get(scale_start..end)).ok_or("resident scale plane is short")?;
            load_rows::pack_planes(bytes,code,scale).map(|_|())
        })?;
        self.slab_wait_ns+=wait.elapsed().as_nanos();self.matrices+=1;self.largest_plane=self.largest_plane.max(code_bytes).max(scale_bytes);
        self.staged.push_back(Staged {name:name.to_owned(),rows,cols,shape,slab,codes:code_bytes,scales:scale_bytes});Ok(())
    }
    /// Queues the oldest staged matrix for the device. This runs while a later matrix loads, or when the load ends, so
    /// an error here names the staged matrix itself (`load_rows::delayed`), not the matrix that loads.
    fn queue_oldest<'b>(&mut self,b:&'b Backend<'d>,scratch:&mut Scratch<'d>,matrices:&mut BTreeMap<String,Matrix<'b,'d>>)->Result<()> {
        let Some(staged)=self.staged.pop_front() else {return Ok(());};
        match self.queue(b,scratch,&staged) {
            Ok(matrix)=>{matrices.insert(staged.name,matrix);Ok(())},
            Err(error)=>Err(load_rows::delayed(&staged.name,&error)),
        }
    }
    /// Two device copies from the matrix's slab to the scratch, then the repack to its final vectors.
    fn queue<'b>(&mut self,b:&'b Backend<'d>,scratch:&mut Scratch<'d>,staged:&Staged)->Result<Matrix<'b,'d>> {
        let Staged {rows,cols,shape,slab,codes,scales,..}=*staged;
        self.ring.as_mut().ok_or("resident slab ring is absent")?.queued()?;
        let wait=Instant::now();let slot=self.slabs.get_mut(slab).ok_or("resident slab is absent")?;slot.finish_pack()?;
        self.pack_wait_ns+=wait.elapsed().as_nanos();let queue=Instant::now();
        if !scratch.as_ref().is_some_and(|(a,c)|a.len()*4>=codes && c.len()*4>=scales) {
            *scratch=None;b.synchronize()?;*scratch=Some((b.vector(codes/4)?,b.vector(scales/4)?));
        }
        let (from_codes,from_scales)=scratch.as_mut().ok_or("column-tile scratch is absent")?;
        b.write_prompt_bytes(from_codes,slot,0,codes)?;b.write_prompt_bytes(from_scales,slot,codes,scales)?;
        // Keep the one-buffer loader's allocation order: scratch growth first, then separate
        // code and scale allocations for this matrix. No later matrix allocates ahead of it.
        let started=Instant::now();
        let (mut weight,mut scale)=(b.vector(codes/4)?,b.vector(scales/4)?);
        self.allocations+=2;self.vector_ns+=started.elapsed().as_nanos();
        // No wait: the next matrix's copies to the scratch are behind this repack on the stream.
        b.nv_tile_repack(from_codes,from_scales,&mut weight,&mut scale,rows.count,cols)?;
        self.queue_ns+=queue.elapsed().as_nanos();
        Ok(Matrix {rows,cols,storage:Storage::Packed {shape,weight:weight.into_resident_shared()?,scales:scale.into_resident_shared()?},absorbed:None})
    }
    /// One line for a load that staged a matrix: the largest ring, the backend's page-locked peak, and where the upload thread spent its time.
    pub(super) fn report(&self,rank:u32,b:&Backend<'d>) {
        if self.matrices==0 {return;}
        eprintln!("RESIDENT LOAD SLABS rank={rank} slabs={} slab_bytes={} ring_pinned_bytes={} peak_pinned_staging_bytes={} matrices={} vector_allocations={} slab_wait_ns={} pack_wait_ns={} vector_allocation_ns={} copy_repack_queue_ns={}",
            self.peak.0,self.peak.1,self.peak.0*self.peak.1,b.peak_pinned_staging_bytes(),self.matrices,self.allocations,self.slab_wait_ns,self.pack_wait_ns,self.vector_ns,self.queue_ns);
    }
    pub(super) fn drain<'b>(&mut self,b:&'b Backend<'d>,scratch:&mut Scratch<'d>,matrices:&mut BTreeMap<String,Matrix<'b,'d>>)->Result<()> {
        while !self.staged.is_empty() {self.queue_oldest(b,scratch,matrices)?;}Ok(())
    }
    /// Ends the stream: every staged matrix is queued, the slabs are freed (each after its last device copy), and the
    /// shared staging buffer is as large as the one-buffer upload of the same planes leaves it, so the page-locked
    /// bytes after a load do not depend on which upload a matrix took.
    pub(super) fn finish<'b>(&mut self,b:&'b Backend<'d>,scratch:&mut Scratch<'d>,matrices:&mut BTreeMap<String,Matrix<'b,'d>>)->Result<()> {
        self.drain(b,scratch,matrices)?;
        if self.slabs.is_empty() {return Ok(());}
        self.slabs.clear();self.slab_bytes=0;self.ring=None;
        b.reserve_exchange(self.largest_plane.div_ceil(4))
    }
}
