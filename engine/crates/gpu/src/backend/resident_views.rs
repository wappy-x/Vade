//! Retained immutable activation ranges for the resident decoder.
use super::*;
const _:()={assert!(std::mem::size_of::<crate::kernel_params::HybridCopyRow>()==24);
    assert!(std::mem::offset_of!(crate::kernel_params::HybridCopyRow,source)==0);
    assert!(std::mem::offset_of!(crate::kernel_params::HybridCopyRow,destination)==8);
    assert!(std::mem::offset_of!(crate::kernel_params::HybridCopyRow,words)==16);
    assert!(std::mem::offset_of!(crate::kernel_params::HybridCopyRow,padding)==20);};
impl<'a> DeviceVector<'a> {
    /// Transfer a contiguous allocation to shared ownership without copying it.
    /// Resident values only read this storage after publishing a retained view.
    pub fn into_resident_shared(self)->Result<Self,String> {
        if self.is_segmented() {return Err("resident shared values require contiguous storage".into());}
        if self.buffer.owner.is_some() {return Ok(self);}
        let owner=Rc::new(self.buffer);let buffer=Buffer::view(&owner,0,owner.bytes)?;
        Ok(Self {buffer,len:self.len,managed:self.managed,
            #[cfg(debug_assertions)] projection_scratch:self.projection_scratch})
    }
    /// A read-only range keeps the original allocation alive through all uses.
    /// The decoder never writes to a value after it publishes a range.
    pub fn resident_range(&self,range:std::ops::Range<usize>)->Result<Self,String> {
        let bytes=crate::vector_byte_range(self.len,range)?;
        if bytes.is_empty() || self.is_segmented() {return Err("resident range requires nonempty contiguous storage".into());}
        let owner=self.buffer.owner.as_ref().ok_or("resident range requires shared storage")?;
        let base=usize::try_from(self.buffer.address-owner.address).map_err(|_|"resident range offset exceeds address space")?;
        let offset=base.checked_add(bytes.start).ok_or("resident range offset overflow")?;
        Ok(Self {buffer:Buffer::view(owner,offset,bytes.len())?,len:bytes.len()/4,managed:self.managed,
            #[cfg(debug_assertions)] projection_scratch:self.projection_scratch.clone()})
    }
}
impl<'a> Backend<'a> {
    /// One private workspace allocation, reused only after every published view
    /// has gone. All its old reads and new writes use this backend's ordered
    /// stream. A live alias forces a new allocation; no output is overwritten.
    /// The caller charges the retained high-water size and clears it at the end
    /// of its workspace lifetime. This does not change the vector pool's cap.
    pub fn resident_workspace(&self,words:usize,storage:&mut Option<DeviceVector<'a>>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        if words==0 {return Err("resident workspace requires nonempty storage".into());}
        if let Some(vector)=storage.as_ref() {self.check_vector(vector,vector.len())?;}
        let reusable=storage.as_ref().is_some_and(|vector|vector.len()>=words && !vector.is_segmented()
            && vector.buffer.owner.as_ref().is_some_and(|owner|Rc::strong_count(owner)==1));
        if !reusable {
            // Drop a private older high-water allocation before growing. Any
            // published old view keeps its own owner until its consumers end.
            storage.take();*storage=Some(self.vector(words)?.into_resident_shared()?);
        }
        storage.as_ref().ok_or("resident workspace allocation is absent")?.resident_range(0..words)
    }
    /// Copy row `i` of `sources` (its vector and first word) into the whole of
    /// `destinations[i]`, all rows with one ordered copy launch: the join's kernel
    /// and descriptor table, with explicit destinations. Raw words; no arithmetic.
    /// The caller owns every destination: no live view reads its words. The
    /// destinations must be pairwise disjoint and disjoint from every source row.
    pub fn resident_scatter(&self,sources:&[(&DeviceVector<'_>,usize)],destinations:&mut [DeviceVector<'_>],table:&mut Option<DeviceVector<'a>>)->Result<(),String> {
        self.require_resident_stream()?;
        if sources.len()!=destinations.len() {return Err("resident scatter row count differs".into());}
        let rows=u32::try_from(sources.len()).ok().filter(|&n|n<=u16::MAX as u32).ok_or("resident scatter row count exceeds grid")?;
        // Byte intervals [start,end) of every source row and destination; checked before any write.
        let mut written=Vec::with_capacity(destinations.len());let mut read=Vec::with_capacity(sources.len());let mut longest=0usize;
        for (&(v,first),d) in sources.iter().zip(destinations.iter()) {
            self.check_vector(v,v.len())?;self.check_vector(d,d.len())?;
            if v.is_segmented() || d.is_segmented() {return Err("resident scatter requires contiguous storage".into());}
            let words=d.len();if words==0 || words>u32::MAX as usize {return Err("resident scatter row extent exceeds ABI".into());}
            if first.checked_add(words).is_none_or(|end|end>v.len()) {return Err("resident scatter source row exceeds its vector".into());}
            let source=v.device_address()+(first*4) as u64;
            read.push((source,source+(words*4) as u64));written.push((d.device_address(),d.device_address()+(words*4) as u64));longest=longest.max(words);
        }
        if rows==0 {return Ok(());}
        let mut sorted=written.clone();sorted.sort_unstable();
        if sorted.windows(2).any(|pair|pair[1].0<pair[0].1) {return Err("resident scatter destinations overlap".into());}
        // A destination that overlaps a source row could be read after it is written.
        let overlaps=|(start,end):(u64,u64)|{let at=sorted.partition_point(|d|d.1<=start);at<sorted.len() && sorted[at].0<end};
        if read.iter().any(|&row|overlaps(row)) {return Err("resident scatter destination overlaps a source".into());}
        if let Some(v)=table.as_ref() {self.check_vector(v,v.len())?;if v.is_segmented() {return Err("resident copy table must be contiguous".into());}}
        self.require_arm_kernels(&["flash_hybrid_copy"])?;
        let extent=rows as usize*6;let mut bytes=Vec::with_capacity(extent*4);
        for (r,w) in read.iter().zip(&written) {
            bytes.extend_from_slice(&r.0.to_ne_bytes());bytes.extend_from_slice(&w.0.to_ne_bytes());
            bytes.extend_from_slice(&(((w.1-w.0)/4) as u32).to_ne_bytes());bytes.extend_from_slice(&0u32.to_ne_bytes());
        }
        if bytes.len()!=extent*4 {return Err("resident scatter row table differs".into());}
        let grid=[(longest as u32).div_ceil(128),rows,1,128,0];
        if self.record_resident_tabled(crate::resident_launch::Kind::Copy,grid,&bytes,|address|crate::resident_launch_encode::copy(
            crate::kernel_params::HybridCopy {source:ptr::null(),destination:ptr::null_mut(),words:0,padding:0,rows:address}))? {self.resident_completion()?;return Ok(());}
        if table.as_ref().is_none_or(|v|v.len()<extent) {*table=Some(self.vector(extent)?);}
        let table=table.as_mut().unwrap();
        let (table_start,table_end)=(table.device_address(),table.device_address()+(extent*4) as u64);
        if read.iter().chain(&written).any(|&(start,end)|start<table_end && table_start<end) {return Err("resident copy table overlaps a scatter row".into());}
        self.write_buffer(&table.buffer,0,&bytes)?;
        let mut p=crate::kernel_params::HybridCopy {source:ptr::null(),destination:ptr::null_mut(),words:0,padding:0,rows:table.device_address()};
        // SAFETY: Each descriptor names one checked same-context contiguous source
        // range inside its vector and the whole of one checked same-context contiguous
        // destination. Destinations are pairwise disjoint, disjoint from every source
        // row and from the table (checked above), so no word is written twice or read
        // after it is written. The caller holds every destination exclusively. Row
        // count and word extents fit the kernel grid and ABI; the table has the exact
        // native HybridCopyRow layout; by-value arguments are copied at submission.
        // Sources, destinations' storage and the table stay live as for
        // resident_join: an owned queue defers their drops until its real drain and
        // ordinary callers wait before return; descriptor uploads use retained pinned
        // staging on this same ordered stream. This also covers a failed or partially
        // submitted launch.
        let launched=unsafe {self.launch_flash_rows("flash_hybrid_copy",(longest as u32).div_ceil(128),rows,128,0,&mut [parameter(&mut p)])};
        let completed=self.resident_completion();launched.and(completed)
    }
    /// Two values that are both complete at the same point need one wait, not two.
    pub fn read_resident_pair(&self,first:&DeviceVector<'_>,second:&DeviceVector<'_>)->Result<(Vec<f32>,Vec<f32>),String> {
        let mut values=self.read_vector_refs(&[first,second])?;let second=values.pop().ok_or("pair read lost a value")?;Ok((values.pop().ok_or("pair read lost a value")?,second))
    }
    /// Concatenate immutable prefixes with one ordered copy launch. Reuse the
    /// descriptor allocation. Ordered uploads preserve prior descriptor contents.\n    /// Queued drops defer contiguous allocations until a real stream drain.
    /// The output has at least `reserve` words (0: just the joined words). Only the
    /// joined words are written; the caller publishes only that prefix. A caller
    /// whose joins differ in length can so ask for one size that a freed buffer matches.
    pub fn resident_join(&self,parts:&[(&DeviceVector<'_>,usize)],table:&mut Option<DeviceVector<'a>>,reserve:usize)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let mut words=0usize;let mut rows=0u32;let mut longest=0usize;
        for &(v,n) in parts {
            self.check_vector(v,v.len())?;
            if v.is_segmented() || n>v.len() {return Err("resident join prefix differs".into());}
            words=words.checked_add(n).filter(|&n|n<=u32::MAX as usize).ok_or("resident join extent exceeds ABI")?;
            if n>0 {rows=rows.checked_add(1).filter(|&n|n<=u16::MAX as u32).ok_or("resident join row count exceeds grid")?;longest=longest.max(n);}
        }
        if let Some(v)=table.as_ref() {self.check_vector(v,v.len())?;if v.is_segmented() {return Err("resident copy table must be contiguous".into());}}
        let output=self.vector(words.max(reserve).max(1))?;
        if rows==0 {return Ok(output);}
        self.require_arm_kernels(&["flash_hybrid_copy"])?;
        // The copy grid is rows times the blocks of the longest row. When parts
        // differ in length, cut them into rows of one length so no block is idle.
        // The same words reach the same places.
        let smallest=parts.iter().map(|part|part.1).filter(|&n|n>0).min().unwrap_or(longest);
        let (mut cut,mut pieces)=(longest,rows as usize);
        if longest>smallest {let mut size=smallest.div_ceil(128)*128;
            loop {let count=parts.iter().map(|part|part.1.div_ceil(size)).sum::<usize>();
                if count<=8192 {if size.div_ceil(128).saturating_mul(count)<longest.div_ceil(128).saturating_mul(rows as usize) {cut=size;pieces=count;}break;}
                size=size.checked_mul(2).ok_or("resident join row size overflow")?;}
        }
        let rows=u32::try_from(pieces).ok().filter(|&n|n<=u16::MAX as u32).ok_or("resident join row count exceeds grid")?;let longest=longest.min(cut);
        let extent=rows as usize*6;
        let mut bytes=Vec::with_capacity(extent*4);let mut offset=0usize;
        for &(v,n) in parts {let mut done=0usize;
            while done<n {let words=(n-done).min(cut);
                bytes.extend_from_slice(&(v.device_address()+(done*4) as u64).to_ne_bytes());
                bytes.extend_from_slice(&(output.device_address()+((offset+done)*4) as u64).to_ne_bytes());
                bytes.extend_from_slice(&(words as u32).to_ne_bytes());bytes.extend_from_slice(&0u32.to_ne_bytes());done+=words;}
            offset+=n;
        }
        if bytes.len()!=extent*4 {return Err("resident join row table differs".into());}
        // A recorded join keeps its rows in the recorder's private area: the same
        // kernel, grid and row bytes, with no upload of its own between kernels.
        let grid=[(longest as u32).div_ceil(128),rows,1,128,0];
        if self.record_resident_tabled(crate::resident_launch::Kind::Copy,grid,&bytes,|address|crate::resident_launch_encode::copy(
            crate::kernel_params::HybridCopy {source:ptr::null(),destination:ptr::null_mut(),words:0,padding:0,rows:address}))? {self.resident_completion()?;return Ok(output);}
        if table.as_ref().is_none_or(|v|v.len()<extent) {*table=Some(self.vector(extent)?);}
        let table=table.as_mut().unwrap();
        let table_start=table.device_address();let table_end=table_start+(extent*4) as u64;
        for &(v,n) in parts {let start=v.device_address();let end=start+(n*4) as u64;
            if n>0 && start<table_end && table_start<end {return Err("resident copy table overlaps a source".into());}}
        self.write_buffer(&table.buffer,0,&bytes)?;
        let mut p=crate::kernel_params::HybridCopy {source:ptr::null(),destination:ptr::null_mut(),words:0,padding:0,rows:table.device_address()};
        // SAFETY: Each descriptor is a range inside one checked same-context
        // contiguous source prefix and a disjoint range of the private output. Row count and word
        // extents fit the kernel grid and ABI. The table has its exact native
        // HybridCopyRow layout. By-value arguments are copied at submission.
        // Sources, table and output stay live: an owned queue defers their drops
        // until its real drain; ordinary callers wait before return. Descriptor
        // uploads use retained pinned staging on this same ordered stream.
        // These guarantees also cover a failed or partially submitted launch.
        let launched=unsafe {self.launch_flash_rows("flash_hybrid_copy",(longest as u32).div_ceil(128),rows,128,0,&mut [parameter(&mut p)])};
        let completed=self.resident_completion();launched.and(completed)?;Ok(output)
    }
}
