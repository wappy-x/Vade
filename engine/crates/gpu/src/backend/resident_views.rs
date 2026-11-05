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
    /// Concatenate immutable prefixes with one ordered copy launch. Reuse the
    /// descriptor allocation. Ordered uploads preserve prior descriptor contents.\n    /// Queued drops defer contiguous allocations until a real stream drain.
    pub fn resident_join(&self,parts:&[(&DeviceVector<'_>,usize)],table:&mut Option<DeviceVector<'a>>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let mut words=0usize;let mut rows=0u32;let mut longest=0usize;
        for &(v,n) in parts {
            self.check_vector(v,v.len())?;
            if v.is_segmented() || n>v.len() {return Err("resident join prefix differs".into());}
            words=words.checked_add(n).filter(|&n|n<=u32::MAX as usize).ok_or("resident join extent exceeds ABI")?;
            if n>0 {rows=rows.checked_add(1).filter(|&n|n<=u16::MAX as u32).ok_or("resident join row count exceeds grid")?;longest=longest.max(n);}
        }
        if let Some(v)=table.as_ref() {self.check_vector(v,v.len())?;if v.is_segmented() {return Err("resident copy table must be contiguous".into());}}
        let output=self.vector(words.max(1))?;
        if rows==0 {return Ok(output);}
        self.require_arm_kernels(&["flash_hybrid_copy"])?;
        let extent=rows as usize*6;
        if table.as_ref().is_none_or(|v|v.len()<extent) {*table=Some(self.vector(extent)?);}
        let table=table.as_mut().unwrap();
        let table_start=table.device_address();let table_end=table_start+(extent*4) as u64;
        for &(v,n) in parts {let start=v.device_address();let end=start+(n*4) as u64;
            if n>0 && start<table_end && table_start<end {return Err("resident copy table overlaps a source".into());}}
        let mut bytes=Vec::with_capacity(extent*4);let mut offset=0usize;
        for &(v,n) in parts {if n==0 {continue;}
            bytes.extend_from_slice(&v.device_address().to_ne_bytes());
            bytes.extend_from_slice(&(output.device_address()+(offset*4) as u64).to_ne_bytes());
            bytes.extend_from_slice(&(n as u32).to_ne_bytes());bytes.extend_from_slice(&0u32.to_ne_bytes());offset+=n;
        }
        self.write_buffer(&table.buffer,0,&bytes)?;
        let mut p=crate::kernel_params::HybridCopy {source:ptr::null(),destination:ptr::null_mut(),words:0,padding:0,rows:table.device_address()};
        // SAFETY: Each descriptor is one checked same-context contiguous source
        // prefix and a disjoint range of the private output. Row count and word
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
