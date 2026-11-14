//! An upper bound for this probe, not a measured device-byte result.
use spark_gpu_probe::{staging,text_edges::Layout};

#[derive(Debug)]
pub(super) struct Budget {pub device:usize,pub pinned:usize,pub required:usize}
fn add(a:usize,b:usize)->Result<usize,String>{a.checked_add(b).ok_or_else(||"text probe budget overflow".into())}
fn fp32(n:usize)->Result<usize,String>{n.checked_mul(4).ok_or_else(||"text probe vector size overflow".into())}
impl Budget {
    pub fn new(layout:Layout,arena:usize)->Result<Self,String>{
        layout.ranges(arena)?;
        let s=layout.shape;
        // The arena includes the original norm. The resident edge also keeps
        // its lossless FP32 copy. Count all vectors, including temporary IDs,
        // as live together. Each loop iteration completes before the next.
        // The probe does not configure the vector pool or capture a graph.
        let mut device=arena;
        for n in [s.hidden(),s.streams(),s.residual(),s.hidden(),s.vocab()] {
            device=add(device,fp32(n)?)?;
        }
        // FileUploader::double_buffered owns its fixed reader slots.
        // Backend::reserve_upload rounds transfers to 64 bytes and grows to
        // a power of two. Its old and new pinned buffers can coexist.
        let transfer=fp32(s.hidden().max(s.streams()))?;
        let rounded=add(transfer,63)?/64*64;
        let capacity=rounded.checked_next_power_of_two().ok_or("text probe staging size overflow")?;
        let mut pinned=add(staging::UPLOAD_BYTES,staging::WEIGHT_ARENA_BYTES)?;
        if capacity>staging::UPLOAD_BYTES {pinned=add(pinned,capacity)?;}
        let required=add(device,pinned)?;
        Ok(Self {device,pinned,required})
    }
    pub fn admit(&self,available:usize)->Result<(),String>{
        if self.required>available {return Err(format!(
            "text probe needs {} bytes: device {} bytes; pinned {} bytes. Available: {available} bytes. Use a host with more free memory.",
            self.required,self.device,self.pinned));}
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_gpu_probe::text_edges::{Shape,NormStorage};
    fn layout(h:u64,v:u64,s:u64,norm_storage:NormStorage)->(Layout,usize){
        let shape=Shape::new(h,v,s).unwrap();let [embedding,head,scales]=shape.weight_bytes().unwrap();
        let norm=(embedding+head+scales).div_ceil(norm_storage.bytes())*norm_storage.bytes();
        (Layout {shape,embedding:0,head:embedding,scales:embedding+head,norm,norm_storage,epsilon:1e-5},
            norm+h as usize*norm_storage.bytes())
    }
    #[test]
    fn saved_checkpoint_shape_fits_derived_budget(){
        // Saved run1.log asks for this arena. host/IDENTITY records the shape.
        let (l,arena)=layout(4096,154880,4,NormStorage::F32);
        assert_eq!(arena,1_923_006_464);
        let b=Budget::new(l,arena).unwrap();
        assert_eq!(b.device,1_923_724_304);assert_eq!(b.pinned,71_303_168);
        assert_eq!(b.required,1_995_027_472);
        b.admit(b.required).unwrap();assert!(b.admit(64<<20).is_err());
        println!("saved_shape arena={arena} device={} pinned={} required={}",b.device,b.pinned,b.required);
    }
    #[test]
    fn small_shapes_and_stored_norm_types(){
        for norm in [NormStorage::Bf16,NormStorage::F32] {
            for (h,v,s) in [(32,1,1),(96,17,2),(256,9,7)] {
                let (l,arena)=layout(h,v,s,norm);let b=Budget::new(l,arena).unwrap();
                assert_eq!(b.device,arena+4*(h as usize*2+s as usize+h as usize*s as usize+v as usize));
                assert_eq!(b.pinned,staging::UPLOAD_BYTES+staging::WEIGHT_ARENA_BYTES);
                b.admit(b.required).unwrap();assert!(b.admit(b.required-1).is_err());
            }
        }
    }
    #[test]
    fn staging_growth_counts_both_buffers(){
        let (l,arena)=layout(1<<21,1,1,NormStorage::F32);let b=Budget::new(l,arena).unwrap();
        assert_eq!(b.pinned,staging::UPLOAD_BYTES+staging::WEIGHT_ARENA_BYTES+(1<<23));
    }
    #[test]
    fn overflow_and_invalid_extents_refuse(){
        assert!(fp32(usize::MAX).is_err());assert!(add(usize::MAX,1).is_err());
        let (mut l,arena)=layout(32,1,1,NormStorage::Bf16);
        assert!(Budget::new(l,usize::MAX).unwrap_err().contains("overflow"));
        assert!(Budget::new(l,arena-1).is_err());
        l.norm=usize::MAX-1;assert!(Budget::new(l,usize::MAX).is_err());
    }
    #[test]
    fn excessive_shape_and_unavailable_memory_refuse(){
        let (l,arena)=layout(1<<20,1<<20,1,NormStorage::F32);let b=Budget::new(l,arena).unwrap();
        assert!(b.admit(128<<30).is_err());assert!(b.admit(0).is_err());
    }
}
