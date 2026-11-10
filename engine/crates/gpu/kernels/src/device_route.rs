use crate::device_route_abi::{Build,descriptors,extents,selected};
use crate::expert_ops::Arguments;
#[inline(always)]
pub(crate) fn build(arguments:Build) {
    // SAFETY: The host admits a bounded equal-shape NVFP4 table and retains every
    // weight and scale owner with that table. IDs cover picks words. The input
    // covers one row or picks complete rows for down. Each private output and
    // descriptor array covers its checked job count. All IDs and duplicates are
    // checked before indexing the table. One lane owns one original route slot
    // in one phase, so all descriptor and output ranges are disjoint. The next
    // quantize/project launches run on this same ordered stream. A trap makes
    // completion fail before any token or state can be published.
    unsafe {
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x,trap};
        let Some((_,jobs))=extents(arguments.experts,arguments.picks,arguments.phases) else {trap();};
        if _block_dim_x()!=32 || _block_idx_x()!=0 || arguments.down>1 || arguments.shared>1 || arguments.down!=0 && arguments.phases!=1 {trap();}
        let job=_thread_idx_x() as usize;if job>=jobs {return;}
        let ids=arguments.ids as *const f32;
        for slot in 0..arguments.picks as usize {
            let value=*ids.add(slot);if selected(value,arguments.experts).is_none() {trap();}
            for prior in 0..slot {if *ids.add(prior)==value {trap();}}
        }
        let slot=job%arguments.picks as usize;let phase=job/arguments.picks as usize;
        let Some(expert)=selected(*ids.add(slot),arguments.experts) else {trap();};
        let entry=(arguments.table as *const Arguments).add(phase*arguments.experts as usize+expert as usize).read();
        let (quantizer,projection)=descriptors(entry,arguments,job as u32);
        (arguments.projects as *mut Arguments).add(job).write(projection);
        *(arguments.projects as *mut u32).add(job*24+23)=0;
        if arguments.shared==0 || job==0 {let quant=if arguments.shared!=0 {0} else {job};
            (arguments.quantizers as *mut Arguments).add(quant).write(quantizer);*(arguments.quantizers as *mut u32).add(quant*24+23)=0;}
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn device_route_build(arguments:Build) {build(arguments)}
