use crate::device_route_abi::{Build,descriptors,extents,raw_admitted,selected};
use crate::expert_ops::Arguments;
#[inline(always)]
pub(crate) fn build(arguments:Build) {
    // SAFETY: The host admits a bounded equal-shape NVFP4 table and retains every
    // weight and scale owner with that table. IDs cover picks words. The input
    // covers one row or picks complete rows for down. Each private output and
    // descriptor array covers its checked job count. All IDs and duplicates are
    // checked before indexing the table. One lane owns one original route slot
    // in one phase, so all descriptor and output ranges are disjoint. The next
    // project launch runs on this same ordered stream. A trap makes
    // completion fail before any token or state can be published.
    // ROUTE-PREPARE: the same launch quantizes the inputs. Block 0 writes the descriptors as before. Block b>0 is
    // quantize block (b-1) % per_job of quantize job (b-1) / per_job, per_job = ceil(cols/128): it reads that job's
    // route slot, checks the ID as block 0 does, and computes the job's Quantize descriptor with the function block 0
    // uses (device_route_abi::descriptors), so it holds the words block 0 stores and reads nothing block 0 writes.
    // The host admits cols as a multiple of 512 and equal in every table entry (device_route::admitted; a differing
    // entry traps here), the input of cols words a job and the private quantized owner of cols/16*17 words a job,
    // which is the contract of nv_quantize::lanes_job. Jobs have disjoint quantized ranges. A block past the last
    // job returns. The quantized words are read only by the project launch behind this one.
    // A raw route (the pair split) changes only the output address a job's descriptor names: rows * raw words a job,
    // which the host allocates (Backend::device_route_projection). Any other `raw` traps.
    unsafe {
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x,trap};
        let Some((_,jobs))=extents(arguments.experts,arguments.picks,arguments.phases) else {trap();};
        if _block_dim_x() as usize!=crate::nv_quantize::THREADS || arguments.down>1 || arguments.shared>1 || arguments.down!=0 && arguments.phases!=1
            || !raw_admitted(arguments.raw,arguments.down) {trap();}
        let ids=arguments.ids as *const f32;
        if _block_idx_x()!=0 {
            let cols=(arguments.table as *const Arguments).read().shape.cols as usize;
            if cols==0 || cols%16!=0 {trap();}
            let per_job=(cols/16).div_ceil(crate::nv_quantize::THREADS/16);let block=_block_idx_x() as usize-1;
            let (job,local)=(block/per_job,block%per_job);
            if job>=if arguments.shared!=0 {1} else {jobs} {return;}
            let slot=job%arguments.picks as usize;let phase=job/arguments.picks as usize;
            let Some(expert)=selected(*ids.add(slot),arguments.experts) else {trap();};
            let entry=(arguments.table as *const Arguments).add(phase*arguments.experts as usize+expert as usize).read();
            let (quantizer,_)=descriptors(entry,arguments,job as u32);
            if quantizer.shape.cols as usize!=cols {trap();}
            crate::nv_quantize::lanes_job(quantizer,local);return;
        }
        let job=_thread_idx_x() as usize;if job>=jobs {return;}
        for slot in 0..arguments.picks as usize {
            let value=*ids.add(slot);if selected(value,arguments.experts).is_none() {trap();}
            for prior in 0..slot {if *ids.add(prior)==value {trap();}}
        }
        let slot=job%arguments.picks as usize;let phase=job/arguments.picks as usize;
        let Some(expert)=selected(*ids.add(slot),arguments.experts) else {trap();};
        let entry=(arguments.table as *const Arguments).add(phase*arguments.experts as usize+expert as usize).read();
        let (quantizer,projection)=descriptors(entry,arguments,job as u32);
        // `Arguments` has no padding: its last word (word 23) is the shape's `role`, the word function of the matrix.
        // The whole-struct write stores it. (Before the role was a named word this word was padding and was cleared
        // here; clearing it now would turn every second matrix into a first one in the batch and trap the decode kernel.)
        (arguments.projects as *mut Arguments).add(job).write(projection);
        if arguments.shared==0 || job==0 {let quant=if arguments.shared!=0 {0} else {job};
            (arguments.quantizers as *mut Arguments).add(quant).write(quantizer);}
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn device_route_build(arguments:Build) {build(arguments)}
