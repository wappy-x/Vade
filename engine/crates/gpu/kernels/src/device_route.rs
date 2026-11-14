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
/// ROUTE ROWS: the build above for `tokens` rows of one step (`device_route_abi::BuildRows`). It writes one
/// projection descriptor a (phase, token, slot) job and quantizes every job's input. A job is the one-token job of
/// its token: the same table entry, the same quantizer words, its own input row and output row (both in pair order).
/// EXPERT ORDER: a phase's descriptors are listed by (expert, pair), so jobs of one expert are neighbours in the
/// project launch and rows that picked the same expert read its weights together. The list order names no address.
#[inline(always)]
pub(crate) fn build_rows(arguments:crate::device_route_abi::BuildRows) {
    // SAFETY: The host admits the table as for `build` (a bounded equal-shape NVFP4 table with every weight and scale
    // owner retained) and gives: `ids` covering (tokens-1) * stride + picks words; `input` covering tokens rows of cols
    // words (tokens * picks rows for down); `quantized` covering cols/16*17 words a token when shared, else a job;
    // `quantizers` one descriptor a token when shared, else a job; `projects` one descriptor a job; `output` rows words
    // a job. `extents_rows` bounds experts, picks, phases and tokens before any use. A descriptor thread checks every
    // ID of its own token's row (in range, no repeat inside the row) before it indexes the table; every token has at
    // least one such thread, and a quantize block checks the one ID it indexes with. An ID may repeat ACROSS rows: two
    // jobs then read the same immutable weights and write their own outputs. One thread owns one job. Its projection
    // descriptor goes to slot phase * pairs + rank, rank < pairs, inside `projects`; a thread that stores has seen every
    // ID of the launch in range, and (expert, pair) is then a strict order of the pairs, so two threads of one phase never
    // hold one rank and the stores are disjoint. A shared quantizer descriptor is stored by its token's first job alone. Quantize blocks of
    // different quantized rows write disjoint ranges, the contract of nv_quantize::lanes_job, and read nothing a
    // descriptor thread writes. The project launch runs behind this one on the same ordered stream. A trap makes
    // completion fail before any token or state can be published. `raw` and a shared down route trap.
    unsafe {
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x,trap};
        use crate::device_route_abi::{descriptors_rows,extents_rows,job_place,ranks_before};
        let build=arguments.build;const THREADS:usize=crate::nv_quantize::THREADS;
        let Some((_,jobs))=extents_rows(build.experts,build.picks,build.phases,arguments.tokens) else {trap();};
        if _block_dim_x() as usize!=THREADS || build.down>1 || build.shared>1 || build.down!=0 && (build.phases!=1 || build.shared!=0) || build.raw!=0
            || arguments.stride<build.picks {trap();}
        let ids=build.ids as *const f32;let picks=build.picks as usize;let stride=arguments.stride as usize;
        let head=jobs.div_ceil(THREADS);let block=_block_idx_x() as usize;
        if block>=head {
            let cols=(build.table as *const Arguments).read().shape.cols as usize;
            if cols==0 || cols%16!=0 {trap();}
            let per_row=(cols/16).div_ceil(THREADS/16);let (quant,local)=((block-head)/per_row,(block-head)%per_row);
            if quant>=if build.shared!=0 {arguments.tokens as usize} else {jobs} {return;}
            // A shared quantized row belongs to its token: its descriptor is that of the token's first job.
            let job=if build.shared!=0 {quant*picks} else {quant};
            let (phase,token,slot)=job_place(job as u32,build.picks,arguments.tokens);
            let Some(expert)=selected(*ids.add(token as usize*stride+slot as usize),build.experts) else {trap();};
            let entry=(build.table as *const Arguments).add(phase as usize*build.experts as usize+expert as usize).read();
            let (quantizer,_)=descriptors_rows(entry,arguments,job as u32);
            if quantizer.shape.cols as usize!=cols {trap();}
            crate::nv_quantize::lanes_job(quantizer,local);return;
        }
        let job=block*THREADS+_thread_idx_x() as usize;if job>=jobs {return;}
        let (phase,token,slot)=job_place(job as u32,build.picks,arguments.tokens);
        let row=ids.add(token as usize*stride);
        for at in 0..picks {
            let value=*row.add(at);if selected(value,build.experts).is_none() {trap();}
            for prior in 0..at {if *row.add(prior)==value {trap();}}
        }
        let Some(expert)=selected(*row.add(slot as usize),build.experts) else {trap();};
        let entry=(build.table as *const Arguments).add(phase as usize*build.experts as usize+expert as usize).read();
        let (quantizer,projection)=descriptors_rows(entry,arguments,job as u32);
        // EXPERT ORDER: the job's place in its phase's list is the rank of its pair by (expert, pair). Every ID of the
        // launch is checked here before it is compared, so the ranks this thread can reach are those of a strict order.
        let (pairs,pair)=(arguments.tokens as usize*picks,token as usize*picks+slot as usize);let mut rank=0usize;
        for other in 0..pairs {
            let Some(id)=selected(*ids.add(other/picks*stride+other%picks),build.experts) else {trap();};
            if ranks_before(id,other as u32,expert,pair as u32) {rank+=1;}
        }
        (build.projects as *mut Arguments).add(phase as usize*pairs+rank).write(projection);
        if build.shared==0 {(build.quantizers as *mut Arguments).add(job).write(quantizer);}
        else if phase==0 && slot==0 {(build.quantizers as *mut Arguments).add(token as usize).write(quantizer);}
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn device_route_rows(arguments:crate::device_route_abi::BuildRows) {build_rows(arguments)}
