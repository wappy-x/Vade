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
    // hold one rank and the stores are disjoint. THE IDS ONCE: every thread of a descriptor block stages pairs t, t + THREADS, ...
    // of the launch (pairs = tokens * picks, at most MAX_TOKENS * MAX_PICKS by `extents_rows`) as checked expert numbers in a
    // static shared region of 4 * MAX_TOKENS * MAX_PICKS bytes, the only shared allocation of the descriptor path (a quantize
    // block returns before it). Each word has one writer and is read only after a bar.sync that every thread of the block
    // executes once (no descriptor thread returns before it), and the words are those the loads below read before: a
    // staged expert is `selected` of the ID word, and two IDs of one row are equal exactly when their experts are equal
    // (+0.0 and -0.0 are the same ID and the same expert). A shared quantizer descriptor is stored by its token's first job alone. Quantize blocks of
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
        // THE IDS ONCE (DRAFTR 2026-11-17, route bookkeeping): the block loads every ID of the launch once, its threads side
        // by side, and checks each as before. The checks and the EXPERT ORDER below read the staged experts; before, every
        // thread loaded each ID from global memory again, one load after the other, so the build grew with the rows.
        let pairs=arguments.tokens as usize*picks;
        let staged=crate::ptx::shared::<{4*(crate::device_route_abi::MAX_TOKENS*crate::device_route_abi::MAX_PICKS) as usize}>();
        let mut at=_thread_idx_x() as usize;
        while at<pairs {
            let Some(id)=selected(*ids.add(at/picks*stride+at%picks),build.experts) else {trap();};
            crate::ptx::store(staged+(4*at) as u32,f32::from_bits(id));at+=THREADS;
        }
        crate::ptx::sync();
        let expert_of=|pair:usize| crate::ptx::load(staged+(4*pair) as u32).to_bits();
        let job=block*THREADS+_thread_idx_x() as usize;if job>=jobs {return;}
        let (phase,token,slot)=job_place(job as u32,build.picks,arguments.tokens);
        let row=token as usize*picks;
        for at in 0..picks {
            let value=expert_of(row+at);
            for prior in 0..at {if expert_of(row+prior)==value {trap();}}
        }
        let expert=expert_of(row+slot as usize);
        let entry=(build.table as *const Arguments).add(phase as usize*build.experts as usize+expert as usize).read();
        let (quantizer,projection)=descriptors_rows(entry,arguments,job as u32);
        // EXPERT ORDER: the job's place in its phase's list is the rank of its pair by (expert, pair). Every ID of the
        // launch was checked when it was staged, so the ranks this thread can reach are those of a strict order.
        let pair=row+slot as usize;let mut rank=0usize;
        for other in 0..pairs {
            if ranks_before(expert_of(other),other as u32,expert,pair as u32) {rank+=1;}
        }
        (build.projects as *mut Arguments).add(phase as usize*pairs+rank).write(projection);
        if build.shared==0 {(build.quantizers as *mut Arguments).add(job).write(quantizer);}
        else if phase==0 && slot==0 {(build.quantizers as *mut Arguments).add(token as usize).write(quantizer);}
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn device_route_rows(arguments:crate::device_route_abi::BuildRows) {build_rows(arguments)}
/// Grouped NVFP4 tensor-core route builder (`device_route_tc_rows`): block 0 validates every ID (in range, no
/// duplicates within any token's picks), groups `(token, slot)` pairs by active expert into compact `NvPart`,
/// `NvTile` work records and `rowmap` pair indices; blocks >= 1 quantize and pack FP32 input rows directly to E2M1
/// codes and UE4M3 scales (`quantize_packed_with`, identical to `nv_quantize_pack` and `nv_cols_b`).
#[inline(never)]
pub(crate) fn build_tc_rows(arguments:crate::device_route_abi::BuildTcRows) {
    // SAFETY: The host (`Backend::device_route_projection_tc`) admits a bounded equal-shape column-tile NVFP4 table
    // with retained weight/scale owners and checks every buffer extent and 4-byte alignment: `ids` holds
    // `(tokens-1)*stride + picks` words; `input` holds `quant_rows * cols` FP32 words (`tokens` when `shared`, else
    // `tokens * picks`); `codes` holds `quant_rows * (cols / 2)` bytes and `scales` holds `quant_rows * (cols / 16)`
    // bytes; `parts` holds `phases * pairs` `NvPart` records, `work` holds `phases * pairs` `NvTile` records, `rowmap`
    // holds `pairs` `u32` pair indices, and `output` holds `phases * pairs * rows` FP32 words. All owners stay live
    // through stream completion. Block 0 stages `pairs <= MAX_TOKENS * MAX_PICKS` IDs and `experts <= MAX_EXPERTS`
    // histogram/offset arrays in its private static shared region (18,440 B), validates every ID and per-token
    // uniqueness before indexing `table`, writes disjoint `rowmap`/`parts`/`work` entries, and marks trailing work
    // slots with `part = u32::MAX`. Quantize blocks (>= 1) own disjoint 16-column blocks of `codes`/`scales` and read
    // immutable `input`, `ids`, and `table`. Every thread of block 0 reaches both block barriers. Any invalid shape or
    // ID traps before `fp4_cohort_t8` runs on the same ordered stream.
    unsafe {
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x,trap};
        use crate::device_route_abi::{extents_rows,MAX_EXPERTS,MAX_PICKS,MAX_TOKENS};
        use crate::tc_abi::{NvPart,NvTile,NV_QUANTIZE_PACK_THREADS};
        const THREADS:usize=NV_QUANTIZE_PACK_THREADS as usize;
        const MAX_PAIRS:usize=(MAX_TOKENS*MAX_PICKS) as usize;
        const MAX_EXP:usize=MAX_EXPERTS as usize;
        let Some((_,jobs))=extents_rows(arguments.experts,arguments.picks,arguments.phases,arguments.tokens) else {trap();};
        if _block_dim_x() as usize!=THREADS || arguments.down>1 || arguments.shared>1
            || (arguments.down!=0 && (arguments.phases!=1 || arguments.shared!=0))
            || (arguments.down==0 && arguments.shared==0)
            || arguments.stride<arguments.picks
            || (arguments.codes|arguments.scales|arguments.input|arguments.rowmap|arguments.parts|arguments.work|arguments.output)&3!=0 {trap();}
        let first=(arguments.table as *const Arguments).read();
        let (rows,cols)=(first.shape.rows as usize,first.shape.cols as usize);
        if rows==0 || cols==0 || cols%16!=0 {trap();}
        let (picks,tokens,stride)=(arguments.picks as usize,arguments.tokens as usize,arguments.stride as usize);
        let pairs=tokens*picks;
        let quant_rows=if arguments.shared!=0 {tokens} else {pairs};
        let tid=_thread_idx_x() as usize;
        let ids=arguments.ids as *const f32;
        if _block_idx_x()!=0 {
            let blocks=cols/16;let per_row=blocks.div_ceil(THREADS);
            let blk=_block_idx_x() as usize-1;let (qrow,local)=(blk/per_row,blk%per_row);
            if qrow>=quant_rows {return;}
            let b=local*THREADS+tid;if b>=blocks {return;}
            let activation=if arguments.shared!=0 {first.shape.activation} else {
                let (token,slot)=(qrow/picks,qrow%picks);
                let Some(expert)=selected(*ids.add(token*stride+slot),arguments.experts) else {trap();};
                let entry=(arguments.table as *const Arguments).add(expert as usize).read();
                if entry.shape.cols as usize!=cols {trap();}
                entry.shape.activation
            };
            let src=(arguments.input as *const f32).add(qrow*cols+b*16);
            let (lo,hi,scale)=crate::expert_ops::quantize_packed_with(|j|*src.add(j),activation);
            let dst=(arguments.codes as *mut u32).add(qrow*(blocks*2)+b*2);
            *dst=lo;*dst.add(1)=hi;
            *(arguments.scales as *mut u8).add(qrow*blocks+b)=scale as u8;
            return;
        }
        const STAGED:usize=0;
        const ACTIVE:usize=STAGED+MAX_PAIRS;
        const COUNTS:usize=ACTIVE+MAX_PAIRS;
        const OFFSETS:usize=COUNTS+MAX_EXP;
        const TILE_OFF:usize=OFFSETS+MAX_EXP;
        const WRITE_CUR:usize=TILE_OFF+MAX_EXP;
        const META:usize=WRITE_CUR+MAX_EXP;
        const SMEM_BYTES:usize=(META+2)*4;
        let smem=crate::ptx::shared::<SMEM_BYTES>();
        let ld=|w:usize| crate::ptx::load(smem+(4*w) as u32).to_bits();
        let st=|w:usize,v:u32| crate::ptx::store(smem+(4*w) as u32,f32::from_bits(v));
        let experts=arguments.experts as usize;
        let mut i=tid;while i<experts {st(COUNTS+i,0);i+=THREADS;}
        let mut p=tid;
        while p<pairs {
            let Some(id)=selected(*ids.add((p/picks)*stride+(p%picks)),arguments.experts) else {trap();};
            st(STAGED+p,id);p+=THREADS;
        }
        crate::ptx::sync();
        let mut tok=tid;
        while tok<tokens {
            let base=tok*picks;
            for s in 0..picks {
                let v=ld(STAGED+base+s);
                for prior in 0..s {if ld(STAGED+base+prior)==v {trap();}}
            }
            tok+=THREADS;
        }
        crate::ptx::sync();
        if tid==0 {
            for p in 0..pairs {
                let e=ld(STAGED+p) as usize;
                st(COUNTS+e,ld(COUNTS+e)+1);
            }
            let (mut u,mut row_cur,mut tile_cur)=(0usize,0u32,0u32);
            for e in 0..experts {
                let c=ld(COUNTS+e);
                if c!=0 {
                    st(ACTIVE+u,e as u32);
                    st(OFFSETS+e,row_cur);
                    st(WRITE_CUR+e,row_cur);
                    row_cur+=c;
                    st(TILE_OFF+e,tile_cur);
                    tile_cur+=(c+7)/8;
                    u+=1;
                }
            }
            let rowmap=arguments.rowmap as *mut u32;
            for p in 0..pairs {
                let e=ld(STAGED+p) as usize;
                let pos=ld(WRITE_CUR+e);
                st(WRITE_CUR+e,pos+1);
                *rowmap.add(pos as usize)=p as u32;
            }
            st(META,u as u32);st(META+1,tile_cur);
        }
        crate::ptx::sync();
        let (u,nt,phases)=(ld(META) as usize,ld(META+1) as usize,arguments.phases as usize);
        let parts=arguments.parts as *mut NvPart;
        let work=arguments.work as *mut NvTile;
        let mut idx=tid;
        while idx<phases*u {
            let (phase,ui)=(idx/u,idx%u);
            let e=ld(ACTIVE+ui) as usize;
            let entry=(arguments.table as *const Arguments).add(phase*experts+e).read();
            if entry.shape.rows as usize!=rows || entry.shape.cols as usize!=cols {trap();}
            let out_base=arguments.output+(phase*pairs*rows*4) as u64;
            let c=ld(COUNTS+e);
            parts.add(idx).write(NvPart {weight:entry.weight,scales:entry.scales,output:out_base,first:ld(OFFSETS+e),tokens:c,global:entry.shape.global,activation:entry.shape.activation});
            let t0=phase*nt+ld(TILE_OFF+e) as usize;
            let n_tiles=((c+7)/8) as usize;
            let mut t=0;
            while t<n_tiles {work.add(t0+t).write(NvTile {part:idx as u32,tile:t as u32});t+=1;}
            idx+=THREADS;
        }
        let mut w=phases*nt+tid;
        while w<jobs {work.add(w).write(NvTile {part:u32::MAX,tile:0});w+=THREADS;}
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn device_route_tc_rows(arguments:crate::device_route_abi::BuildTcRows) {build_tc_rows(arguments)}
