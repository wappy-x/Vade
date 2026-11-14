//! NVFP4 activation quantization with one lane per element. Same arithmetic as the `Op::Quantize` branch of
//! `expert_ops::unit` (one thread per 16-value block): the block maximum of the BF16-rounded magnitudes, the block
//! scale `nearest_e4m3((maximum / activation) / 6)`, and per element `nearest_e2m1((|round(v)| / activation) / block)`
//! with the sign of the original value, written as the decoded E2M1 value; word 16 of the block is the decoded scale.
//! What changes is who does the work: the old launch gave each 16-value block to ONE thread (a binary search plus 16
//! element searches in series, 32-thread blocks, a handful of blocks of work for a 4096-wide row). Here 16 lanes own
//! one block, one lane per element, and the maximum crosses the 16 lanes by a butterfly (a maximum of non-negative
//! values is exact in any order). A 128-thread block holds eight quantization blocks.
use crate::expert_ops::{Arguments,LaunchArguments,Op};

/// Threads per block; `expert_operation` selects this body for `Op::Quantize` by the block size.
pub const THREADS:usize=128;

#[inline(always)]
pub(crate) fn lanes_owned(launch:LaunchArguments) {lanes(&launch,None)}
/// One 128-thread block of one Quantize job whose descriptor the caller computed (`device_route::build`, where the
/// descriptor comes from the route table and not from a descriptor array): `local` is the block's index in the job.
#[inline(always)]
pub(crate) fn lanes_job(job:Arguments,local:usize) {lanes(&LaunchArguments {single:job,batch:0,count:1},Some(local))}
#[inline(always)]
pub(crate) fn lanes(launch:&LaunchArguments,at:Option<usize>) {
    // SAFETY: The host checks the descriptor (`quantize_lanes_admitted`: Quantize, cols a multiple of 16 up to the swept
    // limit), that the input owner holds cols words and the output owner cols/16*17 words, both contiguous and live
    // through stream completion, and that the output is private (raw stores before any read; the host skips its fill, fill_free). Input is
    // immutable. Thread t of block b owns element t%16 of quantization block b*8+t/16: it reads that one input word and
    // writes that one output word (and word 16 for element 0) only when the block index is below cols/16, so every
    // output word has exactly one writing thread. A batch (`launch.batch` nonzero) is a host-checked table of `count`
    // equal-width Quantize jobs with disjoint private outputs; block b belongs to job b / ceil(cols/128) and the
    // whole block returns when that job does not exist. `at` (Some) names the block's index inside the one job
    // `launch.single` in place of the launch's block index: the route build passes it for its quantize blocks, with a
    // descriptor it computed from host-checked route arguments (device_route::build states that contract), so the
    // same ownership holds for block `at` of that job. The body traps unless every job is a Quantize job of the
    // launch's width at 128 threads. All 32 lanes of every warp execute each shuffle (inactive half-warps use 0.0
    // and skip the stores). The caller owns the borrowed launch for this synchronous inline call. No reference escapes.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        let shape=launch.single.shape;
        if shape.op!=Op::Quantize || _block_dim_x() as usize!=THREADS || shape.cols==0 || shape.cols%16!=0 || launch.count==0 {asm!("trap;");}
        let units=shape.cols as usize/16;let per_job=units.div_ceil(THREADS/16);
        let (job,local)=match at {
            Some(local)=>{if launch.batch!=0 {asm!("trap;");}(0,local)},
            None=>if launch.batch==0 {(0,_block_idx_x() as usize)} else {(_block_idx_x() as usize/per_job,_block_idx_x() as usize%per_job)},
        };
        if job>=launch.count as usize {return;}
        let p:Arguments=if launch.batch==0 {launch.single} else {(launch.batch as *const Arguments).add(job).read()};
        let s=p.shape;
        if s.op!=Op::Quantize || s.cols!=shape.cols {asm!("trap;");}
        let tid=_thread_idx_x() as usize;
        let (block,element)=(local*(THREADS/16)+tid/16,tid%16);
        let live=block<units;
        let value=if live {*(p.input[0] as *const f32).add(block*16+element)} else {0.0};
        let magnitude=crate::expert_ops::round(value).abs();
        let mut maximum=magnitude;
        let mut distance=8u32;
        while distance!=0 {
            let other:u32;
            asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)other,v=in(reg32)maximum.to_bits(),d=in(reg32)distance);
            maximum=maximum.max(f32::from_bits(other));
            distance/=2;
        }
        let scale=crate::expert_ops::nearest_e4m3((maximum/s.activation)/6.0);let width=crate::expert_ops::e4m3(scale);
        let code=if width==0.0 {0} else {crate::expert_ops::nearest_e2m1((magnitude/s.activation)/width)};
        let code=code|if value<0.0 {8} else {0};
        if live {
            let out=(p.output as *mut f32).add(block*17);
            *out.add(element)=crate::expert_ops::e2m1(code);
            if element==0 {*out.add(16)=width;}
        }
    }
}
