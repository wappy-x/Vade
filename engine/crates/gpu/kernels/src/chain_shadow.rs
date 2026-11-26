//! CHAIN-SHADOW. The mHC boundary chains are serial FP32 folds that leave DRAM
//! idle. Blocks placed past a launch's arithmetic grid issue L2 prefetches over
//! the next stage's static weights while the chain runs. A prefetch is a hint:
//! it writes no memory and returns no value, so no arithmetic block can observe
//! it except through timing. `chain_add_latency` measures the dependent FP32 add
//! that bounds those folds.
use crate::mhc_params::AddLatency;

/// The tail of a CHAIN-SHADOW launch. It expands only inside the existing,
/// checked unsafe blocks of `mhc_boundary_dots` and `mhc_boundary_finish`, as
/// their first statement after the block index, and returns from that kernel.
/// `$first` is the arithmetic grid size; a block below it continues unchanged.
/// A tail block writes no memory and reaches no barrier. The binder places
/// `[prefetch, prefetch+prefetch_bytes)` inside one live resident weight owner
/// of this context; every address issued is `prefetch + 128*line` with
/// `line < shadow_lines(prefetch_bytes)`, so it stays inside that span.
macro_rules! shadow_tail {($p:expr,$first:expr)=>{{
    let first:usize=$first;let block=core::arch::nvptx::_block_idx_x() as usize;
    if block>=first {
        let p=&$p;
        if p.prefetch!=0 && p.prefetch_bytes!=0 {
            if p.prefetch_policy>1 {core::arch::asm!("trap;");}
            let blocks=core::arch::nvptx::_grid_dim_x() as u64;let threads=core::arch::nvptx::_block_dim_x() as u64;
            let mut line=(block-first) as u64*threads+core::arch::nvptx::_thread_idx_x() as u64;
            let stride=(blocks-first as u64)*threads;let lines=crate::mhc_params::shadow_lines(p.prefetch_bytes);
            while line<lines {
                let address=p.prefetch+line*crate::mhc_params::SHADOW_LINE;
                if p.prefetch_policy==1 {core::arch::asm!("prefetch.global.L2::evict_last [{a}];",a=in(reg64)address,options(nostack,readonly));}
                else {core::arch::asm!("prefetch.global.L2 [{a}];",a=in(reg64)address,options(nostack,readonly));}
                line+=stride;
            }
        }
        return;
    }
}};}
pub(crate) use shadow_tail;

#[no_mangle]
pub extern "ptx-kernel" fn chain_shadow_prefetch(args:crate::mhc_params::ShadowPrefetch) {resident_chain_shadow_prefetch(args)}
/// The same body for the direct entry and for a recorded replay (resident_launch_shadow_prefetch).
#[inline(always)]
pub(crate) fn resident_chain_shadow_prefetch(args:crate::mhc_params::ShadowPrefetch) {
    // SAFETY: The checked binder keeps the nonempty prefetch span within one
    // live resident weight owner of the same context. The grid is nonempty,
    // has complete warps and uses policy 0 or 1. Each issued address is below
    // the checked span end. Every block takes the tail with no arithmetic
    // rows, writes no memory and reaches no barrier. Owner drop drains the
    // ordered stream before the allocation can be freed.
    unsafe {shadow_tail!(args,0);}
}

#[no_mangle]
pub extern "ptx-kernel" fn chain_add_latency(p:AddLatency) {
    // SAFETY: The host binder launches one block of one warp, checks that
    // `output` is a private 6-word FP32 plane of this context, and keeps its
    // owner live through the synchronize that follows the launch. Only thread 0
    // writes, and it writes the three u64 words inside that plane. The clock
    // reads take the chain value as an operand, so the dependent adds stay inside
    // the timed span; FP32 adds are not reassociated without fast math.
    unsafe {
        if core::arch::nvptx::_block_idx_x()!=0 || core::arch::nvptx::_thread_idx_x()!=0 {return;}
        let (c0,t0,start):(u64,u64,u32);
        core::arch::asm!("mov.u64 {c}, %clock64;","mov.u64 {t}, %globaltimer;","mov.b32 {o}, {i};",
            c=out(reg64)c0,t=out(reg64)t0,o=out(reg32)start,i=in(reg32)p.start.to_bits());
        // Group and tail counts come from probe_chunks: `group < groups` never wraps.
        let step=p.step;let mut value=f32::from_bits(start);let (groups,tail)=crate::mhc_params::probe_chunks(p.adds);
        let mut group=0u32;
        while group<groups {
            value+=step;value+=step;value+=step;value+=step;value+=step;value+=step;value+=step;value+=step;
            value+=step;value+=step;value+=step;value+=step;value+=step;value+=step;value+=step;value+=step;
            group+=1;
        }
        let mut left=tail;
        while left>0 {value+=step;left-=1;}
        let (c1,t1,result):(u64,u64,u32);
        core::arch::asm!("mov.b32 {o}, {i};","mov.u64 {c}, %clock64;","mov.u64 {t}, %globaltimer;",
            o=out(reg32)result,i=in(reg32)value.to_bits(),c=out(reg64)c1,t=out(reg64)t1);
        let out=p.output as *mut u64;
        *out=u64::from(result);*out.add(1)=c1.wrapping_sub(c0);*out.add(2)=t1.wrapping_sub(t0);
    }
}
