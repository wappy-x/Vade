//! Split mHC boundaries with the fixed feature tree.
use crate::mhc_params::Boundary;
use core::slice;

macro_rules! boundary_rms_body {($arg:expr)=>{{let p=$arg;
    // SAFETY: The caller validates disjoint planes and retains their owners.
    // One block owns each token. Every guarded feature has one writer, and
    // all threads reach the barrier before warp zero reads the residual.
    // Previous planes are read only when present and within checked extents.
    // The full first warp participates in all shuffles. Lane zero alone writes
    // the inverse in the token's checked partial interval.
    // A block past the row grid is a CHAIN-SHADOW tail: it writes nothing,
    // reaches no barrier, and prefetches only inside the binder-checked live weight range.
    {
        let token=core::arch::nvptx::_block_idx_x() as usize;let shape=p.shape;
        // CHAIN-SHADOW tail: blocks past the row grid only prefetch, then return.
        crate::chain_shadow::shadow_tail!(p,shape.rows as usize);
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if threads<96 || threads>512 || threads%32!=0 {core::arch::asm!("trap;");}
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        let residual=(p.residual as *mut f32).add(token*wide);let mut base=0;
        // The token's BF16-rounded coefficients go to shared memory once (at most 32 of them); every feature then
        // reads them there. Same rounded values and the same combine() order, without per-feature slices.
        let table=crate::ptx::shared::<{32*4}>();let staged=p.previous_input!=0 && count<=32;
        let previous=(p.previous_input as *const f32).add(token*wide);let coefficients=(p.previous_coefficients as *const f32).add(token*count);
        if staged && thread<count {crate::ptx::store(table+(thread*4) as u32,crate::bf16::rounded(*coefficients.add(thread)));}
        core::arch::nvptx::_syncthreads();
        // Sixteen features a thread are made before any is stored, so their loads overlap. Same values as before.
        while base<wide {
            let mut values=[0.0f32;16];
            // 32-bit stream and feature indices (wide < 2^32 is checked by the host shape): a 64-bit division per feature cost more than the sums.
            crate::strip_warp::unroll!(8,a,{crate::strip_warp::unroll!(2,b,{let j=a*2+b;let feature=base+thread+j*threads;if feature<wide {
                values[j]=if p.previous_input==0 {crate::bf16::rounded(*(p.input as *const f32).add(token*wide+feature))}
                    else if staged {
                        // mhc_boundary::combine: residual = sum over src of rounded(comb) * previous, from +0.0 in src order.
                        let destination=(feature as u32/hidden as u32) as usize;let d=feature-destination*hidden;
                        let mut sum=0.0f32;let mut src=0;
                        while src<streams {sum+=crate::ptx::load(table+((2*streams+src*streams+destination)*4) as u32)* *previous.add(src*hidden+d);src+=1;}
                        let block=crate::bf16::rounded(crate::ptx::load(table+((streams+destination)*4) as u32)*crate::bf16::rounded(*(p.input as *const f32).add(token*hidden+d)));
                        crate::bf16::rounded(crate::bf16::rounded(sum)+block)
                    } else {
                        let destination=(feature as u32/hidden as u32) as usize;let d=feature-destination*hidden;
                        crate::mhc_boundary_math::combine(
                        slice::from_raw_parts(previous,wide),slice::from_raw_parts(coefficients,count),
                        crate::bf16::rounded(*(p.input as *const f32).add(token*hidden+d)),
                        streams,hidden,destination,d)};
            }});});
            crate::strip_warp::unroll!(8,a,{crate::strip_warp::unroll!(2,b,{let j=a*2+b;let feature=base+thread+j*threads;if feature<wide {*residual.add(feature)=values[j];}});});
            base+=16*threads;
        }
        core::arch::nvptx::_syncthreads();
        if thread<32 && wide%8==0 && (residual as u64)&15==0 {
            // Lane = strip: groups lane, lane + 32, ... ascending, eight at a time loaded first. Same terms and order as warp_fold.
            let mut sum=0.0f32;let mut first=thread*8;
            while first<wide {
                let mut v=[[0.0f32;8];8];
                crate::strip_warp::unroll!(8,g,{let at=first+g*256;if at<wide {
                    let lo=crate::strip_warp::ld4!(residual.add(at));let hi=crate::strip_warp::ld4!(residual.add(at+4));
                    crate::strip_warp::unroll!(4,e,{v[g][e]=lo[e];v[g][4+e]=hi[e];});
                }});
                crate::strip_warp::unroll!(8,g,{if first+g*256<wide {crate::strip_warp::unroll!(8,e,{crate::strip_warp::add!(sum,crate::strip_warp::mul!(v[g][e],v[g][e]));});}});
                first+=8*256;
            }
            let squares=crate::reduction::warp_root(sum);
            if thread==0 {*(p.norm_partials as *mut f32).add(token*hidden.div_ceil(128))=1.0/crate::qsa_common::root(squares/wide as f32+shape.rms_eps);}
        } else if thread<32 {
            let squares=crate::reduction::warp_fold!(wide,thread,|column|{let value=*residual.add(column);value*value});
            if thread==0 {*(p.norm_partials as *mut f32).add(token*hidden.div_ceil(128))=1.0/crate::qsa_common::root(squares/wide as f32+shape.rms_eps);}
        }
    }
}};}
#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_rms(p:Boundary) {
    // SAFETY: The contract is the comment at the head of boundary_rms_body. The launch binder checks the arguments.
    unsafe {boundary_rms_body!(p)}
}
/// Replay the unchanged rms body with the original grid and checked arguments.
#[inline(always)]
pub(crate) fn resident_mhc_boundary_rms(p:Boundary) {
    // SAFETY: The contract is the comment at the head of boundary_rms_body. The resident program checks the arguments before the replay.
    unsafe {boundary_rms_body!(p)}
}

macro_rules! boundary_dots_body {($arg:expr)=>{{let p=$arg;
    // SAFETY: The caller validates and retains disjoint planes. Each block owns
    // one token/coefficient pair and a disjoint normalized interval. The earlier
    // RMS launch completes residual and inverse writes. Warp zero alone folds
    // the checked row; every lane reaches every shuffle. Tail blocks prefetch
    // only inside the checked range through shadow_tail and write nothing.
    {
        let shape=p.shape;let count=shape.coefficients();let wide=shape.wide();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        let block=core::arch::nvptx::_block_idx_x() as usize;let token=block/count;let coefficient=block%count;
        let arithmetic=shape.rows as usize*count;
        crate::chain_shadow::shadow_tail!(p,arithmetic);
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if threads<96 || threads>512 || threads%32!=0 {core::arch::asm!("trap;");}
        let inverse=*(p.norm_partials as *const f32).add(token*hidden.div_ceil(128));
        let residual=(p.residual as *const f32).add(token*wide);
        let share=wide.div_ceil(count);let begin=coefficient*share;let end=(begin+share).min(wide);let mut feature=begin+thread;
        while feature<end {*(p.mixing_normalized as *mut f32).add(token*wide+feature)=*residual.add(feature)*inverse;feature+=threads;}
        if thread<32 {
            let row=(p.weights as *const f32).add(coefficient*wide);
            let total=crate::tree_reduction::warp_fold!(wide,thread,|column|{let normalized=*residual.add(column)*inverse;*row.add(column)*normalized});
            if thread==0 {
                let group=if coefficient<streams {0} else if coefficient<2*streams {1} else {2};
                *(p.coefficients as *mut f32).add(token*count+coefficient)=total * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(coefficient);
            }
        }
    }
}};}
#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_dots(p:Boundary) {
    // SAFETY: The contract is the comment at the head of boundary_dots_body. The launch binder checks the arguments.
    unsafe {boundary_dots_body!(p)}
}
/// Replay the unchanged dots body with the original grid and checked arguments.
#[inline(always)]
pub(crate) fn resident_mhc_boundary_dots(p:Boundary) {
    // SAFETY: The contract is the comment at the head of boundary_dots_body. The resident program checks the arguments before the replay.
    unsafe {boundary_dots_body!(p)}
}
