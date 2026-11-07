//! Two-launch form of `mhc_boundary_project`: the same arithmetic, the same
//! ascending FP32 chains, no fused multiply-add. The RMS sum and every
//! coefficient dot is ONE serial chain of adds that starts from 0.0 and takes
//! the rounded products in feature order. Producer warps compute the rounded
//! values in parallel into a shared ring; ONE lane folds them in order. The dots
//! run one per block, so a decode token uses one SM per coefficient.
//!
//! Launch: `mhc_boundary_rms` grid `rows`, any block of 96..512 threads (256
//! advised). `mhc_boundary_dots` grid `rows * coefficients` (block `b` is token
//! `b / coefficients`, coefficient `b % coefficients`), same thread rule (256
//! advised). The dots launch follows the rms launch on the same stream.
//! The inverse RMS travels in `norm_partials[token * ceil(hidden/128)]`; the
//! finish launch rewrites every partial before it reads one. Measured on one
//! GB10: faster than `mhc_boundary_project` up to 16 rows, slower from 32 rows.
use crate::mhc_params::Boundary;
use core::slice;

const SLOTS:usize=4;const CH:usize=1024;
const FULL:u32=1;const EMPTY:u32=1+SLOTS as u32;

// These macros expand only inside the two checked kernel boundary blocks.
// A slot is aligned shared memory, with at most CH initialized words. Each
// producer store precedes the full barrier; reuse follows the empty barrier.
macro_rules! fold_slot {($slot:expr,$span:expr,$acc:expr)=>{{
    let slot=$slot;let span=$span;let mut acc=$acc;
        let full=span/16*16;let mut j=0;
        if full>0 {
            // Groups of 16 words, two register sets used in turn: the next group is loaded before the current group's adds and
            // nothing is copied between groups. The adds are the same adds in the same ascending order.
            macro_rules! group {($g:expr)=>{{let at=slot+(($g)*64) as u32;[crate::ptx::load4(at),crate::ptx::load4(at+16),crate::ptx::load4(at+32),crate::ptx::load4(at+48)]}}}
            macro_rules! adds {($r:ident)=>{{for i in 0..16 {acc+=f32::from_bits($r[i/4][i%4]);}}}}
            let groups=full/16;
            let mut a=group!(0usize);let mut g=0usize;
            while g+1<groups {
                let b=group!(g+1);
                adds!(a);
                a=if g+2<groups {group!(g+2)} else {b};
                adds!(b);
                g+=2;
            }
            if g<groups {adds!(a);}
            j=full;
        }
        while j<span {acc+=crate::ptx::load(slot+(j*4) as u32);j+=1;}
    acc
}};}
macro_rules! barrier_wait {($id:expr,$threads:expr)=>{{
    core::arch::asm!("bar.sync {id}, {n};",id=in(reg32)$id,n=in(reg32)$threads);
}};}
macro_rules! barrier_arrive {($id:expr,$threads:expr)=>{{
    core::arch::asm!("bar.arrive {id}, {n};",id=in(reg32)$id,n=in(reg32)$threads);
}};}

// Straight-line copies so that array indices are constants and the arrays stay in registers.
macro_rules! rep4 {($v:ident,$body:block)=>{{{let $v:usize=0;$body}{let $v:usize=1;$body}{let $v:usize=2;$body}{let $v:usize=3;$body}}};}
macro_rules! rep16 {($v:ident,$body:block)=>{{rep4!(hi,{rep4!(lo,{let $v:usize=hi*4+lo;$body});});}};}

#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_rms(p:Boundary) {
    // SAFETY: The caller validates every disjoint plane and retains all owners
    // through the ordered consumer. One block owns one token and has 3..16 whole
    // warps (anything else traps). Warp zero folds; the other warps produce.
    // Producers write each residual word once and one shared word per feature,
    // inside the ring slot they own after the empty barrier. Previous planes are
    // read only when present and have the same token/stream widths.
    unsafe {
        let token=core::arch::nvptx::_block_idx_x() as usize;let shape=p.shape;
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if threads<96 || threads>512 || threads%32!=0 {core::arch::asm!("trap;");}
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        let smem=crate::ptx::shared::<{SLOTS*CH*4}>();let n=threads as u32;let chunks=wide.div_ceil(CH);
        if thread<32 {
            let mut squares=0.0f32;
            for c in 0..chunks {
                let s=c%SLOTS;barrier_wait!(FULL+s as u32,n);
                if thread==0 {squares=fold_slot!(smem+(s*CH*4) as u32,(wide-c*CH).min(CH),squares);}
                if c+SLOTS<chunks {barrier_arrive!(EMPTY+s as u32,n);}
            }
            if thread==0 {
                *(p.norm_partials as *mut f32).add(token*hidden.div_ceil(128))=1.0/crate::qsa_common::root(squares/wide as f32+shape.rms_eps);
            }
        } else {
            let producers=threads-32;let lane=thread-32;
            let fast=p.previous_input!=0 && streams==4 && hidden%CH==0 && producers*16>=CH;
            for c in 0..chunks {
                let s=c%SLOTS;if c>=SLOTS {barrier_wait!(EMPTY+s as u32,n);}
                let first=c*CH;let span=(wide-first).min(CH);let slot=smem+(s*CH*4) as u32;
                if fast {
                    // After-site fast path: with four streams and a hidden width that is a multiple of the chunk, every
                    // feature of this chunk has the same stream index, so the five BF16-rounded coefficients are read
                    // and rounded once, and no feature needs a division. Same values, same order as combine().
                    let destination=first/hidden;let base=first-destination*hidden;
                    let coefficients=(p.previous_coefficients as *const f32).add(token*count);
                    let comb=[crate::bf16::rounded(*coefficients.add(8+destination)),crate::bf16::rounded(*coefficients.add(8+4+destination)),
                        crate::bf16::rounded(*coefficients.add(8+8+destination)),crate::bf16::rounded(*coefficients.add(8+12+destination))];
                    let post=crate::bf16::rounded(*coefficients.add(4+destination));
                    let previous=(p.previous_input as *const f32).add(token*wide);let block=(p.input as *const f32).add(token*hidden);
                    let mut values=[0.0f32;16];
                    rep16!(j,{
                        let at=lane+j*producers;
                        if at<span {
                            let d=base+at;
                            let x0=*previous.add(d);let x1=*previous.add(hidden+d);let x2=*previous.add(2*hidden+d);let x3=*previous.add(3*hidden+d);
                            let b=crate::bf16::rounded(*block.add(d));
                            let mut residual=0.0f32;
                            residual+=comb[0]*x0;residual+=comb[1]*x1;residual+=comb[2]*x2;residual+=comb[3]*x3;
                            let product=crate::bf16::rounded(post*b);
                            values[j]=crate::bf16::rounded(crate::bf16::rounded(residual)+product);
                        }
                    });
                    rep16!(j,{
                        let at=lane+j*producers;
                        if at<span {
                            let value=values[j];
                            *(p.residual as *mut f32).add(token*wide+first+at)=value;
                            crate::ptx::store(slot+(at*4) as u32,value*value);
                        }
                    });
                } else {
                // Compute every value this thread owns first, so its loads overlap; then store.
                let mut values=[0.0f32;16];let mut j=0;
                while j<16 {
                    let at=lane+j*producers;
                    if at<span {
                        let feature=first+at;
                        values[j]=if p.previous_input==0 {crate::bf16::rounded(*(p.input as *const f32).add(token*wide+feature))}
                        else {crate::mhc_boundary_math::combine(
                            slice::from_raw_parts((p.previous_input as *const f32).add(token*wide),wide),
                            slice::from_raw_parts((p.previous_coefficients as *const f32).add(token*count),count),
                            crate::bf16::rounded(*(p.input as *const f32).add(token*hidden+feature%hidden)),
                            streams,hidden,feature/hidden,feature%hidden)};
                    }
                    j+=1;
                }
                j=0;
                while j<16 {
                    let at=lane+j*producers;
                    if at<span {
                        let value=values[j];
                        *(p.residual as *mut f32).add(token*wide+first+at)=value;
                        crate::ptx::store(slot+(at*4) as u32,value*value);
                    }
                    j+=1;
                }
                }
                barrier_arrive!(FULL+s as u32,n);
            }
        }
    }
}

#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_dots(p:Boundary) {
    // SAFETY: As mhc_boundary_rms. One block owns one (token, coefficient) pair:
    // it writes one coefficient word and one slice of the normalized plane; no
    // other block writes either. A block past rows*coefficients is a CHAIN-SHADOW
    // tail: it writes nothing, reaches no barrier, and prefetches only inside the
    // binder-checked live weight range (chain_shadow::shadow_tail). The residual and the inverse were completed by
    // the preceding launch on the same stream.
    unsafe {
        let shape=p.shape;let count=shape.coefficients();let wide=shape.wide();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        let block=core::arch::nvptx::_block_idx_x() as usize;let token=block/count;let coefficient=block%count;
        // CHAIN-SHADOW tail: blocks past the arithmetic grid only prefetch, then return.
        let arithmetic=shape.rows as usize*count;
        crate::chain_shadow::shadow_tail!(p,arithmetic);
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if threads<96 || threads>512 || threads%32!=0 {core::arch::asm!("trap;");}
        let smem=crate::ptx::shared::<{SLOTS*CH*4}>();let n=threads as u32;let chunks=wide.div_ceil(CH);
        let inverse=*(p.norm_partials as *const f32).add(token*hidden.div_ceil(128));
        let residual=(p.residual as *const f32).add(token*wide);
        if thread<32 {
            let mut total=0.0f32;
            for c in 0..chunks {
                let s=c%SLOTS;barrier_wait!(FULL+s as u32,n);
                if thread==0 {total=fold_slot!(smem+(s*CH*4) as u32,(wide-c*CH).min(CH),total);}
                if c+SLOTS<chunks {barrier_arrive!(EMPTY+s as u32,n);}
            }
            if thread==0 {
                let group=if coefficient<streams {0} else if coefficient<2*streams {1} else {2};
                *(p.coefficients as *mut f32).add(token*count+coefficient)=total * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(coefficient);
            }
        } else {
            let producers=threads-32;let lane=thread-32;let row=(p.weights as *const f32).add(coefficient*wide);
            for c in 0..chunks {
                let s=c%SLOTS;if c>=SLOTS {barrier_wait!(EMPTY+s as u32,n);}
                let first=c*CH;let span=(wide-first).min(CH);let slot=smem+(s*CH*4) as u32;
                let mut x=[0.0f32;16];let mut w=[0.0f32;16];let mut j=0;
                while j<16 {
                    let at=lane+j*producers;
                    if at<span {x[j]=*residual.add(first+at);w[j]=*row.add(first+at);}
                    j+=1;
                }
                j=0;
                while j<16 {
                    let at=lane+j*producers;
                    if at<span {
                        let normalized=x[j]*inverse;
                        crate::ptx::store(slot+(at*4) as u32,w[j]*normalized);
                    }
                    j+=1;
                }
                barrier_arrive!(FULL+s as u32,n);
            }
            let share=wide.div_ceil(count);let begin=coefficient*share;let end=(begin+share).min(wide);let mut feature=begin+lane;
            while feature<end {*(p.mixing_normalized as *mut f32).add(token*wide+feature)=*residual.add(feature)*inverse;feature+=producers;}
        }
    }
}
