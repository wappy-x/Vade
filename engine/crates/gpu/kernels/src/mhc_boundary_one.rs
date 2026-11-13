//! D1: the whole one-row mHC boundary in ONE launch. Bit-equal to the split chain (mhc_boundary_rms,
//! mhc_boundary_dots, mhc_boundary_finish) and to the CPU lane (gpu/src/mhc_host.rs), SUM-ORDER v1.
//! Grid: FUSED_BLOCKS blocks of FUSED_THREADS threads. Block k owns strip k: the features 256 p + 8 k + e.
//! 1. The block starts the copy of its strip's BF16 weights (cp.async), makes the residual of its features
//!    and folds its RMS strip (one chain of 8 x periods squares, ascending).
//! 2. Meeting 1: every block publishes its strip sum and waits for all 32; each block joins them itself
//!    (16, 8, 4, 2, 1), so every block holds the same inverse.
//! 3. The block makes y = x * inverse for its features and folds one strip chain for each coefficient row.
//! 4. Meeting 2: the last block to arrive joins the row strips, scales and adds the base, then warp 0 runs
//!    the gates and Sinkhorn while the other seven warps collapse and norm (collapse needs only `pre`).
//! Every product is `mul.rn.f32` and every sum `add.rn.f32` (no fused multiply-add); padded features add
//! +0.0, which leaves a strip that starts at +0.0 unchanged.
use crate::mhc_params::{FusedBoundary,FUSED_BLOCKS,FUSED_THREADS,FUSED_STAGE_BYTES,FUSED_STRIP_WORDS,FUSED_SUMS,FUSED_STAMPS_AT,FUSED_STAMP_SLOTS};
use crate::{mhc_math as math,strip_warp::{add,mul}};
use core::arch::asm;

const X:u32=FUSED_STAGE_BYTES as u32;
const Y:u32=X+FUSED_STRIP_WORDS as u32*4;
const TABLE:u32=Y+FUSED_STRIP_WORDS as u32*4;
const MISC:u32=TABLE+32*4;
const PARTS:u32=MISC+32*4;
const SHARED:usize=PARTS as usize+64*4;

macro_rules! sum {($a:expr,$b:expr)=>{{let mut value:f32=$a;add!(value,$b);value}};}
/// Phase stamp: %globaltimer into this block's slot (diagnostic; it changes no value).
macro_rules! stamp {($stamps:expr,$slot:expr)=>{{let t:u64;asm!("mov.u64 {t}, %globaltimer;",t=out(reg64)t);*$stamps.add($slot)=t;}};}
// The three helpers below are macros so that they expand inside the kernel's one checked unsafe block.
/// A word another block wrote in this launch: read at L2, never from this SM's L1.
macro_rules! load_cg {($p:expr)=>{{let v:f32;asm!("ld.global.cg.f32 {v}, [{p}];",v=out(reg32)v,p=in(reg64)$p,options(readonly,nostack));v}};}
/// Four such words at a 16-byte aligned address.
macro_rules! load_cg4 {($p:expr)=>{{let (a,b,c,d):(f32,f32,f32,f32);
    asm!("ld.global.cg.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg64)$p,options(readonly,nostack));[a,b,c,d]}};}
/// Publish this block's earlier writes and count it in. Gives the count before this block. The caller has passed a
/// block barrier after every write it publishes (one gpu fence then covers the block, as a cooperative grid sync does).
macro_rules! arrive {($counter:expr)=>{{
    let old:u32;
    asm!("fence.acq_rel.gpu;");
    asm!("atom.acq_rel.gpu.global.add.u32 {o}, [{c}], 1;",o=out(reg32)old,c=in(reg64)$counter);
    old
}};}
/// Spin until the counter holds `target` (every block of the grid is resident, see the contract).
macro_rules! wait_count {($counter:expr,$target:expr)=>{{
    loop {
        let v:u32;asm!("ld.acquire.gpu.global.u32 {v}, [{c}];",v=out(reg32)v,c=in(reg64)$counter);
        if v>=$target {break;}
        asm!("nanosleep.u32 32;");
    }
}};}
#[inline(always)]
fn word(bits:u32)->f32 {f32::from_bits(bits)}

macro_rules! fused_body {($arg:expr)=>{{let f:FusedBoundary=$arg;let p=f.boundary;
    // SAFETY contract (the callers wrap this body in one unsafe block):
    // The host binder (ResidentMhcBoundary::submit_fused) admits only fused_admitted shapes (one row,
    // streams <= 4, a strip's planes fit the shared areas), checks every plane extent, alias and
    // context, keeps every owner live through the ordered consumer, and launches exactly
    // FUSED_BLOCKS x FUSED_THREADS (anything else traps). All FUSED_BLOCKS blocks are resident together
    // (32 blocks of 31 KB shared on a 48-SM device), so the spin of meeting 1 ends. Block k writes only the
    // residual and mixing words of strip k, scratch word FUSED_SUMS + k and the words FUSED_SUMS + 32 + 32 r + k;
    // the last block alone writes coefficients, collapsed, partials, output and resets both counters.
    // The previous residual and coefficients are read only when present, inside their checked extents (wide, count).
    // Shared memory: the stage holds at most fused_stage_fits bytes of weights (checked by the binder),
    // later the collapsed plane (hidden <= FUSED_STAGE_BYTES / 4). Every barrier is reached by every thread
    // of the block (bar 1 by exactly warps 1..7); full-mask shuffles run in complete warps.
    {
        let k=core::arch::nvptx::_block_idx_x() as usize;let thread=core::arch::nvptx::_thread_idx_x() as usize;
        if k>=FUSED_BLOCKS || core::arch::nvptx::_block_dim_x() as usize!=FUSED_THREADS {asm!("trap;");}
        let shape=p.shape;let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        let periods=wide.div_ceil(256);let terms=periods*8;
        let smem=crate::ptx::shared::<SHARED>();
        let counters=f.scratch as *mut u32;let sums=(f.scratch as *mut f32).add(FUSED_SUMS);
        let stamps=(f.scratch as *mut u64).add(FUSED_STAMPS_AT/2);let mine=stamps.add(k*FUSED_STAMP_SLOTS);
        if thread==0 {stamp!(mine,0);}
        // 1. This strip's weights toward shared memory, in flight during the residual and meeting 1.
        let previous=p.previous_input!=0;
        // This thread's coefficient row scale and base (threads < count fold rows in the tail), loaded with the first batch.
        let (row_scale,row_base)=if thread<count {let group=if thread<streams {0} else if thread<2*streams {1} else {2};
            (*(p.scale as *const f32).add(group),*(p.base as *const f32).add(thread))} else {(0.0f32,0.0f32)};
        // The residual of this strip's features: the rounded input, or mhc_boundary::combine in its order.
        // A thread owns at most 3 terms (terms <= FUSED_STRIP_WORDS <= 3 x FUSED_THREADS); every load of its terms is
        // issued before any arithmetic or store, so they share one memory round trip.
        let residual=p.residual as *mut f32;let input=p.input as *const f32;
        let (last,coefficients)=(p.previous_input as *const f32,p.previous_coefficients as *const f32);
        let mut held=[0.0f32;3];let mut last_words=[[0.0f32;4];3];let mut mixes=[[0.0f32;5];3];
        crate::strip_warp::unroll!(3,m,{let i=thread+m*FUSED_THREADS;if i<terms {let j=(i/8)*256+k*8+i%8;if j<wide {
            if !previous {held[m]=*input.add(j);} else {
                let destination=(j as u32/hidden as u32) as usize;let d=j-destination*hidden;held[m]=*input.add(d);
                crate::strip_warp::unroll!(4,src,{if src<streams {last_words[m][src]=*last.add(src*hidden+d);mixes[m][src]=*coefficients.add(2*streams+src*streams+destination);}});
                mixes[m][4]=*coefficients.add(streams+destination);
            }
        }}});
        if f.bf16!=0 {
            let chunks=periods*count;let source=(f.strips as *const u8).add(k*chunks*16);
            let mut c=thread;while c<chunks {crate::ptx::copy16(smem+(c*16) as u32,source.add(c*16).cast());c+=FUSED_THREADS;}
            crate::ptx::commit();
        }
        crate::strip_warp::unroll!(3,m,{let i=thread+m*FUSED_THREADS;if i<terms {let j=(i/8)*256+k*8+i%8;
            let x=if j>=wide {0.0f32} else if !previous {crate::bf16::rounded(held[m])} else {
                let mut total=0.0f32;
                crate::strip_warp::unroll!(4,src,{if src<streams {add!(total,mul!(crate::bf16::rounded(mixes[m][src]),last_words[m][src]));}});
                let block=crate::bf16::rounded(mul!(crate::bf16::rounded(mixes[m][4]),crate::bf16::rounded(held[m])));
                crate::bf16::rounded(sum!(crate::bf16::rounded(total),block))
            };
            if j<wide {*residual.add(j)=x;}
            crate::ptx::store(smem+X+(i*4) as u32,x);
        }});
        core::arch::nvptx::_syncthreads();
        if thread==0 {stamp!(mine,1);}
        // 2. The RMS strip: one ascending chain. Meeting 1.
        if thread==0 {
            let mut s=0.0f32;let mut i=0;
            while i<terms {
                let a=crate::ptx::load4(smem+X+(i*4) as u32);let b=crate::ptx::load4(smem+X+(i*4+16) as u32);
                crate::strip_warp::unroll!(4,e,{add!(s,mul!(word(a[e]),word(a[e])));});
                crate::strip_warp::unroll!(4,e,{add!(s,mul!(word(b[e]),word(b[e])));});
                i+=8;
            }
            *sums.add(k)=s;stamp!(mine,2);arrive!(counters);wait_count!(counters,FUSED_BLOCKS as u32);asm!("fence.acq_rel.gpu;");stamp!(mine,3);
        }
        core::arch::nvptx::_syncthreads();
        if thread<32 {
            let squares=crate::reduction::warp_root(load_cg!(sums.add(thread)));
            if thread==0 {crate::ptx::store(smem+MISC,1.0/crate::qsa_common::root(squares/wide as f32+shape.rms_eps));}
        }
        core::arch::nvptx::_syncthreads();
        // 3. y = x * inverse for this strip (the mixing plane), then one strip chain a coefficient row.
        let inverse=crate::ptx::load(smem+MISC);let mut i=thread;
        while i<terms {
            let j=(i/8)*256+k*8+i%8;
            let y=if j<wide {let y=mul!(crate::ptx::load(smem+X+(i*4) as u32),inverse);*(p.mixing_normalized as *mut f32).add(j)=y;y} else {0.0f32};
            crate::ptx::store(smem+Y+(i*4) as u32,y);i+=FUSED_THREADS;
        }
        if f.bf16!=0 {crate::ptx::wait();}
        core::arch::nvptx::_syncthreads();
        if thread==0 {stamp!(mine,4);}
        let (warp,lane)=(thread/32,thread%32);let row=warp*8+lane;
        if lane<8 && row<count {
            let mut s=0.0f32;
            if f.bf16!=0 {
                // Software pipeline: the next period's y and weights are loaded before this period's chain step.
                let load=|period:usize|(crate::ptx::load4(smem+Y+(period*32) as u32),crate::ptx::load4(smem+Y+(period*32+16) as u32),crate::ptx::load4(smem+((period*count+row)*16) as u32));
                let mut now=load(0);let mut period=0;
                while period<periods {
                    let next=if period+1<periods {load(period+1)} else {now};
                    let (a,b,q)=now;
                    let y=[word(a[0]),word(a[1]),word(a[2]),word(a[3]),word(b[0]),word(b[1]),word(b[2]),word(b[3])];
                    let w=[word(q[0]<<16),word(q[0]&0xffff_0000),word(q[1]<<16),word(q[1]&0xffff_0000),word(q[2]<<16),word(q[2]&0xffff_0000),word(q[3]<<16),word(q[3]&0xffff_0000)];
                    let t=[mul!(w[0],y[0]),mul!(w[1],y[1]),mul!(w[2],y[2]),mul!(w[3],y[3]),mul!(w[4],y[4]),mul!(w[5],y[5]),mul!(w[6],y[6]),mul!(w[7],y[7])];
                    crate::strip_warp::unroll!(8,e,{add!(s,t[e]);});
                    now=next;period+=1;
                }
            } else {
                let global=(f.strips as *const f32).add(k*periods*count*8+row*8);let mut period=0;
                while period<periods {
                    let a=crate::ptx::load4(smem+Y+(period*32) as u32);let b=crate::ptx::load4(smem+Y+(period*32+16) as u32);
                    let y=[word(a[0]),word(a[1]),word(a[2]),word(a[3]),word(b[0]),word(b[1]),word(b[2]),word(b[3])];
                    let at=global.add(period*count*8);
                    let w=[*at,*at.add(1),*at.add(2),*at.add(3),*at.add(4),*at.add(5),*at.add(6),*at.add(7)];
                    crate::strip_warp::unroll!(8,e,{add!(s,mul!(w[e],y[e]));});
                    period+=1;
                }
            }
            *sums.add(32+row*32+k)=s;
        }
        // 4. Meeting 2: the last block finishes the boundary.
        core::arch::nvptx::_syncthreads();
        if thread==0 {stamp!(mine,5);for slot in 6..FUSED_STAMP_SLOTS {*mine.add(slot)=0;}}
        if thread==0 {let old=arrive!(counters.add(1));crate::ptx::store(smem+MISC+4,if old as usize+1==FUSED_BLOCKS {1.0} else {0.0});}
        core::arch::nvptx::_syncthreads();
        if crate::ptx::load(smem+MISC+4)==0.0 {return;}
        if thread==0 {asm!("fence.acq_rel.gpu;");}
        core::arch::nvptx::_syncthreads();
        // Every block has passed meeting 1 (it arrived at meeting 2 after it), so both counters may rest at 0.
        if thread==0 {*counters=0;*counters.add(1)=0;}
        // Fast tail (hidden a multiple of 4, at most 5 quads a worker): every residual quad and norm-weight quad a
        // worker needs is loaded now, in one batch, while warp 0 joins the rows.
        let workers=FUSED_THREADS-32;let quads=hidden/4;let fast=hidden%4==0 && quads<=5*workers;
        let mut x=[[[0.0f32;4];4];5];let mut weight=[[0.0f32;4];5];
        if fast && thread>=32 {let worker=thread-32;crate::strip_warp::unroll!(5,m,{let q=worker+m*workers;if q<quads {
            crate::strip_warp::unroll!(4,src,{if src<streams {x[m][src]=load_cg4!(residual.add(src*hidden+q*4));}});
            weight[m]=load_cg4!((p.norm_weight as *const f32).add(q*4));
        }});}
        if thread<count {
            // The row's 32 strips joined at 16, 8, 4, 2, 1 (reduction::StripFold::finish), then scale and base.
            let mut s=[0.0f32;32];crate::strip_warp::unroll!(8,a,{crate::strip_warp::unroll!(4,b,{s[a*4+b]=load_cg!(sums.add(32+thread*32+a*4+b));});});
            crate::strip_warp::unroll!(4,a,{crate::strip_warp::unroll!(4,b,{let i=a*4+b;s[i]=sum!(s[i],s[i+16]);});});
            crate::strip_warp::unroll!(8,i,{s[i]=sum!(s[i],s[i+8]);});
            crate::strip_warp::unroll!(4,i,{s[i]=sum!(s[i],s[i+4]);});
            crate::strip_warp::unroll!(2,i,{s[i]=sum!(s[i],s[i+2]);});
            s[0]=sum!(s[0],s[1]);
            crate::ptx::store(smem+TABLE+(thread*4) as u32,sum!(mul!(s[0],row_scale),row_base));
        }
        core::arch::nvptx::_syncthreads();
        if thread==0 {stamp!(mine,6);}
        if thread<32 {
            // Gates and Sinkhorn: the streams * streams <= 32 branch of mhc_boundary_finish, raw values from shared memory.
            let coefficients=p.coefficients as *mut f32;
            let raw=|index:usize|crate::ptx::load(smem+TABLE+(index*4) as u32);
            let pre=if lane<streams {math::sigmoid(raw(lane))+shape.eps} else {0.0};
            let post=if lane<streams {2.0*math::sigmoid(raw(streams+lane))} else {0.0};
            let mut value=if lane<streams*streams {raw(2*streams+lane)} else {0.0};
            let row=(lane/streams).min(streams-1);let column=lane%streams;
            macro_rules! read_lane {($source:expr)=>{f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),(lane^($source)) as u32,31))}}
            if streams==4 {
                // The same reads, sums and divisions in the same order, with the shuffle masks fixed once: for a cell lane
                // (row 4 r + column c) the row source 4 r + s is lane ^ (c ^ s) and the column source 4 s + c is lane ^ 4 (r ^ s).
                // Lanes 16..31 read other lanes and store nothing, as before.
                let along=[(column^0) as u32,(column^1) as u32,(column^2) as u32,(column^3) as u32];
                let down=[((row^0)*4) as u32,((row^1)*4) as u32,((row^2)*4) as u32,((row^3)*4) as u32];
                macro_rules! read_mask {($mask:expr)=>{f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),$mask,31))}}
                let mut maximum=f32::NEG_INFINITY;
                crate::strip_warp::unroll!(4,source,{maximum=maximum.max(read_mask!(along[source]));});
                value=crate::qsa_common::exponential(value-maximum);
                let mut total=0.0f32;crate::strip_warp::unroll!(4,source,{total+=read_mask!(along[source]);});
                value=value/total+shape.eps;
                let mut total=0.0f32;crate::strip_warp::unroll!(4,source,{total+=read_mask!(down[source]);});
                value/=total+shape.eps;
                for _iteration in 1..shape.iterations {
                    let mut total=0.0f32;crate::strip_warp::unroll!(4,source,{total+=read_mask!(along[source]);});
                    value/=total+shape.eps;
                    let mut total=0.0f32;crate::strip_warp::unroll!(4,source,{total+=read_mask!(down[source]);});
                    value/=total+shape.eps;
                }
            } else {
                let mut maximum=f32::NEG_INFINITY;
                for source in 0..streams {maximum=maximum.max(read_lane!(row*streams+source));}
                value=crate::qsa_common::exponential(value-maximum);
                let mut total=0.0f32;
                for source in 0..streams {total+=read_lane!(row*streams+source);}
                value=value/total+shape.eps;
                let mut total=0.0f32;
                for source in 0..streams {total+=read_lane!(source*streams+column);}
                value/=total+shape.eps;
                for _iteration in 1..shape.iterations {
                    let mut total=0.0f32;
                    for source in 0..streams {total+=read_lane!(row*streams+source);}
                    value/=total+shape.eps;
                    let mut total=0.0f32;
                    for source in 0..streams {total+=read_lane!(source*streams+column);}
                    value/=total+shape.eps;
                }
            }
            if lane<streams {*coefficients.add(lane)=pre;*coefficients.add(streams+lane)=post;}
            if lane<streams*streams {*coefficients.add(2*streams+lane)=value;}
            if lane==0 {stamp!(mine,7);}
        } else {
            // Warps 1..7: collapse (it needs only pre), then the ordinary norm of mhc_boundary_finish.
            let worker=thread-32;
            let mut pre=[0.0f32;4];crate::strip_warp::unroll!(4,src,{if src<streams {pre[src]=math::sigmoid(crate::ptx::load(smem+TABLE+(src*4) as u32))+shape.eps;}});
            // mhc_boundary::collapse: from +0.0, ascending source, one round. The residual words come from every block:
            // L2 reads, 16 of them in flight a thread (4 quads x 4 streams) before any arithmetic.
            let mut collapsed=[[0.0f32;4];5];
            if fast {
                crate::strip_warp::unroll!(5,m,{let q=worker+m*workers;if q<quads {crate::strip_warp::unroll!(4,e,{
                    let mut value=0.0f32;crate::strip_warp::unroll!(4,src,{if src<streams {add!(value,mul!(pre[src],x[m][src][e]));}});
                    let value=crate::bf16::rounded(value);let feature=q*4+e;collapsed[m][e]=value;
                    *(p.collapsed as *mut f32).add(feature)=value;crate::ptx::store(smem+(feature*4) as u32,value);
                });}});
            } else {
                let mut feature=worker;
                while feature<hidden {
                    let mut value=0.0f32;let mut src=0;
                    while src<streams {add!(value,mul!(pre[src],load_cg!(residual.add(src*hidden+feature))));src+=1;}
                    let value=crate::bf16::rounded(value);
                    *(p.collapsed as *mut f32).add(feature)=value;crate::ptx::store(smem+(feature*4) as u32,value);feature+=workers;
                }
            }
            asm!("bar.sync 1, {n};",n=const FUSED_THREADS-32);
            let chunks=hidden.div_ceil(128);let mut chunk=worker/32;
            while chunk<chunks {
                let mut squares=0.0f32;let mut feature=chunk*128+lane;
                while feature<(chunk*128+128).min(hidden) {let value=crate::ptx::load(smem+(feature*4) as u32);squares+=value*value;feature+=32;}
                let mut distance=16;
                while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
                if lane==0 {*(p.norm_partials as *mut f32).add(chunk)=squares;crate::ptx::store(smem+PARTS+(chunk*4) as u32,squares);}
                chunk+=workers/32;
            }
            asm!("bar.sync 1, {n};",n=const FUSED_THREADS-32);
            let mut squares=0.0f32;let mut chunk=lane;
            while chunk<chunks {squares+=crate::ptx::load(smem+PARTS+(chunk*4) as u32);chunk+=32;}
            let mut distance=16;
            while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
            let inverse=1.0/crate::layer::sqrt(squares/hidden as f32+p.norm_epsilon.abs());
            if fast {crate::strip_warp::unroll!(5,m,{let q=worker+m*workers;if q<quads {crate::strip_warp::unroll!(4,e,{
                let value=crate::layer::round_bf16(collapsed[m][e]*inverse*1.0) * weight[m][e];
                *(p.output as *mut f32).add(q*4+e)=if p.norm_bf16==0 {value} else {crate::bf16::rounded(value)};
            });}});}
            let mut feature=if fast {hidden} else {worker};
            while feature<hidden {
                let normalized=crate::layer::round_bf16(crate::ptx::load(smem+(feature*4) as u32)*inverse*1.0);
                let value=normalized * *(p.norm_weight as *const f32).add(feature);
                *(p.output as *mut f32).add(feature)=if p.norm_bf16==0 {value} else {crate::bf16::rounded(value)};
                feature+=workers;
            }
            asm!("bar.sync 1, {n};",n=const FUSED_THREADS-32);
            if worker==0 {stamp!(stamps,FUSED_BLOCKS*FUSED_STAMP_SLOTS);}
        }
    }
}};}
#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_fused(f:FusedBoundary) {
    // SAFETY: The contract is the comment at the head of fused_body. The launch binder checks the arguments.
    unsafe {fused_body!(f)}
}
/// Replay the unchanged fused body with the original grid and checked arguments.
#[inline(always)]
pub(crate) fn resident_mhc_boundary_fused(f:FusedBoundary) {
    // SAFETY: The contract is the comment at the head of fused_body. The resident program checks the arguments before the replay.
    unsafe {fused_body!(f)}
}
