//! Fixed-order latent absorption on the tensor cores, the step form (one token): query, scores, values and project.
//! The prompt form (a group of tokens) is `latent_absorption_tile.rs`. The word function and every index rule live in
//! `gpu/src/absorb_tc.rs`; the CPU twin is `gpu/src/tc_twin_bf16.rs` (`fold`). Both forms give the same word: they
//! differ only in WHERE a word is computed (which block, warp and step order of loads).
use crate::{absorb_tc::{self as tc,Form,Kind},canonical::{ZERO,add,madd}};
use core::arch::{asm,nvptx::{_thread_idx_x,_block_idx_x,_block_dim_x}};
macro_rules! tiles {($t:ident=>$body:block)=>{{const $t:usize=0;$body}{const $t:usize=1;$body}{const $t:usize=2;$body}{const $t:usize=3;$body}};}
pub(crate) use tiles;
/// Close range `part` of a warp that runs the four ranges one after another: ranges 0 and 1 join into `p0`, range 2
/// waits in `p2`, range 3 joins the word `fl(fl(p0 + p1) + fl(p2 + p3))`.
macro_rules! close {($part:ident,$acc:ident,$p0:ident,$p2:ident)=>{
    if $part==0 {$p0=$acc;$acc=ZERO;} else if $part==1 {$p0=add($p0,$acc);$acc=ZERO;} else if $part==2 {$p2=$acc;$acc=ZERO;} else {$acc=add($p0,add($p2,$acc));}
    $part+=1;
};}
/// Pull the 64-byte line at `$p` toward the L2 cache: no value, no order, no fault. A warp runs in order and waits
/// for each load it uses; the hint puts the bytes of its whole tile in flight before the first load asks for them.
macro_rules! fetch {($p:expr)=>{asm!("prefetch.global.L2 [{p}];",p=in(reg64)$p,options(nostack,readonly))};}
/// The join of the range-parallel step form: a warp hands its four accumulators of each live N tile to shared memory,
/// or adds the handed ones to its own (one FP32 add a word).
macro_rules! hand {($stage:ident,$acc:ident,$n0:ident,$n_all:ident,$lane:ident,$index:expr)=>{tiles!(T=>{if $n0+T*8<$n_all {
    let at=$stage+(((($index)*tc::TILES+T)*32+$lane)*16) as u32;
    crate::ptx::store(at,$acc[T].a);crate::ptx::store(at+4,$acc[T].b);crate::ptx::store(at+8,$acc[T].c);crate::ptx::store(at+12,$acc[T].d);
}})};}
macro_rules! take {($stage:ident,$acc:ident,$n0:ident,$n_all:ident,$lane:ident,$index:expr)=>{tiles!(T=>{if $n0+T*8<$n_all {
    let o=crate::ptx::load4($stage+(((($index)*tc::TILES+T)*32+$lane)*16) as u32);
    $acc[T]=add($acc[T],crate::canonical::Acc {a:f32::from_bits(o[0]),b:f32::from_bits(o[1]),c:f32::from_bits(o[2]),d:f32::from_bits(o[3])});
}})};}
/// The step form of query and project (one token): the word of `run`, computed where one launch costs least.
/// A block owns one head and 128 rows of M; a warp owns one M tile and runs the four ranges in ascending order. The one
/// column of B (the token's row) is staged once, by the first threads of the block. A warp issues the weight loads of
/// four steps before it uses any of them, and the loads of the next four before the products of the current four, so a
/// sum of 16 or 32 steps waits for memory four or eight times, not at every step.
macro_rules! dense_step {($p:expr)=>{{
    const DENSE_STAGE:usize=2048;
    let p:crate::causal_ops::Arguments=$p;
    let Some(f)=Form::step(p.shape) else {asm!("trap;");return;};
    if _block_dim_x() as u32!=tc::THREADS || f.attention() || f.n()!=1 {asm!("trap;");}
    let thread=_thread_idx_x() as usize;let (warp,lane)=(thread/32,thread%32);
    let (outer,m0,_)=f.step_block(_block_idx_x() as usize);
    if outer>=f.outer() {return;}
    let m_all=f.m();let terms=f.terms(0);let steps=tc::steps(terms);
    if steps*32>DENSE_STAGE {asm!("trap;");}
    let (x,w,out)=(p.input[0] as *const u32,p.weight as *const u16,p.output as *mut f32);
    let stage=crate::ptx::shared::<DENSE_STAGE>();
    // The weight lines of the four steps from `s` in this warp's 16 rows (2 lines of 32 weights a row): one hint a lane.
    // The hints run two batches ahead of the products, so a batch of loads finds its lines in flight or there.
    let mw=m0+warp*16;
    let hint=#[inline(always)] |s:usize| {
        let (m,k)=(mw+lane/2,s*tc::STEP+(lane%2)*32);
        if m<m_all && k<terms {fetch!(w.cast::<u8>().add(f.weight(outer,m,k)*2));}
    };
    hint(0);hint(4);
    // B: column 0 lives in lanes 0..4 of a warp. Entry `e` is (step, lane): the lane's two registers, four words.
    let mut e=thread;
    while e<steps*4 {
        let k=(e/4)*16+(e%4)*2;let mut raw=[0u32;4];
        crate::strip_warp::unroll!(4,i,{let kk=k+(i/2)*8+i%2;if kk<terms {raw[i]=*x.add(f.b(outer,kk,0));}});
        crate::ptx::store(stage+(e*8) as u32,f32::from_bits((raw[0]>>16)|(raw[1]&0xffff_0000)));
        crate::ptx::store(stage+(e*8+4) as u32,f32::from_bits((raw[2]>>16)|(raw[3]&0xffff_0000)));
        e+=tc::THREADS as usize;
    }
    crate::ptx::sync();
    if mw>=m_all {return;}
    // Two adjacent BF16 weights of one row: the checkpoint's own bits, column `k` low.
    let wload=#[inline(always)] |m:usize,k:usize|->u32 {
        // The guard comes before the pointer: a padded column names no element of the weights.
        if m>=m_all || k>=terms {return 0;}
        let q=w.add(f.weight(outer,m,k));
        if k+1<terms && (q as usize)&3==0 {return *q.cast::<u32>();}
        let mut packed=*q as u32;
        if k+1<terms {packed|=(*q.add(1) as u32)<<16;}
        packed
    };
    let batch=#[inline(always)] |s0:usize|->[[u32;4];4] {
        let mut a=[[0u32;4];4];
        crate::strip_warp::unroll!(4,i,{let s=s0+i;if s<steps {let (k,m)=(s*tc::STEP+(lane%4)*2,mw+lane/4);
            a[i]=[wload(m,k),wload(m+8,k),wload(m,k+8),wload(m+8,k+8)];}});
        a
    };
    let (mut acc,mut p0,mut p2)=(ZERO,ZERO,ZERO);let (mut part,mut s0)=(0usize,0usize);
    let mut ahead=batch(0);
    while s0<steps {
        hint(s0+8);
        let now=ahead;
        ahead=if s0+4<steps {batch(s0+4)} else {[[0u32;4];4]};
        crate::strip_warp::unroll!(4,i,{let s=s0+i;if s<steps {
            while part<3 && s>=tc::range_end(steps,part) {close!(part,acc,p0,p2);}
            let b=if lane<4 {crate::ptx::load2(stage+((s*4+lane)*8) as u32)} else {[0u32;2]};
            acc=madd(now[i],b,acc);
        }});
        s0+=4;
    }
    while part<4 {close!(part,acc,p0,p2);}
    // Column 0 of C: accumulator registers 0 and 2 of the lanes 0, 4, 8, .. (`absorb_tc::c_place`).
    if lane%4==0 {
        let (m,n)=(mw+lane/4,0usize);
        if m<m_all {out.add(f.out(outer,m,n)).write(crate::latent_absorb::round(acc.a));}
        if m+8<m_all {out.add(f.out(outer,m+8,n)).write(crate::latent_absorb::round(acc.c));}
    }
}};}
/// The step form of scores and values (one token): the word of `run`, with the four ranges of a sum side by side.
/// A block owns 2 M tiles and 32 columns of N. Warp `w` owns M tile `w / 4` and range `w % 4`: it sums the steps of
/// its range in ascending order from +0.0, eight steps a staged chunk. All threads stage the B operand of the chunk
/// (the 4 x 8 steps of the four ranges); each thread issues its loads before it packs or stores any of them. A warp
/// issues the A loads of the chunk's eight steps before the barrier and uses them after it. The ranges then meet in
/// shared memory: range 1 joins range 0 and range 3 joins range 2, then range 2 joins range 0
/// (`fl(fl(p0 + p1) + fl(p2 + p3))`), and the warp of range 0 writes the words.
macro_rules! attention_step {($p:expr,$scores:literal,$stage:expr)=>{{
    const SCORES:bool=$scores;
    let p:crate::causal_ops::Arguments=$p;
    let Some(f)=Form::step(p.shape) else {asm!("trap;");return;};
    if _block_dim_x() as u32!=tc::THREADS || f.kind!=(if SCORES {Kind::Scores} else {Kind::Values}) {asm!("trap;");}
    let thread=_thread_idx_x() as usize;let (warp,lane)=(thread/32,thread%32);
    let (outer,m0,n0)=f.step_block(_block_idx_x() as usize);
    if outer>=f.outer() {return;}
    let (m_all,n_all)=(f.m(),f.n());
    let (x,latent,selected,out)=(p.input[0] as *const u32,p.input[1] as *const u32,p.input[2] as *const f32,p.output as *mut f32);
    let word=*selected;
    if !(word>=0.0 && word==word as u32 as f32) || word as usize>f.c {asm!("trap;");}
    let count=word as usize;
    let terms=f.terms(count);let steps=tc::steps(terms);
    let (tile_m,part)=(warp/4,warp%4);let mw=m0+tile_m*16;
    let live=mw<if SCORES {count} else {m_all};
    let (start,end)=(steps*part/4,steps*(part+1)/4);
    let chunks=steps.div_ceil(4).div_ceil(tc::STEP_CHUNK);
    let stage:u32=$stage;
    // Staging: this thread packs lane `lane` of N tile `warp % 4` for the staged steps `thread / 128 + 2 i`.
    let tile=warp%tc::TILES;
    // Scores: the two cache rows of this lane (the gathered rows of the token: slot j is row j).
    let slot=mw+lane/4;
    let row0=if SCORES && slot<count {latent.add(slot*f.k)} else {core::ptr::null()};
    let row1=if SCORES && slot+8<count {latent.add((slot+8)*f.k)} else {core::ptr::null()};
    // Two adjacent words of one row, column `k` low. One 8-byte load when the pair is aligned and whole.
    let sload=|row:*const u32,k:usize|->u64 {
        // The guard comes before the pointer: a padded column names no word of the row.
        if row.is_null() || k>=terms {return 0;}
        let q=row.add(k);
        if k+1<terms && (q as usize)&7==0 {return *q.cast::<u64>();}
        let mut both=*q as u64;
        if k+1<terms {both|=(*q.add(1) as u64)<<32;}
        both
    };
    let sbatch=|s0:usize|->[[u64;4];4] {
        let mut a=[[0u64;4];4];
        crate::strip_warp::unroll!(4,i,{let s=s0+i;if s<end {let k=s*tc::STEP+(lane%4)*2;
            a[i]=[sload(row0,k),sload(row1,k),sload(row0,k+8),sload(row1,k+8)];}});
        a
    };
    // Values: feature `m` of the cache row of slot `at`.
    let vload=|at:usize,m:usize|->u32 {if at<count && m<m_all {*latent.add(at*f.k+m)} else {0}};
    let vbatch=|s0:usize|->[[u32;8];4] {
        let mut a=[[0u32;8];4];
        crate::strip_warp::unroll!(4,i,{let s=s0+i;if s<end {let (k,m)=(s*tc::STEP+(lane%4)*2,mw+lane/4);
            a[i]=[vload(k,m),vload(k+1,m),vload(k,m+8),vload(k+1,m+8),vload(k+8,m),vload(k+9,m),vload(k+8,m+8),vload(k+9,m+8)];}});
        a
    };
    let spack=|both:u64|->u32 {((both>>16)&0xffff) as u32|((both>>32)&0xffff_0000) as u32};
    let vpack=|low:u32,high:u32|->u32 {(low>>16)|(high&0xffff_0000)};
    let mut acc=[ZERO;tc::TILES];let mut chunk=0usize;
    while chunk<chunks {
        if n0+tile*8<n_all {
            let mut group=0usize;
            while group<4 {
                let mut raw=[[0u32;4];4];
                crate::strip_warp::unroll!(4,j,{
                    let staged=thread/128+(group*4+j)*2;let (r,q)=(staged/tc::STEP_CHUNK,staged%tc::STEP_CHUNK);
                    let s=steps*r/4+chunk*tc::STEP_CHUNK+q;
                    if s<steps*(r+1)/4 {crate::strip_warp::unroll!(2,reg,{
                        let (kk,nn)=tc::b_place(lane,reg);let (k,n)=(s*tc::STEP+kk,n0+tile*8+nn);
                        if n<n_all {
                            if k<terms {raw[j][reg*2]=*x.add(f.b(outer,k,n));}
                            if k+1<terms {raw[j][reg*2+1]=*x.add(f.b(outer,k+1,n));}
                        }
                    });}
                });
                crate::strip_warp::unroll!(4,j,{
                    let at=stage+tc::stage_at(thread/128+(group*4+j)*2,tile,lane) as u32;
                    crate::ptx::store(at,f32::from_bits(vpack(raw[j][0],raw[j][1])));
                    crate::ptx::store(at+4,f32::from_bits(vpack(raw[j][2],raw[j][3])));
                });
                group+=1;
            }
        }
        let base=start+chunk*tc::STEP_CHUNK;
        let (mut s_now,mut s_next,mut v_now,mut v_next)=([[0u64;4];4],[[0u64;4];4],[[0u32;8];4],[[0u32;8];4]);
        if live {
            if SCORES {s_now=sbatch(base);s_next=sbatch(base+4);} else {v_now=vbatch(base);v_next=vbatch(base+4);}
        }
        crate::ptx::sync();
        if live {
            crate::strip_warp::unroll!(4,i,{if base+i<end {
                let a=if SCORES {[spack(s_now[i][0]),spack(s_now[i][1]),spack(s_now[i][2]),spack(s_now[i][3])]}
                    else {[vpack(v_now[i][0],v_now[i][1]),vpack(v_now[i][2],v_now[i][3]),vpack(v_now[i][4],v_now[i][5]),vpack(v_now[i][6],v_now[i][7])]};
                tiles!(T=>{if n0+T*8<n_all {acc[T]=madd(a,crate::ptx::load2(stage+tc::stage_at(part*tc::STEP_CHUNK+i,T,lane) as u32),acc[T]);}});
            }});
            crate::strip_warp::unroll!(4,i,{if base+4+i<end {
                let a=if SCORES {[spack(s_next[i][0]),spack(s_next[i][1]),spack(s_next[i][2]),spack(s_next[i][3])]}
                    else {[vpack(v_next[i][0],v_next[i][1]),vpack(v_next[i][2],v_next[i][3]),vpack(v_next[i][4],v_next[i][5]),vpack(v_next[i][6],v_next[i][7])]};
                tiles!(T=>{if n0+T*8<n_all {acc[T]=madd(a,crate::ptx::load2(stage+tc::stage_at(part*tc::STEP_CHUNK+4+i,T,lane) as u32),acc[T]);}});
            }});
        }
        // Every warp is done with the chunk before it is staged again (or before the join reuses the stage).
        crate::ptx::sync();
        chunk+=1;
    }
    // The join. A handed range lies at 16 bytes a lane: [M tile and pair][N tile][lane].
    if part&1==1 {hand!(stage,acc,n0,n_all,lane,tile_m*2+part/2);}
    crate::ptx::sync();
    if part&1==0 {take!(stage,acc,n0,n_all,lane,tile_m*2+part/2);}
    crate::ptx::sync();
    if part==2 {hand!(stage,acc,n0,n_all,lane,tile_m);}
    crate::ptx::sync();
    if part!=0 {return;}
    take!(stage,acc,n0,n_all,lane,tile_m);
    let scale=1.0/crate::causal_ops::root(f.d as f32);
    tiles!(T=>{
        let sums=[acc[T].a,acc[T].b,acc[T].c,acc[T].d];let mut reg=0;
        while reg<4 {
            let (mm,nn)=tc::c_place(lane,reg);let (m,n)=(mw+mm,n0+T*8+nn);
            if m<m_all && n<n_all {
                let sum=sums[reg];
                let value=if SCORES {
                    if m<count {let scaled:f32;
                        asm!("mul.rn.f32 {v}, {a}, {b};",v=out(reg32)scaled,a=in(reg32)sum,b=in(reg32)scale,options(nomem,nostack));scaled} else {0.0}
                } else {crate::latent_absorb::round(sum)};
                out.add(f.out(outer,m,n)).write(value);
            }
            reg+=1;
        }
    });
}};}
/// The step form: the causal backend validates `latent_absorb::lengths` and launches `causal_ops::absorb_blocks` blocks
/// of `absorb_tc::THREADS` threads. The latent rows are the gathered rows of the one token, in selection order; the
/// count is one word.
#[inline(always)]
pub(crate) fn resident_latent_absorption(p:crate::causal_ops::Arguments) {
    // SAFETY: the backend validates every extent with the shape's `lengths` and launches `absorb_tc::THREADS` threads
    // a block (any other block size, or an operation that is not query or project, traps here as a whole block). A
    // block past `Form::step_blocks` names a head past the last one and returns as a whole block before any read.
    // Every global read is below an extent the host checked: the one token's row below `terms`, weights inside
    // [heads][key + value rows][latent] (a row at or past `m` is not read; a hint reads nothing). The 2,048 shared
    // bytes belong to the block; the staged steps fit them or the block traps. A thread stores only its own staged
    // words, before the one barrier, and reads staged words only after it. All 256 threads reach that barrier; a warp
    // whose M tile holds no row leaves after it. Each thread writes its own in-range output words once; threads and
    // blocks own disjoint words. Inputs and the private output are disjoint and stay live on the ordered stream.
    unsafe {dense_step!(p);}
}
#[inline(always)]
pub(crate) fn resident_latent_absorption_scores(p:crate::causal_ops::Arguments) {
    // SAFETY: the backend validates every extent with the shape's `lengths` and launches `absorb_tc::THREADS` threads
    // a block (any other block size or operation traps here, as a whole block). A block past `Form::step_blocks`
    // names a token past the one token and returns as a whole block before any read. The count is a finite integer at
    // most the rows of the shape, or the block traps before any read that would use it. Every global read is below an
    // extent the host checked: the absorbed query below `terms` and the heads, cache rows below the count. The 32,768
    // shared bytes belong to the block. A thread stores only its own staged words (its lane of its N tile, or its
    // lane of its M tile and range in the join), and a staged word is read only after the barrier that follows its
    // store and before the barrier that precedes its next store. All 256 threads reach every barrier: the chunk count
    // comes from the shape and the one count word that the whole block shares, and a thread leaves only before the
    // first barrier (a whole block) or after the last one. Each thread writes its own in-range output words once;
    // threads and blocks own disjoint words. Inputs and the private output are disjoint and stay live on the stream.
    unsafe {attention_step!(p,true,crate::ptx::shared::<{tc::STAGE_BYTES}>());}
}
#[inline(always)]
pub(crate) fn resident_latent_absorption_values(p:crate::causal_ops::Arguments) {
    // SAFETY: as `resident_latent_absorption_scores`, for the values: the probabilities are read below the count and
    // the heads, and a cache row is read only for a slot below the count and a feature below the latent rank.
    unsafe {attention_step!(p,false,crate::ptx::shared::<{tc::STAGE_BYTES}>());}
}
/// The merged attention entry: scores or values by the shape's operation, on ONE block array. Two arrays (one a body)
/// add up to 65,536 static shared bytes in one entry, which the offline compiler refuses (its limit is 49,152).
#[inline(always)]
pub(crate) fn resident_latent_absorption_attention(p:crate::causal_ops::Arguments) {
    // SAFETY: as `resident_latent_absorption_scores` and `resident_latent_absorption_values`: a block runs exactly one
    // of the two bodies (the operation is the same for every thread of the launch), so the 32,768 shared bytes belong to
    // that body alone.
    unsafe {
        let stage=crate::ptx::shared::<{tc::STAGE_BYTES}>();
        if p.shape.op==crate::causal_ops::Op::AbsorbScores {attention_step!(p,true,stage);} else {attention_step!(p,false,stage);}
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn latent_absorption(p:crate::causal_ops::Arguments) {resident_latent_absorption(p)}
#[no_mangle]
pub extern "ptx-kernel" fn latent_absorption_scores(p:crate::causal_ops::Arguments) {resident_latent_absorption_scores(p)}
#[no_mangle]
pub extern "ptx-kernel" fn latent_absorption_values(p:crate::causal_ops::Arguments) {resident_latent_absorption_values(p)}
