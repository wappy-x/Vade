//! Latent absorption of a check pass (1 to `absorb_tc::SMALL_TOKENS` token rows) on the bodies of the step form, so a
//! pass of m rows costs about what the one-row step costs. The word of every output is the word of
//! `latent_absorption_tile` and of `absorb_tc::word`: the operands' high halves, 16-term steps in ascending `k` (the last
//! filled with +0.0), four ranges cut by the step count only, each one ascending chain from +0.0, the join
//! `fl(fl(p0 + p1) + fl(p2 + p3))`, then the operation's last step. Only WHERE a word is computed differs (contract:
//! `gpu/src/absorb_tc.rs`; device proof: `kernel-bench --latent-rows`, every output word against the tile of the same image).
//! - `latent_rows_dense` (query, project): the tile's small body (`latent_absorption_tile.rs`) alone in an entry that
//!   holds only the shared bytes it stages (`latent_absorb_rows::ROWS_DENSE_STAGE`), not the tile's 34,816.
//! - `latent_rows_attention` (scores, values): the step form's body (`latent_absorption.rs`, `attention_step`) for each
//!   token of the launch. A block owns one token, 2 M tiles (32 rows) and 32 columns of N; warp `w` owns M tile `w / 4`
//!   and range `w % 4`, so the four ranges of a sum run side by side and meet in the join. A token's cache rows are the
//!   rows its row of the selection table names (each ID checked against the row bound), its count the last word there.
use crate::{absorb_tc::{self as tc,Form,Kind},canonical::{ZERO,add,madd},latent_absorption::tiles};
use core::arch::{asm,nvptx::{_thread_idx_x,_block_idx_x,_block_dim_x}};
/// Close range `part` of the one accumulator: ranges 0 and 1 join into `p0`, range 2 waits in `p2`, range 3 joins the word.
macro_rules! close_one {($part:ident,$acc:ident,$p0:ident,$p2:ident)=>{
    if $part==0 {$p0=$acc;$acc=ZERO;} else if $part==1 {$p0=add($p0,$acc);$acc=ZERO;} else if $part==2 {$p2=$acc;$acc=ZERO;} else {$acc=add($p0,add($p2,$acc));}
    $part+=1;
};}
/// The join of the side-by-side ranges: a warp hands its four accumulators of each live N tile to shared memory, or adds
/// the handed ones to its own (one FP32 add a word). As `latent_absorption.rs`.
macro_rules! hand {($stage:ident,$acc:ident,$n0:ident,$n_all:ident,$lane:ident,$index:expr)=>{tiles!(T=>{if $n0+T*8<$n_all {
    let at=$stage+(((($index)*tc::TILES+T)*32+$lane)*16) as u32;
    crate::ptx::store(at,$acc[T].a);crate::ptx::store(at+4,$acc[T].b);crate::ptx::store(at+8,$acc[T].c);crate::ptx::store(at+12,$acc[T].d);
}})};}
macro_rules! take {($stage:ident,$acc:ident,$n0:ident,$n_all:ident,$lane:ident,$index:expr)=>{tiles!(T=>{if $n0+T*8<$n_all {
    let o=crate::ptx::load4($stage+(((($index)*tc::TILES+T)*32+$lane)*16) as u32);
    $acc[T]=add($acc[T],crate::canonical::Acc {a:f32::from_bits(o[0]),b:f32::from_bits(o[1]),c:f32::from_bits(o[2]),d:f32::from_bits(o[3])});
}})};}
/// Query and project of 1 to `SMALL_TOKENS` tokens: the small body of `latent_absorption_tile`, token for token. A block
/// owns one head and 128 rows of M (the grid of `Form::block`, one N tile); a warp owns one M tile and runs the four ranges
/// in ascending order, its weight loads issued four steps before their products and L2 hints eight steps ahead. Every
/// step of B is staged at once: lane l holds column l / 4 (a token).
macro_rules! dense_rows {($p:expr)=>{{
    let p:crate::prompt_ops::Arguments=$p;
    let Some(f)=Form::rows(p.shape) else {asm!("trap;");return;};
    if _block_dim_x() as u32!=tc::THREADS || f.attention() || f.n()>tc::SMALL_TOKENS {asm!("trap;");}
    let thread=_thread_idx_x() as usize;let (warp,lane)=(thread/32,thread%32);
    let (outer,m0,_)=f.block(_block_idx_x() as usize);
    if outer>=f.outer() {return;}
    let (m_all,n_all)=(f.m(),f.n());let terms=f.terms(0);let steps=tc::steps(terms);
    if steps*tc::SMALL_STEP_BYTES>crate::latent_absorb_rows::ROWS_DENSE_STAGE {asm!("trap;");}
    let (x,w,out,selected)=(p.input[0] as *const u32,p.weight as *const u16,p.output as *mut f32,p.input[1] as *const f32);
    let stage=crate::ptx::shared::<{crate::latent_absorb_rows::ROWS_DENSE_STAGE}>();
    let mw=m0+warp*16;
    // The weight lines of the four steps from `s` in this warp's 16 rows (2 lines of 32 weights a row): one hint a lane.
    let hint=|s:usize| {
        let (m,k)=(mw+lane/2,s*tc::STEP+(lane%2)*32);
        if m<m_all && k<terms {asm!("prefetch.global.L2 [{p}];",p=in(reg64)w.cast::<u8>().add(f.weight(outer,m,k)*2),options(nostack,readonly));}
    };
    hint(0);hint(4);
    // B: entry `e` is (step, lane). Lane l holds column l / 4 (a token) at the step's rows (l % 4) * 2 + {0, 1, 8, 9}
    // (`absorb_tc::b_place`), two packed registers; a column past the tokens or a row past the terms is +0.0.
    let mut e=thread;
    while e<steps*32 {
        let (s,l)=(e/32,e%32);let (n,k)=(l/4,s*tc::STEP+(l%4)*2);let mut raw=[0u32;4];
        if n<n_all {crate::strip_warp::unroll!(4,i,{let kk=k+(i/2)*8+i%2;if kk<terms {raw[i]=*x.add(f.b(outer,kk,n));}});}
        crate::ptx::store(stage+(e*8) as u32,f32::from_bits((raw[0]>>16)|(raw[1]&0xffff_0000)));
        crate::ptx::store(stage+(e*8+4) as u32,f32::from_bits((raw[2]>>16)|(raw[3]&0xffff_0000)));
        e+=tc::THREADS as usize;
    }
    crate::ptx::sync();
    if mw>=m_all {return;}
    // Two adjacent BF16 weights of one row: the checkpoint's own bits, column `k` low. A padded row or column is +0.0.
    let wload=|m:usize,k:usize|->u32 {
        // The guard comes before the pointer: a padded column names no element of the weights.
        if m>=m_all || k>=terms {return 0;}
        let q=w.add(f.weight(outer,m,k));
        if k+1<terms && (q as usize)&3==0 {return *q.cast::<u32>();}
        let mut packed=*q as u32;
        if k+1<terms {packed|=(*q.add(1) as u32)<<16;}
        packed
    };
    let batch=|s0:usize|->[[u32;4];4] {
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
            while part<3 && s>=tc::range_end(steps,part) {close_one!(part,acc,p0,p2);}
            acc=madd(now[i],crate::ptx::load2(stage+((s*32+lane)*8) as u32),acc);
        }});
        s0+=4;
    }
    while part<4 {close_one!(part,acc,p0,p2);}
    let sums=[acc.a,acc.b,acc.c,acc.d];
    crate::strip_warp::unroll!(4,reg,{
        let (mm,n)=tc::c_place(lane,reg);let m=mw+mm;
        if m<m_all && n<n_all {
            // Project: a token with no selected row writes +0.0, as the tile does (its count word checked the same way).
            let empty=f.kind==Kind::Project && f.table!=0 && {
                let raw=*selected.add(f.count_word(n));
                if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>f.c {asm!("trap;");}
                raw as usize==0
            };
            out.add(f.out(outer,m,n)).write(if empty {0.0} else {crate::latent_absorb::round(sums[reg])});
        }
    });
}};}
/// Scores and values of 1 to `SMALL_TOKENS` tokens: the step form's body for each token, the four ranges side by side.
/// A block owns one token, 2 M tiles and 32 columns of N (`Form::step_block`). Warp `w` owns M tile `w / 4` and range
/// `w % 4`: it sums the steps of its range in ascending order from +0.0, eight steps a staged chunk. All threads stage the
/// B operand of the chunk (the 4 x 8 steps of the four ranges) and, for values, the cache row of each slot of those steps
/// (a word offset, or `NO_ROW` past the count); each thread issues its loads before it packs or stores any of them. The
/// ranges then meet in shared memory as `latent_absorption.rs` does, and the warp of range 0 writes the words.
macro_rules! attention_rows {($p:expr,$scores:literal,$stage:expr)=>{{
    const SCORES:bool=$scores;
    let p:crate::prompt_ops::Arguments=$p;
    let Some(f)=Form::rows(p.shape) else {asm!("trap;");return;};
    if _block_dim_x() as u32!=tc::THREADS || f.kind!=(if SCORES {Kind::Scores} else {Kind::Values}) || f.packed || f.table==0 || f.tokens>tc::SMALL_TOKENS {asm!("trap;");}
    let thread=_thread_idx_x() as usize;let (warp,lane)=(thread/32,thread%32);
    let (outer,m0,n0)=f.step_block(_block_idx_x() as usize);
    if outer>=f.outer() {return;}
    let (m_all,n_all)=(f.m(),f.n());
    let (x,latent,selected,out)=(p.input[0] as *const u32,p.input[1] as *const u32,p.input[2] as *const f32,p.output as *mut f32);
    // The token's count: the last word of its selection row, a finite integer at most the score capacity, or the block traps.
    let word=*selected.add(f.count_word(outer));
    if !(word>=0.0 && word==word as u32 as f32) || word as usize>f.c {asm!("trap;");}
    let count=word as usize;
    // A selected ID: a finite integer below the row bound, or the block traps.
    let checked=|raw:f32|->usize {
        if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>=f.rows {asm!("trap;");}
        raw as usize
    };
    let table=selected.add(outer*f.table);
    let terms=f.terms(count);let steps=tc::steps(terms);
    let (tile_m,part)=(warp/4,warp%4);let mw=m0+tile_m*16;
    let live=mw<if SCORES {count} else {m_all};
    let (start,end)=(steps*part/4,steps*(part+1)/4);
    let chunks=steps.div_ceil(4).div_ceil(tc::STEP_CHUNK);
    let stage:u32=$stage;let ids=stage+tc::STAGE_BYTES as u32;
    // Staging: this thread packs lane `lane` of N tile `warp % 4` for the staged steps `thread / 128 + 2 i`.
    let tile=warp%tc::TILES;
    // Scores: the selection words of this lane's two slots, loaded now and checked after the first chunk's staging loads.
    let slot=mw+lane/4;
    let raw0=if SCORES && slot<count {*table.add(slot)} else {0.0};
    let raw1=if SCORES && slot+8<count {*table.add(slot+8)} else {0.0};
    let (mut row0,mut row1)=(core::ptr::null::<u32>(),core::ptr::null::<u32>());
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
    // Values: feature `m` of the cache row at word offset `at` (`NO_ROW`: a slot past the count, +0.0).
    let vload=|at:u32,m:usize|->u32 {if at!=tc::NO_ROW && m<m_all {*latent.add(at as usize+m)} else {0}};
    let spack=|both:u64|->u32 {((both>>16)&0xffff) as u32|((both>>32)&0xffff_0000) as u32};
    let vpack=|low:u32,high:u32|->u32 {(low>>16)|(high&0xffff_0000)};
    let mut acc=[ZERO;tc::TILES];let mut chunk=0usize;
    while chunk<chunks {
        let base=start+chunk*tc::STEP_CHUNK;
        // Values: the selection words of the chunk's slots. Entry `e` is (range, chunk step, slot of the step): two a
        // thread, loaded before the B operand's loads and checked after them.
        let (mut at,mut raw)=([tc::NO_ROW;2],[0.0f32;2]);
        if !SCORES {crate::strip_warp::unroll!(2,j,{
            let e=thread+j*tc::THREADS as usize;let (r,q,i)=(e/128,e%128/16,e%16);
            let s=steps*r/4+chunk*tc::STEP_CHUNK+q;let at_slot=s*tc::STEP+i;
            if s<steps*(r+1)/4 && at_slot<count {raw[j]=*table.add(at_slot);at[j]=0;}
        });}
        if n0+tile*8<n_all {
            let mut group=0usize;
            while group<4 {
                let mut words=[[0u32;4];4];
                crate::strip_warp::unroll!(4,j,{
                    let staged=thread/128+(group*4+j)*2;let (r,q)=(staged/tc::STEP_CHUNK,staged%tc::STEP_CHUNK);
                    let s=steps*r/4+chunk*tc::STEP_CHUNK+q;
                    if s<steps*(r+1)/4 {crate::strip_warp::unroll!(2,reg,{
                        let (kk,nn)=tc::b_place(lane,reg);let (k,n)=(s*tc::STEP+kk,n0+tile*8+nn);
                        if n<n_all {
                            if k<terms {words[j][reg*2]=*x.add(f.b(outer,k,n));}
                            if k+1<terms {words[j][reg*2+1]=*x.add(f.b(outer,k+1,n));}
                        }
                    });}
                });
                crate::strip_warp::unroll!(4,j,{
                    let place=stage+tc::stage_at(thread/128+(group*4+j)*2,tile,lane) as u32;
                    crate::ptx::store(place,f32::from_bits(vpack(words[j][0],words[j][1])));
                    crate::ptx::store(place+4,f32::from_bits(vpack(words[j][2],words[j][3])));
                });
                group+=1;
            }
        }
        if !SCORES {crate::strip_warp::unroll!(2,j,{
            let e=thread+j*tc::THREADS as usize;
            if at[j]!=tc::NO_ROW {at[j]=(checked(raw[j])*f.k) as u32;}
            crate::ptx::store(ids+(e*4) as u32,f32::from_bits(at[j]));
        });}
        if SCORES && chunk==0 {
            if slot<count {row0=latent.add(checked(raw0)*f.k);}
            if slot+8<count {row1=latent.add(checked(raw1)*f.k);}
        }
        let sbatch=|s0:usize|->[[u64;4];4] {
            let mut a=[[0u64;4];4];
            crate::strip_warp::unroll!(4,i,{let s=s0+i;if s<end {let k=s*tc::STEP+(lane%4)*2;
                a[i]=[sload(row0,k),sload(row1,k),sload(row0,k+8),sload(row1,k+8)];}});
            a
        };
        // Values: the four slots of this lane's A registers at each step, through the staged word offsets.
        let vbatch=|s0:usize|->[[u32;8];4] {
            let mut a=[[0u32;8];4];
            crate::strip_warp::unroll!(4,i,{let s=s0+i;if s<end {
                let e=part*128+(s-base)*tc::STEP+(lane%4)*2;let m=mw+lane/4;
                let low=crate::ptx::load2(ids+(e*4) as u32);let high=crate::ptx::load2(ids+((e+8)*4) as u32);
                a[i]=[vload(low[0],m),vload(low[1],m),vload(low[0],m+8),vload(low[1],m+8),vload(high[0],m),vload(high[1],m),vload(high[0],m+8),vload(high[1],m+8)];
            }});
            a
        };
        let (mut s_now,mut s_next,mut v_now,mut v_next)=([[0u64;4];4],[[0u64;4];4],[[0u32;8];4],[[0u32;8];4]);
        if live && SCORES {s_now=sbatch(base);s_next=sbatch(base+4);}
        crate::ptx::sync();
        if live && !SCORES {v_now=vbatch(base);v_next=vbatch(base+4);}
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
/// Query and project of a check pass: `latent_absorb_rows::launch` validates `latent_absorb_rows::lengths` and launches
/// `Form::blocks` blocks of `absorb_tc::THREADS` threads for 1 to `SMALL_TOKENS` tokens whose staged steps fit the stage.
/// The direct entry and the recorded entry (`resident_launch_latent_rows_dense`) run this one body with the same
/// arguments and grid.
#[export_name="latent_rows_dense"]
pub extern "ptx-kernel" fn entry_latent_rows_dense(p:crate::prompt_ops::Arguments) {latent_rows_dense(p)}
#[inline(always)]
pub(crate) fn latent_rows_dense(p:crate::prompt_ops::Arguments) {
    // SAFETY: the backend validates every extent with the shape's `lengths` and launches `absorb_tc::THREADS` threads a
    // block; any other block size, a scores or values shape, more than `SMALL_TOKENS` tokens, or staged steps past the
    // `ROWS_DENSE_STAGE` shared bytes trap here as a whole block. A block past `Form::blocks` names a head past the last
    // one and returns as a whole block before any read. Every global read is below an extent the host checked: B words
    // below `terms` and `n`, weights inside [heads][key + value rows][latent] for a row below `m` and a column below
    // `terms` (a hint reads nothing), and a token's count word of the selection table for project (a finite integer at
    // most the score capacity, or the block traps). The shared bytes belong to the block; a thread stores only its own
    // staged words, all 256 threads reach the one barrier before any staged read, and only then does a warp with no row
    // of M return. Each thread writes its own in-range output words once; threads and blocks own disjoint words. Inputs
    // and the private output are disjoint and stay live on the ordered stream. The accumulators live in registers.
    unsafe {dense_rows!(p);}
}
/// Scores and values of a check pass: `latent_absorb_rows::launch` validates `latent_absorb_rows::lengths` and launches
/// `Form::step_blocks` blocks of `absorb_tc::THREADS` threads for 1 to `SMALL_TOKENS` tokens (unpacked cache rows).
/// The direct entry and the recorded entry (`resident_launch_latent_rows_attention`) run this one body with the same
/// arguments and grid.
#[export_name="latent_rows_attention"]
pub extern "ptx-kernel" fn entry_latent_rows_attention(p:crate::prompt_ops::Arguments) {latent_rows_attention(p)}
#[inline(always)]
pub(crate) fn latent_rows_attention(p:crate::prompt_ops::Arguments) {
    // SAFETY: the backend validates every extent with the shape's `lengths` and launches `absorb_tc::THREADS` threads a
    // block; any other block size, a query or project or packed shape, or more than `SMALL_TOKENS` tokens trap here as a
    // whole block. A block past `Form::step_blocks` names a token past the last one and returns as a whole block before
    // any read. A token's count is a finite integer at most the score capacity, and a selected ID a finite integer below
    // the row bound, or the block traps before any read that would use it. Every global read is below an extent the host
    // checked: the token's selection row (slots below the count, and its count word), B words below `terms` and the heads,
    // cache rows named by a checked ID below the latent rank (`launch` checks that every such word offset fits 32 bits
    // below `NO_ROW`). The 34,816 shared bytes belong to the block, and a block runs exactly one of the two bodies (the
    // operation is the same for every thread of the launch). A thread stores only its own staged words (its lane of its
    // N tile, its two word offsets, or its lane of its M tile and range in the join), and a staged word is read only
    // after the barrier that follows its store and before the barrier that precedes its next store. All 256 threads
    // reach every barrier: the chunk count comes from the shape and the token's one count word that the whole block
    // shares, and a thread leaves only before the first barrier (a whole block) or after the last one. Each thread writes
    // its own in-range output words once; threads and blocks own disjoint words. Inputs and the private output are
    // disjoint and stay live on the ordered stream.
    unsafe {
        let stage=crate::ptx::shared::<{tc::STAGE_BYTES+tc::ID_BYTES}>();
        if matches!(p.shape.op,crate::prompt_ops::Op::AbsorbScores) {attention_rows!(p,true,stage);} else {attention_rows!(p,false,stage);}
    }
}
