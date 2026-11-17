//! Fixed-order latent absorption on the tensor cores, the prompt form (a group of tokens): one body runs the four fold
//! operations (query, scores, values, project). The step form (one token) is `latent_absorption.rs`. The word function
//! and every index rule live in `gpu/src/absorb_tc.rs`; the CPU twin is `gpu/src/tc_twin_bf16.rs` (`fold`).
//!
//! A block owns one outer index, 128 rows of M (8 warps x 16) and 32 columns of N (4 tiles x 8). A warp owns one
//! 16-row M tile for the whole sum: it carries the 4 x (16 x 8) accumulators of its tile through the steps in ascending
//! order, closes the four ranges at the step counts that `absorb_tc::range_end` gives, and joins them
//! `fl(fl(p0 + p1) + fl(p2 + p3))`. The B operand of the block is staged in shared memory as packed BF16 pairs, one
//! chunk of 32 steps at a time; a thread issues the loads of four staged places before it packs or stores any of them
//! (a warp runs in order: a load whose value is used at once makes the warp wait for memory). The A operand goes from
//! global memory into the registers of the step that reads it, one step ahead.
//! A padded row, column or term is a +0.0 operand of a word that is dropped or of a product that adds nothing.
use crate::{absorb_tc::{self as tc,Form,Kind},canonical::{ZERO,add,madd},latent_absorption::tiles};
use core::arch::{asm,nvptx::{_thread_idx_x,_block_idx_x,_block_dim_x}};
/// Close range `part` of every N tile: ranges 0 and 1 join into `p0`, range 2 waits in `p2`, range 3 joins the word.
macro_rules! close_tiles {($part:ident,$acc:ident,$p0:ident,$p2:ident)=>{
    if $part==0 {tiles!(T=>{$p0[T]=$acc[T];$acc[T]=ZERO;});}
    else if $part==1 {tiles!(T=>{$p0[T]=add($p0[T],$acc[T]);$acc[T]=ZERO;});}
    else if $part==2 {tiles!(T=>{$p2[T]=$acc[T];$acc[T]=ZERO;});}
    else {tiles!(T=>{$acc[T]=add($p0[T],add($p2[T],$acc[T]));});}
    $part+=1;
};}
/// Close range `part` of the one accumulator of the small form: as `close_tiles`, for one N tile.
macro_rules! close_one {($part:ident,$acc:ident,$p0:ident,$p2:ident)=>{
    if $part==0 {$p0=$acc;$acc=ZERO;} else if $part==1 {$p0=add($p0,$acc);$acc=ZERO;} else if $part==2 {$p2=$acc;$acc=ZERO;} else {$acc=add($p0,add($p2,$acc));}
    $part+=1;
};}
/// The prompt form: `prompt_operation` validates `latent_absorb_rows::lengths` and launches `latent_absorb_rows::launch`.
#[no_mangle]
pub extern "ptx-kernel" fn latent_absorption_tile(p:crate::prompt_ops::Arguments) {
    // SAFETY: the backend validates every extent with the shape's `lengths` and launches `absorb_tc::THREADS` threads
    // a block (any other block size traps here, as a whole block). A block past `Form::blocks` names an outer index
    // past the last one and returns as a whole block before any read. A count is a finite integer at most the score
    // capacity, and a selected ID is a finite integer below the row bound, or the block traps before the read that
    // would use it. Every global read is below an extent the host checked: B words below `terms` and `n`, A rows below
    // `m` (or below the token's count), weights inside [heads][key + value rows][latent]. The 34,816 shared bytes
    // belong to this block. A thread stores only its own staged words, and reads staged words only after the barrier
    // that follows the stores; a chunk is staged again only after the barrier that ends the steps that read it. All
    // 256 threads reach every barrier: the step and chunk counts come from the shape, or from the one token's count
    // that the whole block shares. Each thread writes its own in-range output words once; threads and blocks own
    // disjoint words. Inputs and the private output are disjoint and stay live on the ordered stream.
    // The small body of query and project stages every step of B at once in the first `steps * SMALL_STEP_BYTES` of
    // the same shared bytes, a size checked against `STAGE_BYTES` before the body is chosen; a thread stores only its
    // own staged words, all 256 threads reach its one barrier before any staged read, and only then does a warp with no
    // row of M return. It reads staged words below that size, A rows below `m` and B words below `terms` and `n`.
    // The accumulators live in registers. A join is an FP32 add of two registers with nothing to contract.
    unsafe {
        let Some(f)=Form::rows(p.shape) else {asm!("trap;");return;};
        let (x,w,out)=(p.input[0] as *const u32,p.weight as *const u16,p.output as *mut f32);
        let (latent,selected)=if f.attention() {(p.input[1] as *const u32,p.input[2] as *const f32)} else {(core::ptr::null::<u32>(),p.input[1] as *const f32)};
        if _block_dim_x() as u32!=tc::THREADS {asm!("trap;");}
        let thread=_thread_idx_x() as usize;let (warp,lane)=(thread/32,thread%32);
        let (outer,m0,n0)=f.block(_block_idx_x() as usize);
        if outer>=f.outer() {return;}
        let stage=crate::ptx::shared::<{tc::STAGE_BYTES+tc::ID_BYTES}>();
        // THE SMALL FORM of query and project (1 to `tc::SMALL_TOKENS` tokens; the grid above is the step form's grid for these
        // counts): the step form's body (`latent_absorption.rs`, `dense_step`) with each token one column of the block's one N
        // tile. A warp owns one M tile and runs the four ranges in ascending order, with its weight loads issued four steps
        // before their products and L2 hints eight steps ahead. The tensor-core step adds column n's products to column n's
        // accumulators only, so each word is the word of its own token below: the same steps, ranges, joins and last step;
        // only the order of the loads differs. All 256 threads reach the one barrier; a warp with no row of M leaves after it.
        if !f.attention() && f.n()<=tc::SMALL_TOKENS && tc::steps(f.terms(0))*tc::SMALL_STEP_BYTES<=tc::STAGE_BYTES {
            let selected=p.input[1] as *const f32;
            let (m_all,n_all)=(f.m(),f.n());let terms=f.terms(0);let steps=tc::steps(terms);let mw=m0+warp*16;
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
                    // Project: a token with no selected row writes +0.0, as below (its count word is checked the same way).
                    let empty=f.kind==Kind::Project && f.table!=0 && {
                        let raw=*selected.add(f.count_word(n));
                        if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>f.c {asm!("trap;");}
                        raw as usize==0
                    };
                    out.add(f.out(outer,m,n)).write(if empty {0.0} else {crate::latent_absorb::round(sums[reg])});
                }
            });
            return;
        }
        let (m_all,n_all)=(f.m(),f.n());
        let (scores,values)=(f.kind==Kind::Scores,f.kind==Kind::Values);
        // A count word: a finite integer at most the score capacity, or the block traps.
        let count_at=|word:usize|->usize {
            let raw=*selected.add(word);
            if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>f.c {asm!("trap;");}
            raw as usize
        };
        let count=if f.attention() {count_at(f.count_word(outer))} else {0};
        // A selected ID: a finite integer below the row bound, or the block traps.
        let checked=|raw:f32|->usize {
            if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>=f.rows {asm!("trap;");}
            raw as usize
        };
        let id=|slot:usize|->usize {if f.table==0 {slot} else {checked(*selected.add(outer*f.table+slot))}};
        let terms=f.terms(count);let steps=tc::steps(terms);
        let ids=stage+tc::STAGE_BYTES as u32;
        let mw=m0+warp*16;
        // The warp has products to sum: its M tile holds a row (scores: a selected row).
        let live=mw<if scores {count} else {m_all};
        // Scores: the two selected cache rows of this lane.
        let (mut row0,mut row1)=(core::ptr::null::<u32>(),core::ptr::null::<u32>());
        if scores {let slot=mw+lane/4;
            if slot<count {row0=latent.add(id(slot)*f.k);}
            if slot+8<count {row1=latent.add(id(slot+8)*f.k);}
        }
        // Two adjacent words of one row as one packed BF16 pair: the high halves, column `k` low. A column at or past
        // `terms` is +0.0. One 8-byte load when the pair is aligned and whole.
        let pair=|row:*const u32,k:usize|->u32 {
            if row.is_null() {return 0;}
            // An address only (`wrapping_add` asks for no bounds): a padded column, at or past `terms`, is never read.
            let q=row.wrapping_add(k);
            if k+1<terms && (q as usize)&7==0 {let both=*q.cast::<u64>();return ((both>>16)&0xffff) as u32|((both>>32)&0xffff_0000) as u32;}
            let mut packed=0u32;
            if k<terms {packed=*q>>16;}
            if k+1<terms {packed|=*q.wrapping_add(1)&0xffff_0000;}
            packed
        };
        // Feature `m` of two selected cache rows as one packed BF16 pair, the first row low. `at`: the rows' word offsets.
        let column=|at:[u32;2],m:usize|->u32 {
            if m>=m_all {return 0;}
            let mut packed=0u32;
            if at[0]!=tc::NO_ROW {packed=*latent.add(at[0] as usize+m)>>16;}
            if at[1]!=tc::NO_ROW {packed|=*latent.add(at[1] as usize+m)&0xffff_0000;}
            packed
        };
        // Two adjacent BF16 weights of one row: the checkpoint's own bits, column `k` low.
        let weights=|m:usize,k:usize|->u32 {
            if m>=m_all {return 0;}
            // An address only (`wrapping_add` asks for no bounds): a padded column, at or past `terms`, is never read.
            let q=w.wrapping_add(f.weight(outer,m,k));
            if k+1<terms && (q as usize)&3==0 {return *q.cast::<u32>();}
            let mut packed=0u32;
            if k<terms {packed=*q as u32;}
            if k+1<terms {packed|=(*q.wrapping_add(1) as u32)<<16;}
            packed
        };
        // The four A registers of step `at` of the chunk that starts at step `first`.
        let load_a=|at:usize,first:usize|->[u32;4] {
            if scores {
                let k=at*tc::STEP+(lane%4)*2;
                [pair(row0,k),pair(row1,k),pair(row0,k+8),pair(row1,k+8)]
            } else if values {
                let (cs,kk,m)=(at-first,(lane%4)*2,mw+lane/4);
                let low=crate::ptx::load2(ids+((cs*tc::STEP+kk)*4) as u32);let high=crate::ptx::load2(ids+((cs*tc::STEP+kk+8)*4) as u32);
                [column(low,m),column(low,m+8),column(high,m),column(high,m+8)]
            } else {
                let (k,m)=(at*tc::STEP+(lane%4)*2,mw+lane/4);
                [weights(m,k),weights(m+8,k),weights(m,k+8),weights(m+8,k+8)]
            }
        };
        let tile=warp%tc::TILES;
        let mut acc=[ZERO;tc::TILES];let mut p0=[ZERO;tc::TILES];let mut p2=[ZERO;tc::TILES];
        let (mut part,mut s,mut chunk)=(0usize,0usize,0usize);
        while chunk*tc::CHUNK<steps {
            let first=chunk*tc::CHUNK;
            // Stage the chunk's B operand: this thread packs its lane of N tile `warp % 4` for the chunk steps
            // `thread / 128 + 2 i`, four steps (16 words) a round of loads.
            if n0+tile*8<n_all {
                let mut group=0usize;
                while group<4 {
                    let mut raw=[[0u32;4];4];
                    crate::strip_warp::unroll!(4,j,{
                        let cs=thread/128+(group*4+j)*2;
                        if first+cs<steps {crate::strip_warp::unroll!(2,reg,{
                            let (kk,nn)=tc::b_place(lane,reg);let (k,n)=((first+cs)*tc::STEP+kk,n0+tile*8+nn);
                            if n<n_all {
                                if k<terms {raw[j][reg*2]=*x.add(f.b(outer,k,n));}
                                if k+1<terms {raw[j][reg*2+1]=*x.add(f.b(outer,k+1,n));}
                            }
                        });}
                    });
                    crate::strip_warp::unroll!(4,j,{
                        let at=stage+tc::stage_at(thread/128+(group*4+j)*2,tile,lane) as u32;
                        crate::ptx::store(at,f32::from_bits((raw[j][0]>>16)|(raw[j][1]&0xffff_0000)));
                        crate::ptx::store(at+4,f32::from_bits((raw[j][2]>>16)|(raw[j][3]&0xffff_0000)));
                    });
                    group+=1;
                }
            }
            // Values: the cache row of each slot of the chunk, in the token's selection order. Both loads, then both checks.
            if values {
                let (slot0,slot1)=(first*tc::STEP+thread,first*tc::STEP+thread+tc::THREADS as usize);
                let raw0=if f.table!=0 && slot0<count {*selected.add(outer*f.table+slot0)} else {0.0};
                let raw1=if f.table!=0 && slot1<count {*selected.add(outer*f.table+slot1)} else {0.0};
                let at0=if slot0<count {((if f.table==0 {slot0} else {checked(raw0)})*f.k) as u32} else {tc::NO_ROW};
                let at1=if slot1<count {((if f.table==0 {slot1} else {checked(raw1)})*f.k) as u32} else {tc::NO_ROW};
                crate::ptx::store(ids+(thread*4) as u32,f32::from_bits(at0));
                crate::ptx::store(ids+((thread+tc::THREADS as usize)*4) as u32,f32::from_bits(at1));
            }
            crate::ptx::sync();
            let end=if first+tc::CHUNK<steps {first+tc::CHUNK} else {steps};
            let mut ahead=if live && s<end {load_a(s,first)} else {[0;4]};
            while s<end {
                while part<3 && s>=tc::range_end(steps,part) {close_tiles!(part,acc,p0,p2);}
                if live {
                    let cs=s-first;let a=ahead;
                    ahead=if s+1<end {load_a(s+1,first)} else {[0;4]};
                    tiles!(T=>{if n0+T*8<n_all {acc[T]=madd(a,crate::ptx::load2(stage+tc::stage_at(cs,T,lane) as u32),acc[T]);}});
                }
                s+=1;
            }
            // Every warp is done with the chunk before it is staged again.
            crate::ptx::sync();
            chunk+=1;
        }
        while part<4 {close_tiles!(part,acc,p0,p2);}
        let scale=1.0/crate::causal_ops::root(f.d as f32);
        tiles!(T=>{
            let sums=[acc[T].a,acc[T].b,acc[T].c,acc[T].d];let mut reg=0;
            while reg<4 {
                let (mm,nn)=tc::c_place(lane,reg);let (m,n)=(mw+mm,n0+T*8+nn);
                if m<m_all && n<n_all {
                    let sum=sums[reg];
                    let value=if scores {
                        if m<count {let scaled:f32;
                            asm!("mul.rn.f32 {v}, {a}, {b};",v=out(reg32)scaled,a=in(reg32)sum,b=in(reg32)scale,options(nomem,nostack));scaled} else {0.0}
                    } else if f.kind==Kind::Project && f.table!=0 && count_at(f.count_word(n))==0 {0.0}
                    else {crate::latent_absorb::round(sum)};
                    out.add(f.out(outer,m,n)).write(value);
                }
                reg+=1;
            }
        });
    }
}
