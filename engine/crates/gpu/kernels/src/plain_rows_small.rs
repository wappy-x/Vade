//! Plain token rows for a SMALL token count (1 to 16): a draft verify pass. Same arithmetic and order
//! as the `prompt_rows` ProjectRows fast path: every output `[row][token]` is ONE ascending FP32 chain
//! over the columns, from 0.0, each product rounded before its add, no fused multiply-add. Nothing is
//! split along the columns, so no sum changes order.
//!
//! A block takes 32 weight rows (a thread owns one row) and all tokens; warp `g` takes tokens
//! `4g..4g+4`, so a block has a fixed 4 warps (only `ceil(tokens/4)` of them compute) and holds up to 4 independent chains a thread.
//! The grid is `ceil(rows/32)` blocks, so a 64-row matrix is no longer one block on one SM. Columns
//! pass through shared memory in chunks of 64 in ascending order, through a ring of `D` (3) stages of
//! `cp.async` copies, so many chunks are in flight while one is computed (a lone warp that loads and
//! waits for each piece is latency bound). A weight row is padded to 68 words so the 16-byte reads
//! of 32 lanes have no bank conflict; input reads are one address for the whole warp.
//!
//! Launch: grid = ceil(rows/32), block = 128, no dynamic shared memory. Same
//! `Arguments` as `prompt_rows`. Needs 1 <= tokens <= 16, cols % 4 == 0 and 16-byte aligned input
//! and weight, one output word per unit; anything else keeps the old kernels.
use crate::prompt_ops::{Arguments,Op};

#[no_mangle]
pub extern "ptx-kernel" fn plain_rows_small(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents, with same-context
    // contiguous owners that live through stream completion. Input and weight are immutable and
    // 16-byte aligned (checked here, else the block traps). `cols` is a positive multiple of 4, so
    // every 16-byte copy is four whole columns of one row; only rows below `rows` and tokens below
    // `tokens` are copied, and a copy never passes `cols`. The output overlaps no input and each
    // (row, token) word has exactly one writer: the thread of that row whose warp holds that token.
    // Every shared write is a cp.async into the ring slot `chunk % D`, inside the block's own static
    // array. A slot is refilled only after the barrier that follows every thread's reads of its
    // previous chunk, and is read only after `cp.async.wait_group` and a barrier that follow its
    // fill: every shared read follows a write. A thread reads only the weight row of its lane and the
    // tokens of its warp, and those were written for this chunk (rows and tokens past the shape are
    // neither copied nor read). All threads of the block reach every barrier; the loop bounds depend
    // only on the shape. The output may be unwritten: each word of an admitted shape is stored once, by the one lane that owns it, through the raw output pointer, and is never read; no reference over the output is built.
    unsafe {
        use core::arch::asm;
        // Ring depth, rows a block, columns a chunk, words a padded weight row.
        const D:usize=3;const RB:usize=32;const KC:usize=64;const SK:usize=68;const Q:usize=KC/4;
        // The L2 prefetch runs this many chunks ahead of the copy.
        const PF:usize=8;
        const WEIGHT_BYTES:usize=RB*SK*4;const STAGE:usize=WEIGHT_BYTES+16*KC*4;
        macro_rules! pending {($n:expr)=>{asm!("cp.async.wait_group {n};",n=const {$n})};}
        let s=p.shape;let (m,rows,k)=(s.tokens as usize,s.rows as usize,s.cols as usize);
        let groups=m.div_ceil(4);let threads=core::arch::nvptx::_block_dim_x() as usize;
        if s.op!=Op::ProjectRows || m==0 || m>16 || k==0 || k%4!=0 || (p.input[0]|p.weight)&15!=0 || p.words!=1 || threads!=128 {asm!("trap;");}
        let smem=crate::ptx::shared::<{D*STAGE}>();
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (lane,group)=(tid%32,tid/32);
        let row0=core::arch::nvptx::_block_idx_x() as usize*RB;
        // A warp past the last token group only copies: it reaches every barrier and computes nothing.
        let active=row0+lane<rows && group<groups;
        let first=group*4;let count=if group<groups {(m-first).min(4)} else {1};
        let (weight,input)=(p.weight as *const f32,p.input[0] as *const f32);
        let chunks=k.div_ceil(KC);
        // Copy chunk `c` of this block's weight rows and of every token row into slot `c % D`.
        let issue=|c:usize| {
            let stage=smem+((c%D)*STAGE) as u32;let col0=c*KC;
            let mut item=tid;
            while item<RB*Q {
                let (r,q)=(item/Q,item%Q);let col=col0+q*4;
                if col<k && row0+r<rows {crate::ptx::copy16(stage+(r*SK*4+q*16) as u32,weight.add((row0+r)*k+col).cast());}
                item+=threads;
            }
            item=tid;
            while item<m*Q {
                let (t,q)=(item/Q,item%Q);let col=col0+q*4;
                if col<k {crate::ptx::copy16(stage+(WEIGHT_BYTES+t*KC*4+q*16) as u32,input.add(t*k+col).cast());}
                item+=threads;
            }
        };
        // Hint the cache line of chunk `c` of every weight row into L2. Reads nothing, writes nothing.
        let hint=|c:usize| {
            let col0=c*KC;
            if col0<k {
                let mut r=tid;
                while r<RB {
                    if row0+r<rows {
                        let mut line=0;
                        while line<KC/32 && col0+line*32<k {asm!("prefetch.global.L2 [{p}];",p=in(reg64)weight.add((row0+r)*k+col0+line*32),options(nostack,readonly));line+=1;}
                    }
                    r+=threads;
                }
            }
        };
        // The whole chunk loop for `$t` tokens in this warp. Returns the chains.
        macro_rules! run {($t:literal)=>{{
            let mut sum=[0.0f32;$t];
            let mut c=0;
            while c<D-1 {if c<chunks {issue(c);}crate::ptx::commit();c+=1;}
            c=D-1;
            while c<PF {hint(c);c+=1;}
            c=0;
            while c<chunks {
                pending!(D-2);crate::ptx::sync();
                if c+D-1<chunks {issue(c+D-1);}crate::ptx::commit();
                hint(c+PF);
                if active {
                    let stage=smem+((c%D)*STAGE) as u32;let valid=(k-c*KC).min(KC)/4;
                    let mut q=0;
                    // Four quads at a time: all shared loads first, so they overlap, then the chains.
                    while q+4<=valid {
                        let mut w=[[0u32;4];4];let mut x=[[[0u32;4];4];$t];
                        let mut i=0;
                        while i<4 {w[i]=crate::ptx::load4(stage+(lane*SK*4+(q+i)*16) as u32);i+=1;}
                        let mut t=0;
                        while t<$t {
                            i=0;while i<4 {x[t][i]=crate::ptx::load4(stage+(WEIGHT_BYTES+(first+t)*KC*4+(q+i)*16) as u32);i+=1;}
                            t+=1;
                        }
                        i=0;
                        while i<4 {
                            t=0;
                            while t<$t {
                                let mut acc=sum[t];
                                acc+=f32::from_bits(x[t][i][0])*f32::from_bits(w[i][0]);acc+=f32::from_bits(x[t][i][1])*f32::from_bits(w[i][1]);
                                acc+=f32::from_bits(x[t][i][2])*f32::from_bits(w[i][2]);acc+=f32::from_bits(x[t][i][3])*f32::from_bits(w[i][3]);
                                sum[t]=acc;t+=1;
                            }
                            i+=1;
                        }
                        q+=4;
                    }
                    while q<valid {
                        let w=crate::ptx::load4(stage+(lane*SK*4+q*16) as u32);
                        let mut t=0;
                        while t<$t {
                            let x=crate::ptx::load4(stage+(WEIGHT_BYTES+(first+t)*KC*4+q*16) as u32);
                            let mut acc=sum[t];
                            acc+=f32::from_bits(x[0])*f32::from_bits(w[0]);acc+=f32::from_bits(x[1])*f32::from_bits(w[1]);
                            acc+=f32::from_bits(x[2])*f32::from_bits(w[2]);acc+=f32::from_bits(x[3])*f32::from_bits(w[3]);
                            sum[t]=acc;t+=1;
                        }
                        q+=1;
                    }
                }
                c+=1;
            }
            sum
        }};}
        let output=(p.output as *mut f32).wrapping_add((row0+lane)*m+first);
        match count {
            4=>{let v=run!(4);if active {*output=v[0];*output.add(1)=v[1];*output.add(2)=v[2];*output.add(3)=v[3];}}
            3=>{let v=run!(3);if active {*output=v[0];*output.add(1)=v[1];*output.add(2)=v[2];}}
            2=>{let v=run!(2);if active {*output=v[0];*output.add(1)=v[1];}}
            _=>{let v=run!(1);if active {*output=v[0];}}
        }
    }
}
