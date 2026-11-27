//! Plain token rows for 1 to 4 tokens and at most 512 rows (a draft verify pass). Same arithmetic and order as the `prompt_rows` ProjectRows fast
//! path and as `plain_rows_small`: every output `[row][token]` is ONE ascending FP32 chain over the columns, from +0.0, each product rounded before its
//! add, no fused multiply-add. Nothing is split along the columns, so no sum changes order.
//!
//! `plain_rows_small` puts 32 rows in a block and one warp takes 4 tokens, so a call of 64 to 192 rows runs on 2 to 6 blocks and one computing warp
//! holds up to 4 chains: at 4096 columns it takes 45 us (1 token) to 62 us (4 tokens), the copy pipeline and the issue rate of that one warp being the
//! limit, not the bytes. Here a block takes only 8 rows (a thread owns one row; lanes 8 to 31 of each warp are idle) and warp `g` takes token `g`
//! alone, so the 4 warps of a block compute the 4 tokens at the same time and a 192-row call runs on 24 blocks. Columns pass through shared memory in
//! chunks of 256 through a ring of 3 stages of `cp.async` copies, with an L2 hint 2 chunks ahead.
//!
//! Launch: grid = ceil(rows/8), block = 128, no dynamic shared memory. Same `Arguments` as `prompt_rows`. Needs 1 <= tokens <= 4, cols % 4 == 0
//! and 16-byte aligned input and weight, one output word per unit; anything else keeps the other kernels.
use crate::prompt_ops::{Arguments,Op};

#[no_mangle]
pub extern "ptx-kernel" fn plain_rows_few(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents, with same-context contiguous owners that live through stream
    // completion. Input and weight are immutable and 16-byte aligned (checked here, else the block traps). `cols` is a positive multiple of 4, so
    // every 16-byte copy is four whole columns of one row; only rows below `rows` and tokens below `tokens` are copied, and a copy never passes
    // `cols`. The output overlaps no input and each (row, token) word has exactly one writer: the thread of that row in the warp of that token.
    // Every shared write is a cp.async into the ring slot `chunk % D`, inside the block's own static array. A slot is refilled only after the
    // barrier that follows every thread's reads of its previous chunk, and is read only after `cp.async.wait_group` and a barrier that follow its
    // fill: every shared read follows a write. A thread reads only the weight row of its lane and the token row of its warp, and those were written
    // for this chunk (rows and tokens past the shape are neither copied nor read). All threads of the block reach every barrier; the loop bounds
    // depend only on the shape. The output is initialized (zeroed) earlier on the stream.
    unsafe {
        use core::arch::asm;
        // Ring depth, rows a block, columns a chunk, words a padded weight row, tokens at most, L2 hint distance in chunks.
        const D:usize=3;const RB:usize=8;const KC:usize=256;const SK:usize=KC+4;const Q:usize=KC/4;const TOKENS:usize=4;const PF:usize=2;
        const WEIGHT_BYTES:usize=RB*SK*4;const STAGE:usize=WEIGHT_BYTES+TOKENS*KC*4;
        macro_rules! pending {($n:expr)=>{asm!("cp.async.wait_group {n};",n=const {$n})};}
        let s=p.shape;let (m,rows,k)=(s.tokens as usize,s.rows as usize,s.cols as usize);
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if s.op!=Op::ProjectRows || m==0 || m>TOKENS || k==0 || k%4!=0 || (p.input[0]|p.weight)&15!=0 || p.words!=1 || threads!=128 {asm!("trap;");}
        let smem=crate::ptx::shared::<{D*STAGE}>();
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (lane,token)=(tid%32,tid/32);
        let row0=core::arch::nvptx::_block_idx_x() as usize*RB;
        // Lanes past the block's rows and a warp past the last token only copy: they reach every barrier and compute nothing.
        let active=lane<RB && row0+lane<rows && token<m;
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
        // Hint the cache lines of chunk `c` of every weight row into L2. Reads nothing, writes nothing.
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
        let mut sum=0.0f32;
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
                // Four quads at a time: all shared loads first, so they overlap, then the chain.
                while q+4<=valid {
                    let mut w=[[0u32;4];4];let mut x=[[0u32;4];4];
                    let mut i=0;
                    while i<4 {
                        w[i]=crate::ptx::load4(stage+(lane*SK*4+(q+i)*16) as u32);
                        x[i]=crate::ptx::load4(stage+(WEIGHT_BYTES+token*KC*4+(q+i)*16) as u32);
                        i+=1;
                    }
                    i=0;
                    while i<4 {
                        sum+=f32::from_bits(x[i][0])*f32::from_bits(w[i][0]);sum+=f32::from_bits(x[i][1])*f32::from_bits(w[i][1]);
                        sum+=f32::from_bits(x[i][2])*f32::from_bits(w[i][2]);sum+=f32::from_bits(x[i][3])*f32::from_bits(w[i][3]);
                        i+=1;
                    }
                    q+=4;
                }
                while q<valid {
                    let w=crate::ptx::load4(stage+(lane*SK*4+q*16) as u32);
                    let x=crate::ptx::load4(stage+(WEIGHT_BYTES+token*KC*4+q*16) as u32);
                    sum+=f32::from_bits(x[0])*f32::from_bits(w[0]);sum+=f32::from_bits(x[1])*f32::from_bits(w[1]);
                    sum+=f32::from_bits(x[2])*f32::from_bits(w[2]);sum+=f32::from_bits(x[3])*f32::from_bits(w[3]);
                    q+=1;
                }
            }
            c+=1;
        }
        if active {*(p.output as *mut f32).add((row0+lane)*m+token)=sum;}
    }
}
