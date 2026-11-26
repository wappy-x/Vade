//! Plain token rows in shared-memory tiles. Same arithmetic and order as the
//! `prompt_rows` ProjectRows fast path: every output `[row][token]` is ONE
//! ascending FP32 chain over the columns, from 0.0, each product rounded before
//! its add, no fused multiply-add. A block takes 64 weight rows and 64 tokens;
//! each of its 256 threads owns 4 rows x 4 tokens (16 independent chains), so
//! each shared-memory load feeds several multiplies. Columns pass through shared
//! memory in chunks of 32 in ascending order. Edge tiles skip invalid rows and
//! tokens; they never pad a chain with extra terms.
//!
//! Launch: grid = ceil(tokens/64) * ceil(rows/64) blocks (token tile fastest, so
//! neighbouring blocks share weight lines), 256 threads, no dynamic shared
//! memory. Same `Arguments` as `prompt_rows`. Needs cols % 4 == 0 and 16-byte
//! aligned input and weight; anything else keeps `prompt_rows`.
use crate::prompt_ops::{Arguments,Op};

#[no_mangle]
pub extern "ptx-kernel" fn plain_rows_tile(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents,
    // with same-context contiguous owners that live through stream completion.
    // Input and weight are immutable and 16-byte aligned (checked here, else the
    // block traps); cols is a positive multiple of 4, so every 16-byte copy
    // reads four whole columns of one row, and only rows below `rows` and tokens
    // below `tokens` are copied. The output overlaps no input and each (row,
    // token) word has exactly one writer. Every shared read and write stays in
    // the block's own two-stage static array. All 256 threads reach every
    // barrier; the loop bounds depend only on the shape.
    unsafe {
        const BR:usize=64;const BT:usize=64;const KC:usize=32;const SK:usize=36;const STAGE:usize=(BR+BT)*SK;
        let s=p.shape;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if s.op!=Op::ProjectRows || s.cols==0 || s.cols%4!=0 || (p.input[0]|p.weight)&15!=0 || threads!=256 {core::arch::asm!("trap;");}
        let (m,rows,k)=(s.tokens as usize,s.rows as usize,s.cols as usize);let words=p.words as usize;
        let token_tiles=m.div_ceil(BT);let block=core::arch::nvptx::_block_idx_x() as usize;
        let (tile_t,tile_r)=(block%token_tiles,block/token_tiles);
        if tile_r*BR>=rows {return;}
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (ty,tx)=(tid/16,tid%16);
        let smem=crate::ptx::shared::<{2*STAGE*4}>();
        let weight=p.weight as *const f32;let input=p.input[0] as *const f32;
        let chunks=k.div_ceil(KC);
        // Copy chunk `c` of this block's 64 weight rows and 64 token rows into stage `c & 1`.
        let issue=|c:usize| {
            let stage=smem+((c&1)*STAGE*4) as u32;let first=c*KC;
            let mut item=tid;
            while item<(BR+BT)*8 {
                let (which,row,quad)=(item/(BR*8),item%(BR*8)/8,item%8);let column=first+quad*4;
                if column<k {
                    let dst=stage+((which*BR+row)*SK*4+quad*16) as u32;
                    if which==0 {
                        let r=tile_r*BR+row;if r<rows {crate::ptx::copy16(dst,weight.add(r*k+column).cast());}
                    } else {
                        let t=tile_t*BT+row;if t<m {crate::ptx::copy16(dst,input.add(t*k+column).cast());}
                    }
                }
                item+=256;
            }
        };
        let mut acc=[[0.0f32;4];4];
        issue(0);crate::ptx::commit();
        let mut c=0;
        while c<chunks {
            if c+1<chunks {issue(c+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
            crate::ptx::sync();
            let stage=smem+((c&1)*STAGE*4) as u32;let valid=(k-c*KC).min(KC);
            let mut q=0;
            while q<valid/4 {
                let mut w=[[0u32;4];4];let mut x=[[0u32;4];4];
                for i in 0..4 {w[i]=crate::ptx::load4(stage+(((ty*4+i)*SK+q*4)*4) as u32);}
                for j in 0..4 {x[j]=crate::ptx::load4(stage+(((BR+tx+16*j)*SK+q*4)*4) as u32);}
                for e in 0..4 {
                    for i in 0..4 {for j in 0..4 {acc[i][j]+=f32::from_bits(x[j][e])*f32::from_bits(w[i][e]);}}
                }
                q+=1;
            }
            crate::ptx::sync();
            c+=1;
        }
        for i in 0..4 {
            let row=tile_r*BR+ty*4+i;
            if row<rows {
                for j in 0..4 {
                    let token=tile_t*BT+tx+16*j;
                    if token<m {*(p.output as *mut f32).add((row*m+token)*words)=acc[i][j];}
                }
            }
        }
    }
}

/// Narrow plain rows: the same arithmetic, order and arguments as `plain_rows_tile`, in 16-row x 32-token blocks of 128 threads
/// (each thread 2 rows x 2 tokens, 4 independent chains). A projection of up to 192 rows has only 1 to 3 blocks of 64 rows per token
/// tile, so the 64 x 64 tile leaves most of the 48 SMs idle (a 4-block launch runs 217 us); here the same call has 8 to 24 times
/// as many blocks. Every output word is still ONE ascending FP32 chain over the columns from 0.0, each product rounded before
/// its add, no fused multiply-add; only the block shape changes.
///
/// Launch: grid = ceil(tokens/32) * ceil(rows/16) blocks (token tile fastest), 128 threads, no dynamic shared memory.
#[no_mangle]
pub extern "ptx-kernel" fn plain_rows_narrow(p:Arguments) {
    // SAFETY: As plain_rows_tile: the host validates Shape::lengths and passes those exact extents, with same-context contiguous
    // owners that live through stream completion. Input and weight are immutable and 16-byte aligned (checked here, else the block
    // traps); cols is a positive multiple of 4, so every 16-byte copy reads four whole columns of one row, and only rows below
    // `rows` and tokens below `tokens` are copied. The output overlaps no input and each (row, token) word has exactly one writer.
    // Every shared read and write stays in the block's own two-stage static array. All 128 threads reach every barrier; the loop
    // bounds depend only on the shape.
    unsafe {
        const BR:usize=16;const BT:usize=32;const KC:usize=32;const SK:usize=36;const STAGE:usize=(BR+BT)*SK;
        let s=p.shape;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if s.op!=Op::ProjectRows || s.cols==0 || s.cols%4!=0 || (p.input[0]|p.weight)&15!=0 || threads!=128 {core::arch::asm!("trap;");}
        let (m,rows,k)=(s.tokens as usize,s.rows as usize,s.cols as usize);let words=p.words as usize;
        let token_tiles=m.div_ceil(BT);let block=core::arch::nvptx::_block_idx_x() as usize;
        let (tile_t,tile_r)=(block%token_tiles,block/token_tiles);
        if tile_r*BR>=rows {return;}
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (ty,tx)=(tid/16,tid%16);
        let smem=crate::ptx::shared::<{2*STAGE*4}>();
        let weight=p.weight as *const f32;let input=p.input[0] as *const f32;
        let chunks=k.div_ceil(KC);
        // Copy chunk `c` of this block's 16 weight rows and 32 token rows into stage `c & 1`.
        let issue=|c:usize| {
            let stage=smem+((c&1)*STAGE*4) as u32;let first=c*KC;
            let mut item=tid;
            while item<(BR+BT)*8 {
                let (which,row,quad)=if item<BR*8 {(0,item/8,item%8)} else {(1,(item-BR*8)/8,item%8)};let column=first+quad*4;
                if column<k {
                    let dst=stage+((which*BR+row)*SK*4+quad*16) as u32;
                    if which==0 {
                        let r=tile_r*BR+row;if r<rows {crate::ptx::copy16(dst,weight.add(r*k+column).cast());}
                    } else {
                        let t=tile_t*BT+row;if t<m {crate::ptx::copy16(dst,input.add(t*k+column).cast());}
                    }
                }
                item+=128;
            }
        };
        let mut acc=[[0.0f32;2];2];
        issue(0);crate::ptx::commit();
        let mut c=0;
        while c<chunks {
            if c+1<chunks {issue(c+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
            crate::ptx::sync();
            let stage=smem+((c&1)*STAGE*4) as u32;let valid=(k-c*KC).min(KC);
            let mut q=0;
            while q<valid/4 {
                let mut w=[[0u32;4];2];let mut x=[[0u32;4];2];
                for i in 0..2 {w[i]=crate::ptx::load4(stage+(((ty*2+i)*SK+q*4)*4) as u32);}
                for j in 0..2 {x[j]=crate::ptx::load4(stage+(((BR+tx+16*j)*SK+q*4)*4) as u32);}
                for e in 0..4 {
                    for i in 0..2 {for j in 0..2 {acc[i][j]+=f32::from_bits(x[j][e])*f32::from_bits(w[i][e]);}}
                }
                q+=1;
            }
            crate::ptx::sync();
            c+=1;
        }
        for i in 0..2 {
            let row=tile_r*BR+ty*2+i;
            if row<rows {
                for j in 0..2 {
                    let token=tile_t*BT+tx+16*j;
                    if token<m {*(p.output as *mut f32).add((row*m+token)*words)=acc[i][j];}
                }
            }
        }
    }
}
