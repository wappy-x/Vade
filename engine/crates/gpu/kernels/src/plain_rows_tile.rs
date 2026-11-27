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
//! neighbouring blocks share weight lines), 256 threads, dynamic shared memory
//! sized to the op (`28,672` B for BF16 weights -> 3 CTAs/SM, `36,864` B for FP32).
//! Same `Arguments` as `prompt_rows`. Needs cols % 4 == 0 and 16-byte aligned
//! input and weight; anything else keeps `prompt_rows`.
use crate::prompt_ops::{Arguments,Op};

macro_rules! plain_tile_body {
    ($p:expr, $bf16:expr) => {{
        const BR:usize=64;const BT:usize=64;const KC:usize=16;
        const BF16:bool=$bf16;
        // Multiple of 4 words (16 B) so every cp.async.ca 16-byte destination is 16-byte aligned.
        // KC = 16 gives 16,640 B (BF16, 6 CTAs/SM) and 20,480 B (FP32, 4 CTAs/SM), doubling SM occupancy.
        const SK_W:usize=if BF16 {12} else {20};
        const SK_X:usize=20;
        const W_WORDS:usize=BR*SK_W;
        const STAGE:usize=W_WORDS+BT*SK_X;
        const NEED:u32=if (2*STAGE*4)>16640 {(2*STAGE*4) as u32} else {16640};
        let dynamic:u32;
        core::arch::asm!("mov.u32 {d}, %dynamic_smem_size;",d=out(reg32)dynamic,options(nomem,nostack));
        if dynamic<NEED {core::arch::asm!("trap;");}
        let s=$p.shape;
        let (m,rows,k)=(s.tokens as usize,s.rows as usize,s.cols as usize);let words=$p.words as usize;
        let token_tiles=m.div_ceil(BT);let block=core::arch::nvptx::_block_idx_x() as usize;
        let (tile_t,tile_r)=(block%token_tiles,block/token_tiles);
        if tile_r*BR>=rows {return;}
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (ty,tx)=(tid/16,tid%16);
        let smem=0u32;
        let weight_f32=$p.weight as *const f32;let weight_u32=$p.weight as *const u32;let input=$p.input[0] as *const f32;
        let chunks=k.div_ceil(KC);
        let w_items=if BF16 {BR*2} else {BR*4};
        let total_items=w_items+BT*4;
        let issue=|c:usize| {
            let stage=smem+((c&1)*STAGE*4) as u32;let first=c*KC;
            let mut item=tid;
            while item<total_items {
                if item<w_items {
                    if BF16 {
                        let (row,quad)=(item/2,item%2);let column=first+quad*8;
                        if column<k {let r=tile_r*BR+row;if r<rows {crate::ptx::copy16(stage+(row*SK_W*4+quad*16) as u32,weight_u32.add((r*k+column)/2));}}
                    } else {
                        let (row,quad)=(item/4,item%4);let column=first+quad*4;
                        if column<k {let r=tile_r*BR+row;if r<rows {crate::ptx::copy16(stage+(row*SK_W*4+quad*16) as u32,weight_f32.add(r*k+column).cast());}}
                    }
                } else {
                    let xitem=item-w_items;let (row,quad)=(xitem/4,xitem%4);let column=first+quad*4;
                    if column<k {let t=tile_t*BT+row;if t<m {crate::ptx::copy16(stage+((W_WORDS+row*SK_X)*4+quad*16) as u32,input.add(t*k+column).cast());}}
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
                if BF16 {
                    for i in 0..4 {
                        let pk=crate::ptx::load2(stage+(((ty*4+i)*SK_W+q*2)*4) as u32);
                        w[i]=[pk[0]<<16,pk[0]&0xffff_0000,pk[1]<<16,pk[1]&0xffff_0000];
                    }
                } else {
                    for i in 0..4 {w[i]=crate::ptx::load4(stage+(((ty*4+i)*SK_W+q*4)*4) as u32);}
                }
                for j in 0..4 {x[j]=crate::ptx::load4(stage+((W_WORDS+(tx+16*j)*SK_X+q*4)*4) as u32);}
                for e in 0..4 {
                    for i in 0..4 {for j in 0..4 {acc[i][j]+=f32::from_bits(x[j][e])*f32::from_bits(w[i][e]);}}
                }
                q+=1;
            }
            crate::ptx::sync();
            c+=1;
        }
        let do_round=(s.first&1)!=0;
        let token_major=(s.first&2)!=0;
        if token_major {
            // Reuse smem (`64 * 65 * 4 = 16,640 B <= 27,648 B`) to transpose the output tile for coalesced [m, rows] stores.
            for i in 0..4 {
                for j in 0..4 {
                    let v=if do_round {crate::bf16::rounded(acc[i][j])} else {acc[i][j]};
                    crate::ptx::store(smem+(((tx+16*j)*65+(ty*4+i))*4) as u32,v);
                }
            }
            crate::ptx::sync();
            let out=$p.output as *mut f32;
            let mut idx=tid;
            while idx<BR*BT {
                let (tl,rl)=(idx/BR,idx%BR);
                let (token,row)=(tile_t*BT+tl,tile_r*BR+rl);
                if token<m && row<rows {*out.add((token*rows+row)*words)=crate::ptx::load(smem+((tl*65+rl)*4) as u32);}
                idx+=256;
            }
        } else {
            for i in 0..4 {
                let row=tile_r*BR+ty*4+i;
                if row<rows {
                    for j in 0..4 {
                        let token=tile_t*BT+tx+16*j;
                        if token<m {
                            let v=if do_round {crate::bf16::rounded(acc[i][j])} else {acc[i][j]};
                            *($p.output as *mut f32).add((row*m+token)*words)=v;
                        }
                    }
                }
            }
        }
    }};
}

#[no_mangle]
pub extern "ptx-kernel" fn plain_rows_tile(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents,
    // with same-context contiguous owners that live through stream completion.
    // Input and weight are immutable and 16-byte aligned (checked here, else the
    // block traps); cols is a positive multiple of 4 (and 8 for BF16 weights), so
    // every 16-byte copy reads whole columns of one row, and only rows below `rows`
    // and tokens below `tokens` are copied. Dynamic shared memory size is checked
    // via `%dynamic_smem_size` before any shared access. The output overlaps no
    // input and each (row, token) word has exactly one writer. All 256 threads
    // reach every barrier.
    unsafe {
        let s=p.shape;
        let bf16=s.op==Op::ProjectRowsBf16;
        if (s.op!=Op::ProjectRows && !bf16) || s.cols==0 || s.cols%4!=0 || (bf16 && s.cols%8!=0) || (p.input[0]|p.weight)&15!=0 || core::arch::nvptx::_block_dim_x()!=256 {core::arch::asm!("trap;");}
        if bf16 {plain_tile_body!(p, true);} else {plain_tile_body!(p, false);}
    }
}

macro_rules! plain_narrow_body {
    ($p:expr, $bf16:expr) => {{
        const BR:usize=16;const BT:usize=32;const KC:usize=32;
        const BF16:bool=$bf16;
        const SK_W:usize=if BF16 {20} else {36};
        const SK_X:usize=36;
        const W_WORDS:usize=BR*SK_W;
        const STAGE:usize=W_WORDS+BT*SK_X;
        const NEED:u32=(2*STAGE*4) as u32;
        let dynamic:u32;
        core::arch::asm!("mov.u32 {d}, %dynamic_smem_size;",d=out(reg32)dynamic,options(nomem,nostack));
        if dynamic<NEED {core::arch::asm!("trap;");}
        let s=$p.shape;
        let (m,rows,k)=(s.tokens as usize,s.rows as usize,s.cols as usize);let words=$p.words as usize;
        let token_tiles=m.div_ceil(BT);let block=core::arch::nvptx::_block_idx_x() as usize;
        let (tile_t,tile_r)=(block%token_tiles,block/token_tiles);
        if tile_r*BR>=rows {return;}
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (ty,tx)=(tid/16,tid%16);
        let smem=0u32;
        let weight_f32=$p.weight as *const f32;let weight_u32=$p.weight as *const u32;let input=$p.input[0] as *const f32;
        let chunks=k.div_ceil(KC);
        let w_items=if BF16 {BR*4} else {BR*8};
        let total_items=w_items+BT*8;
        let issue=|c:usize| {
            let stage=smem+((c&1)*STAGE*4) as u32;let first=c*KC;
            let mut item=tid;
            while item<total_items {
                if item<w_items {
                    if BF16 {
                        let (row,quad)=(item/4,item%4);let column=first+quad*8;
                        if column<k {let r=tile_r*BR+row;if r<rows {crate::ptx::copy16(stage+(row*SK_W*4+quad*16) as u32,weight_u32.add((r*k+column)/2));}}
                    } else {
                        let (row,quad)=(item/8,item%8);let column=first+quad*4;
                        if column<k {let r=tile_r*BR+row;if r<rows {crate::ptx::copy16(stage+(row*SK_W*4+quad*16) as u32,weight_f32.add(r*k+column).cast());}}
                    }
                } else {
                    let xitem=item-w_items;let (row,quad)=(xitem/8,xitem%8);let column=first+quad*4;
                    if column<k {let t=tile_t*BT+row;if t<m {crate::ptx::copy16(stage+((W_WORDS+row*SK_X)*4+quad*16) as u32,input.add(t*k+column).cast());}}
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
                if BF16 {
                    for i in 0..2 {
                        let pk=crate::ptx::load2(stage+(((ty*2+i)*SK_W+q*2)*4) as u32);
                        w[i]=[pk[0]<<16,pk[0]&0xffff_0000,pk[1]<<16,pk[1]&0xffff_0000];
                    }
                } else {
                    for i in 0..2 {w[i]=crate::ptx::load4(stage+(((ty*2+i)*SK_W+q*4)*4) as u32);}
                }
                for j in 0..2 {x[j]=crate::ptx::load4(stage+((W_WORDS+(tx+16*j)*SK_X+q*4)*4) as u32);}
                for e in 0..4 {
                    for i in 0..2 {for j in 0..2 {acc[i][j]+=f32::from_bits(x[j][e])*f32::from_bits(w[i][e]);}}
                }
                q+=1;
            }
            crate::ptx::sync();
            c+=1;
        }
        let do_round=(s.first&1)!=0;
        let token_major=(s.first&2)!=0;
        if token_major {
            for i in 0..2 {
                for j in 0..2 {
                    let v=if do_round {crate::bf16::rounded(acc[i][j])} else {acc[i][j]};
                    crate::ptx::store(smem+(((tx+16*j)*17+(ty*2+i))*4) as u32,v);
                }
            }
            crate::ptx::sync();
            let out=$p.output as *mut f32;
            let mut idx=tid;
            while idx<BR*BT {
                let (tl,rl)=(idx/BR,idx%BR);
                let (token,row)=(tile_t*BT+tl,tile_r*BR+rl);
                if token<m && row<rows {*out.add((token*rows+row)*words)=crate::ptx::load(smem+((tl*17+rl)*4) as u32);}
                idx+=128;
            }
        } else {
            for i in 0..2 {
                let row=tile_r*BR+ty*2+i;
                if row<rows {
                    for j in 0..2 {
                        let token=tile_t*BT+tx+16*j;
                        if token<m {
                            let v=if do_round {crate::bf16::rounded(acc[i][j])} else {acc[i][j]};
                            *($p.output as *mut f32).add((row*m+token)*words)=v;
                        }
                    }
                }
            }
        }
    }};
}

/// Narrow plain rows: the same arithmetic, order and arguments as `plain_rows_tile`, in 16-row x 32-token blocks of 128 threads.
#[no_mangle]
pub extern "ptx-kernel" fn plain_rows_narrow(p:Arguments) {
    // SAFETY: As plain_rows_tile: the host validates Shape::lengths and passes those exact extents, with same-context contiguous
    // owners that live through stream completion. Input and weight are immutable and 16-byte aligned (checked here, else trap);
    // cols is a positive multiple of 4 (and 8 for BF16 weights). Dynamic shared memory size is checked via `%dynamic_smem_size`.
    // Each (row, token) word has one writer. All 128 threads reach every barrier.
    unsafe {
        let s=p.shape;
        let bf16=s.op==Op::ProjectRowsBf16;
        if (s.op!=Op::ProjectRows && !bf16) || s.cols==0 || s.cols%4!=0 || (bf16 && s.cols%8!=0) || (p.input[0]|p.weight)&15!=0 || core::arch::nvptx::_block_dim_x()!=128 {core::arch::asm!("trap;");}
        if bf16 {plain_narrow_body!(p, true);} else {plain_narrow_body!(p, false);}
    }
}
