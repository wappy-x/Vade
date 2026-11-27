//! The indexer's pool scores (`Op::PoolScoresRows`) in shared-memory tiles, with the method of `scores_rows_tile`.
//! A block takes 64 pools and 64 tokens; each of its 256 threads owns 4 pools x 4 tokens. For `dim == 128`, the 64
//! pools' keys (`64 * 132 * 4 = 33,792 B`) are staged ONCE into shared memory and reused across all heads (32x less
//! key global memory traffic), while only the queries stream through the remaining 9,216 B stage (`43,008 B <= 48 KB`).
//! Any other `dim` keeps the general two-stage key+query path. Every (pool, token) keeps ONE ascending FP32 chain over
//! the columns from +0, each product rounded before its add, and adds head terms in ascending head order.
//!
//! Launch: entry `pool_scores_tile`, the `Arguments` of `prompt_rows`, grid ceil(tokens / 64) * ceil(pools / 64) blocks
//! (token tiles fastest), 256 threads. Needs dim a multiple of 4 and queries and keys 16-byte aligned; else the block
//! traps and the host keeps `prompt_rows` (`pool_scores_tile::admitted`). Every output word is stored.
use crate::prompt_ops::{Arguments,Op,root};

#[no_mangle]
pub extern "ptx-kernel" fn pool_scores_tile(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents, with same-context contiguous owners
    // that live through stream completion: queries of tokens * heads * dim words, keys of pools * dim words, head
    // weights of tokens * heads words, and a private output of tokens * pools words that overlaps no input. Queries
    // and keys are immutable and 16-byte aligned (checked here, else the block traps); dim is a multiple of 4, so
    // every 16-byte copy reads four whole columns inside its row, and only pools below `rows` and tokens below
    // `tokens` are copied, read as weights or written. Each (token, pool) word has one writer: the thread that owns
    // the pair, in the one block of its tile. Every shared read and write stays in the block's static array
    // (`max(2*(64+64)*36, 64*132 + 64*36) * 4 = 43,008 B <= 48 KB`). A block whose pools are complete for none of its
    // tokens stores its zeros and leaves before the first barrier, all its threads alike; in every other block all
    // 256 threads reach every barrier, and the loop bounds depend only on the shape.
    unsafe {
        const BR:usize=64;const BT:usize=64;const KC:usize=32;const SK:usize=36;
        const STAGE:usize=(BR+BT)*SK;
        const K128_STRIDE:usize=132;
        const K128_WORDS:usize=BR*K128_STRIDE;
        const Q128_KC:usize=16;const Q128_SK:usize=20;
        const Q128_STAGE_WORDS:usize=BT*Q128_SK;
        const Q128_WORDS:usize=2*Q128_STAGE_WORDS;
        const SMEM_WORDS:usize=if 2*STAGE>K128_WORDS+Q128_WORDS {2*STAGE} else {K128_WORDS+Q128_WORDS};
        let s=p.shape;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if s.op!=Op::PoolScoresRows || s.dim==0 || s.dim%4!=0 || s.heads==0 || s.taps==0 || s.tokens==0 || (p.input[0]|p.input[1])&15!=0 || threads!=256 {core::arch::asm!("trap;");}
        let (m,h,d,t,rows,first)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.first as usize);
        let token_tiles=m.div_ceil(BT);
        let block=core::arch::nvptx::_block_idx_x() as usize;
        let (tile_t,tile_r)=(block%token_tiles,block/token_tiles);
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (ty,tx)=(tid/16,tid%16);
        let out=p.output as *mut f32;
        let last=if tile_t*BT+BT<m {tile_t*BT+BT-1} else {m-1};
        if tile_r*BR>=(first+last)/t {
            for i in 0..4 {
                let row=tile_r*BR+ty*4+i;if row>=rows {continue;}
                for j in 0..4 {let token=tile_t*BT+tx+16*j;if token<m {*out.add(token*rows+row)=0.0;}}
            }
            return;
        }
        let smem=crate::ptx::shared::<{SMEM_WORDS*4}>();
        let keys=p.input[1] as *const f32;let weights=p.input[2] as *const f32;
        let inv_root_h=1.0/root(h as f32);
        let inv_root_d=1.0/root(d as f32);
        let mut score=[[0.0f32;4];4];
        if d==128 {
            // Stage all 64 pools' 128-dim keys ONCE into smem[0..K128_WORDS], reused across all heads.
            let mut item=tid;
            while item<BR*32 {
                let (row,quad)=(item/32,item%32);
                let r=tile_r*BR+row;
                if r<rows {crate::ptx::copy16(smem+((row*K128_STRIDE+quad*4)*4) as u32,keys.add(r*128+quad*4).cast());}
                item+=256;
            }
            crate::ptx::commit();
            crate::ptx::wait();
            crate::ptx::sync();
            let q_smem=smem+(K128_WORDS*4) as u32;
            let q_base=p.input[0] as *const f32;
            let issue_q=|step:usize| {
                let hd=step/8;let ch=step&7;let start=ch*Q128_KC;
                let dst_base=q_smem+((step&1)*Q128_STAGE_WORDS*4) as u32;
                let (row,quad)=(tid/4,tid%4);
                let token=tile_t*BT+row;
                if token<m {crate::ptx::copy16(dst_base+((row*Q128_SK+quad*4)*4) as u32,q_base.add((token*h+hd)*128+start+quad*4).cast());}
            };
            let total_steps=h*8;
            issue_q(0);crate::ptx::commit();
            let mut head=0;
            while head<h {
                let mut acc=[[0.0f32;4];4];
                let mut c=0usize;
                while c<8 {
                    let step=head*8+c;
                    if step+1<total_steps {issue_q(step+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
                    crate::ptx::sync();
                    let cur_q=q_smem+((step&1)*Q128_STAGE_WORDS*4) as u32;
                    let start=c*Q128_KC;
                    let mut q=0usize;
                    while q<4 {
                        let mut w=[[0u32;4];4];let mut x=[[0u32;4];4];
                        for i in 0..4 {w[i]=crate::ptx::load4(smem+(((ty*4+i)*K128_STRIDE+start+q*4)*4) as u32);}
                        for j in 0..4 {x[j]=crate::ptx::load4(cur_q+(((tx+16*j)*Q128_SK+q*4)*4) as u32);}
                        for e in 0..4 {
                            for i in 0..4 {for j in 0..4 {acc[i][j]+=f32::from_bits(x[j][e])*f32::from_bits(w[i][e]);}}
                        }
                        q+=1;
                    }
                    crate::ptx::sync();
                    c+=1;
                }
                for j in 0..4 {
                    let token=tile_t*BT+tx+16*j;if token>=m {continue;}
                    let w_scaled=(*weights.add(token*h+head))*inv_root_h;
                    for i in 0..4 {score[i][j]+=w_scaled*(acc[i][j]*inv_root_d).max(0.0);}
                }
                head+=1;
            }
        } else {
            let chunks=d.div_ceil(KC);
            let mut head=0;
            while head<h {
                let queries=(p.input[0] as *const f32).add(head*d);
                let issue=|c:usize| {
                    let stage=smem+((c&1)*STAGE*4) as u32;let start=c*KC;
                    let mut item=tid;
                    while item<(BR+BT)*8 {
                        let (which,row,quad)=(item/(BR*8),item%(BR*8)/8,item%8);let column=start+quad*4;
                        if column<d {
                            let dst=stage+((which*BR+row)*SK*4+quad*16) as u32;
                            if which==0 {
                                let r=tile_r*BR+row;if r<rows {crate::ptx::copy16(dst,keys.add(r*d+column).cast());}
                            } else {
                                let token=tile_t*BT+row;if token<m {crate::ptx::copy16(dst,queries.add(token*h*d+column).cast());}
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
                    let stage=smem+((c&1)*STAGE*4) as u32;let valid=(d-c*KC).min(KC);
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
                for j in 0..4 {
                    let token=tile_t*BT+tx+16*j;if token>=m {continue;}
                    let w_scaled=(*weights.add(token*h+head))*inv_root_h;
                    for i in 0..4 {score[i][j]+=w_scaled*(acc[i][j]*inv_root_d).max(0.0);}
                }
                head+=1;
            }
        }
        for i in 0..4 {
            let row=tile_r*BR+ty*4+i;if row>=rows {continue;}
            for j in 0..4 {
                let token=tile_t*BT+tx+16*j;if token>=m {continue;}
                *out.add(token*rows+row)=if row<(first+token)/t {score[i][j]} else {0.0};
            }
        }
    }
}
