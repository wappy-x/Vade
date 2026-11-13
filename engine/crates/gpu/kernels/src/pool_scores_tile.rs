//! The indexer's pool scores (`Op::PoolScoresRows`) in shared-memory tiles, with the method of `scores_rows_tile`.
//! A block takes 64 pools and 64 tokens; each of its 256 threads owns 4 pools x 4 tokens. For each head in ascending
//! order the block stages the pools' keys and the tokens' queries of that head, 32 columns at a time, and every
//! (pool, token) keeps ONE ascending FP32 chain over the columns from +0, each product rounded before its add: the dot
//! of `prompt_ops::unit`. A thread then adds the head's term to its 16 running scores, heads in ascending order from
//! +0, with the unit's own expression. A pool that is not complete at a token's context scores zero, as there.
//! The generic unit made each word in its own thread and read the token's whole query row again for every pool.
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
    // the pair, in the one block of its tile. Every shared read and write stays in the block's own two-stage static
    // array. A block whose pools are complete for none of its tokens stores its zeros and leaves before the first
    // barrier, all its threads alike (the test reads the block index and the shape only); in every other block all
    // 256 threads reach every barrier, and the loop bounds depend only on the shape.
    unsafe {
        const BR:usize=64;const BT:usize=64;const KC:usize=32;const SK:usize=36;const STAGE:usize=(BR+BT)*SK;
        let s=p.shape;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        if s.op!=Op::PoolScoresRows || s.dim==0 || s.dim%4!=0 || s.heads==0 || s.taps==0 || s.tokens==0 || (p.input[0]|p.input[1])&15!=0 || threads!=256 {core::arch::asm!("trap;");}
        let (m,h,d,t,rows,first)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.first as usize);
        let token_tiles=m.div_ceil(BT);
        let block=core::arch::nvptx::_block_idx_x() as usize;
        let (tile_t,tile_r)=(block%token_tiles,block/token_tiles);
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let (ty,tx)=(tid/16,tid%16);
        let out=p.output as *mut f32;
        // The tile's last token has the most complete pools of its tokens. With none in this pool tile, every word is zero.
        let last=if tile_t*BT+BT<m {tile_t*BT+BT-1} else {m-1};
        if tile_r*BR>=(first+last)/t {
            for i in 0..4 {
                let row=tile_r*BR+ty*4+i;if row>=rows {continue;}
                for j in 0..4 {let token=tile_t*BT+tx+16*j;if token<m {*out.add(token*rows+row)=0.0;}}
            }
            return;
        }
        let smem=crate::ptx::shared::<{2*STAGE*4}>();
        let keys=p.input[1] as *const f32;let weights=p.input[2] as *const f32;
        let chunks=d.div_ceil(KC);
        let mut score=[[0.0f32;4];4];
        let mut head=0;
        while head<h {
            let queries=(p.input[0] as *const f32).add(head*d);
            // Copy chunk `c` of this block's 64 keys and of its 64 queries of this head into stage `c & 1`.
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
            // The head's term, as the unit writes it: score += (weight * (1 / root(heads))) * (dot * (1 / root(dim))).max(0).
            for j in 0..4 {
                let token=tile_t*BT+tx+16*j;if token>=m {continue;}
                let weight=*weights.add(token*h+head);
                for i in 0..4 {score[i][j]+=(weight*(1.0/root(h as f32)))*(acc[i][j]*(1.0/root(d as f32))).max(0.0);}
            }
            head+=1;
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
