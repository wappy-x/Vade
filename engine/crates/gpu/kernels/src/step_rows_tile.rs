//! The recurrent update of a prompt group (head size 128), with each thread's column in REGISTERS and the tokens'
//! vectors staged ahead of the folds.
//! `step_rows_block` keeps a thread's column of 128 words in shared memory and loads and stores every word in both
//! folds, with a loop around each group of four; it also reads a token's keys, update, decay, target and beta from
//! global memory when the recurrence reaches the token. A device warp runs about one instruction every 2 ns, so that
//! kernel's time is its instruction count: about 2,000 a token. Here the column is 128 local values that the compiler
//! keeps in registers, both folds are straight-line code (two four-word shared loads and the arithmetic of four keys
//! a group), and a block stages TILE tokens at a time with asynchronous copies into shared memory, two stages in
//! turn. Every column keeps the products, the adds and the order of `step_column`, so the bits are equal.
//!
//! Launch: entry `step_rows_tile`, the `Arguments` of `prompt_rows`, the grid of `step_rows_block`: heads * 4 blocks
//! of 32 threads. Every output word is stored: each token's output, and the state at the end.
use crate::prompt_ops::{Arguments,Op,round};
use crate::strip_warp::unroll;
/// The head size: a column's 128 words are 128 registers of its thread.
const MAX:usize=128;
/// Threads and value columns of one block: one warp.
const VALUES:usize=32;
/// Tokens of one stage.
const TILE:usize=4;
/// Words a token takes in a stage: keys, update and decay (MAX each), the block's 32 targets, beta, and 3 spare words
/// that keep every token's vectors 16-byte aligned.
const TOKEN:usize=3*MAX+VALUES+4;

#[no_mangle]
pub extern "ptx-kernel" fn step_rows_tile(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents, with same-context contiguous owners
    // that live through stream completion. Inputs are immutable. The block index names a head and 32 values, the
    // thread one of them, all checked here, so global reads stay inside the head's rows of tokens below `tokens`, and
    // each output word has one writer. Shared words stay in the static array: dim is MAX, or the block traps, and a
    // token's stage words start at a multiple of 16 bytes, so four-word loads are aligned. A thread's column lives in
    // its own registers. A thread writes to shared memory only by asynchronous copies: its own share of a tile's
    // vectors, its target, and (thread 0) beta. A tile is read only after every thread has waited for its own copies of it and passed the
    // barrier. The two stages alternate, and the barrier at a tile's end means no thread still reads the stage that
    // the next copies overwrite. All threads run the same tiles and tokens: the loop bounds read the shape only.
    unsafe {
        let s=p.shape;let (m,h,d)=(s.tokens as usize,s.heads as usize,s.dim as usize);
        let block=core::arch::nvptx::_block_idx_x() as usize;let local=core::arch::nvptx::_thread_idx_x() as usize;
        if s.op!=Op::StepRows || d!=MAX || core::arch::nvptx::_block_dim_x() as usize!=VALUES {core::arch::asm!("trap;");}
        let parts=d/VALUES;let (head,part)=(block/parts,block%parts);let value=part*VALUES+local;
        if head>=h {core::arch::asm!("trap;");}
        let stages=crate::ptx::shared::<{2*TILE*TOKEN*4}>();
        let stride=4*d+1;let width=h*d;let own=d/VALUES;
        let (input,state,decay)=(p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32);let out=p.output as *mut f32;
        let base=head*d*d+value;
        // The column: 128 values, every index a constant below, so they are 128 registers.
        let mut col=[0.0f32;MAX];
        unroll!(8,a,{unroll!(4,b,{unroll!(4,e,{let i=(a*4+b)*4+e;col[i]=*state.add(base+i*MAX);});});});
        let tiles=m.div_ceil(TILE);
        // Start the copies of tile `c` into stage `c & 1`: this thread's share of each token's keys, update and decay,
        // its own target, and beta from thread 0.
        let issue=|c:usize| {
            let stage=stages+((c&1)*TILE*TOKEN*4) as u32;
            let mut j=0;
            while j<TILE {
                let token=c*TILE+j;
                if token<m {
                    let at=token*h+head;let row=input.add(at*stride);let slot=stage+(j*TOKEN*4) as u32;
                    let mut e=local*own;let end=e+own;
                    while e<end {
                        crate::ptx::copy4(slot+(e*4) as u32,row.add(e).cast());
                        crate::ptx::copy4(slot+((MAX+e)*4) as u32,row.add(d+e).cast());
                        crate::ptx::copy4(slot+((2*MAX+e)*4) as u32,decay.add(at*d+e).cast());
                        e+=1;
                    }
                    crate::ptx::copy4(slot+((3*MAX+local)*4) as u32,row.add(2*d+value).cast());
                    if local==0 {crate::ptx::copy4(slot+((3*MAX+VALUES)*4) as u32,row.add(4*d).cast());}
                }
                j+=1;
            }
        };
        issue(0);crate::ptx::commit();
        let mut c=0;
        while c<tiles {
            if c+1<tiles {issue(c+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
            crate::ptx::sync();
            let stage=stages+((c&1)*TILE*TOKEN*4) as u32;
            let mut j=0;
            while j<TILE {
                let token=c*TILE+j;
                if token<m {
                    let slot=stage+(j*TOKEN*4) as u32;let (keys,update,decays)=(slot,slot+(MAX*4) as u32,slot+(2*MAX*4) as u32);
                    let (target,beta)=(crate::ptx::load(slot+((3*MAX+local)*4) as u32),crate::ptx::load(slot+((3*MAX+VALUES)*4) as u32));
                    // The two folds of `step_column`, key ascending, as straight-line code: a group of four keys a turn.
                    let mut prediction=0.0f32;
                    unroll!(8,a,{unroll!(4,b,{let at=((a*4+b)*16) as u32;
                        let (r,u)=(crate::ptx::load4(decays+at),crate::ptx::load4(update+at));
                        unroll!(4,e,{let i=(a*4+b)*4+e;let v=col[i]*f32::from_bits(r[e]);col[i]=v;prediction+=v*f32::from_bits(u[e]);});
                    });});
                    let delta=(target-prediction)*beta;
                    let mut sum=0.0f32;
                    unroll!(8,a,{unroll!(4,b,{let at=((a*4+b)*16) as u32;
                        let (u,k)=(crate::ptx::load4(update+at),crate::ptx::load4(keys+at));
                        unroll!(4,e,{let i=(a*4+b)*4+e;let v=col[i]+f32::from_bits(u[e])*delta;col[i]=v;sum+=v*f32::from_bits(k[e]);});
                    });});
                    *out.add(width*d+head*d+value+token*width)=round(sum);
                }
                j+=1;
            }
            crate::ptx::sync();
            c+=1;
        }
        unroll!(8,a,{unroll!(4,b,{unroll!(4,e,{let i=(a*4+b)*4+e;*out.add(base+i*MAX)=col[i];});});});
    }
}
