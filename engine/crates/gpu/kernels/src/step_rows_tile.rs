//! The recurrent update of a prompt group (head size 128), keeping keys `0..64` in registers and keys `64..128` in
//! bank-conflict-free shared memory (`21,632 B` total static smem -> 4 warps/SM, `~104` regs, `0` spills) while staging
//! tokens' vectors ahead of the folds with asynchronous copies.
//! Every column keeps the products, the adds and the exact ascending key order of `step_column`, so the bits are equal.
//!
//! Launch: entry `step_rows_tile`, the `Arguments` of `prompt_rows`, the grid of `step_rows_block`: heads * 4 blocks
//! of 32 threads. Every output word is stored: each token's output, and the state at the end.
use crate::prompt_ops::{Arguments,Op,PREPARE_DECAY_EXP,round};
use crate::strip_warp::unroll;
const MAX:usize=128;
/// Lower 64 keys stay in thread registers; upper 64 keys sit in bank-conflict-free shared `[key][lane]` storage.
const REG_KEYS:usize=64;
const SMEM_KEYS:usize=MAX-REG_KEYS;
/// Threads and value columns of one block: one warp.
const VALUES:usize=32;
/// Tokens of one stage.
const TILE:usize=4;
/// Words a token takes in a stage: keys, update and decay (MAX each), the block's 32 targets, beta, and 3 spare words
/// that keep every token's vectors 16-byte aligned.
const TOKEN:usize=3*MAX+VALUES+4;
const STAGE_WORDS:usize=2*TILE*TOKEN;
const COL_HI_WORDS:usize=SMEM_KEYS*VALUES;
const TOTAL_SMEM_BYTES:usize=(STAGE_WORDS+COL_HI_WORDS)*4;

#[no_mangle]
pub extern "ptx-kernel" fn step_rows_tile(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents, with same-context contiguous owners
    // that live through stream completion. Inputs are immutable. The block index names a head and 32 values, the
    // thread one of them, all checked here, so global reads stay inside the head's rows of tokens below `tokens`, and
    // each output word has one writer. Shared words stay in the static array: dim is MAX, or the block traps, and a
    // token's stage words start at a multiple of 16 bytes, so four-word loads are aligned. When `s.first == PREPARE_DECAY_EXP`
    // (`2`), pre-exponentiated decays are read from `row + 3*d` instead of `p.input[2]` (which is unused).
    // A tile is read only after every thread has waited for its own copies of it and passed the barrier. The two stages
    // alternate, and the barrier at a tile's end means no thread still reads the stage that the next copies overwrite.
    // All threads run the same tiles and tokens: the loop bounds read the shape only.
    unsafe {
        let s=p.shape;let (m,h,d)=(s.tokens as usize,s.heads as usize,s.dim as usize);
        let block=core::arch::nvptx::_block_idx_x() as usize;let local=core::arch::nvptx::_thread_idx_x() as usize;
        if s.op!=Op::StepRows || d!=MAX || (s.first!=0 && s.first!=PREPARE_DECAY_EXP) || core::arch::nvptx::_block_dim_x() as usize!=VALUES {core::arch::asm!("trap;");}
        let parts=d/VALUES;let (head,part)=(block/parts,block%parts);let value=part*VALUES+local;
        if head>=h {core::arch::asm!("trap;");}
        let inline_decay=s.first==PREPARE_DECAY_EXP;
        let stages=crate::ptx::shared::<TOTAL_SMEM_BYTES>();
        let col_hi=stages+((STAGE_WORDS+local)*4) as u32;
        let stride=4*d+1;let width=h*d;let own=d/VALUES;
        let (input,state,decay)=(p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32);let out=p.output as *mut f32;
        let base=head*d*d+value;
        // Lower 64 keys in registers; upper 64 keys in 32-way bank-conflict-free shared memory (`[k_hi][local]`).
        let mut col=[0.0f32;REG_KEYS];
        unroll!(4,a,{unroll!(4,b,{unroll!(4,e,{let i=(a*4+b)*4+e;col[i]=*state.add(base+i*MAX);});});});
        let mut ki=0usize;
        while ki<SMEM_KEYS {
            crate::ptx::store(col_hi+((ki*VALUES)*4) as u32,*state.add(base+(REG_KEYS+ki)*MAX));
            ki+=1;
        }
        let tiles=m.div_ceil(TILE);
        let issue=|c:usize| {
            let stage=stages+((c&1)*TILE*TOKEN*4) as u32;
            let mut j=0;
            while j<TILE {
                let token=c*TILE+j;
                if token<m {
                    let at=token*h+head;let row=input.add(at*stride);let slot=stage+(j*TOKEN*4) as u32;
                    let mut q=0;
                    while q<own {
                        let e=local+q*VALUES;
                        crate::ptx::copy4(slot+(e*4) as u32,row.add(e).cast());
                        crate::ptx::copy4(slot+((MAX+e)*4) as u32,row.add(d+e).cast());
                        let dptr=if inline_decay {row.add(3*d+e)} else {decay.add(at*d+e)};
                        crate::ptx::copy4(slot+((2*MAX+e)*4) as u32,dptr.cast());
                        q+=1;
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
                    let slot=stage+(j*TOKEN*4) as u32;let (mut keys,mut update,mut decays)=(slot,slot+(MAX*4) as u32,slot+(2*MAX*4) as u32);
                    let (target,beta)=(crate::ptx::load(slot+((3*MAX+local)*4) as u32),crate::ptx::load(slot+((3*MAX+VALUES)*4) as u32));
                    let mut prediction=0.0f32;
                    unroll!(4,a,{unroll!(4,b,{let at=((a*4+b)*16) as u32;
                        let (r,u)=(crate::ptx::load4(decays+at),crate::ptx::load4(update+at));
                        let base_i=(a*4+b)*4;
                        let v0=col[base_i]*f32::from_bits(r[0]);let v1=col[base_i+1]*f32::from_bits(r[1]);
                        let v2=col[base_i+2]*f32::from_bits(r[2]);let v3=col[base_i+3]*f32::from_bits(r[3]);
                        col[base_i]=v0;col[base_i+1]=v1;col[base_i+2]=v2;col[base_i+3]=v3;
                        prediction+=v0*f32::from_bits(u[0]);prediction+=v1*f32::from_bits(u[1]);
                        prediction+=v2*f32::from_bits(u[2]);prediction+=v3*f32::from_bits(u[3]);
                    });
                        core::arch::asm!("/* {0} {1} {2} */",inout(reg32)prediction,inout(reg32)decays,inout(reg32)update,options(nomem,nostack));
                    });
                    let mut g=0u32;
                    while g<16 {
                        core::arch::asm!("/* {0} */",inout(reg32)g,options(nomem,nostack));
                        let at=(REG_KEYS*4) as u32+g*16;
                        let (r,u)=(crate::ptx::load4(decays+at),crate::ptx::load4(update+at));
                        let cp=col_hi+g*(4*VALUES*4) as u32;
                        let v0=crate::ptx::load(cp)*f32::from_bits(r[0]);
                        let v1=crate::ptx::load(cp+(VALUES*4) as u32)*f32::from_bits(r[1]);
                        let v2=crate::ptx::load(cp+(2*VALUES*4) as u32)*f32::from_bits(r[2]);
                        let v3=crate::ptx::load(cp+(3*VALUES*4) as u32)*f32::from_bits(r[3]);
                        crate::ptx::store(cp,v0);crate::ptx::store(cp+(VALUES*4) as u32,v1);
                        crate::ptx::store(cp+(2*VALUES*4) as u32,v2);crate::ptx::store(cp+(3*VALUES*4) as u32,v3);
                        prediction+=v0*f32::from_bits(u[0]);prediction+=v1*f32::from_bits(u[1]);
                        prediction+=v2*f32::from_bits(u[2]);prediction+=v3*f32::from_bits(u[3]);
                        g+=1;
                    }
                    let delta=(target-prediction)*beta;
                    let mut sum=0.0f32;
                    unroll!(4,a,{unroll!(4,b,{let at=((a*4+b)*16) as u32;
                        let (u,k)=(crate::ptx::load4(update+at),crate::ptx::load4(keys+at));
                        let base_i=(a*4+b)*4;
                        let v0=col[base_i]+f32::from_bits(u[0])*delta;let v1=col[base_i+1]+f32::from_bits(u[1])*delta;
                        let v2=col[base_i+2]+f32::from_bits(u[2])*delta;let v3=col[base_i+3]+f32::from_bits(u[3])*delta;
                        col[base_i]=v0;col[base_i+1]=v1;col[base_i+2]=v2;col[base_i+3]=v3;
                        sum+=v0*f32::from_bits(k[0]);sum+=v1*f32::from_bits(k[1]);
                        sum+=v2*f32::from_bits(k[2]);sum+=v3*f32::from_bits(k[3]);
                    });
                        core::arch::asm!("/* {0} {1} {2} */",inout(reg32)sum,inout(reg32)update,inout(reg32)keys,options(nomem,nostack));
                    });
                    let mut g=0u32;
                    while g<16 {
                        core::arch::asm!("/* {0} */",inout(reg32)g,options(nomem,nostack));
                        let at=(REG_KEYS*4) as u32+g*16;
                        let (u,k)=(crate::ptx::load4(update+at),crate::ptx::load4(keys+at));
                        let cp=col_hi+g*(4*VALUES*4) as u32;
                        let v0=crate::ptx::load(cp)+f32::from_bits(u[0])*delta;
                        let v1=crate::ptx::load(cp+(VALUES*4) as u32)+f32::from_bits(u[1])*delta;
                        let v2=crate::ptx::load(cp+(2*VALUES*4) as u32)+f32::from_bits(u[2])*delta;
                        let v3=crate::ptx::load(cp+(3*VALUES*4) as u32)+f32::from_bits(u[3])*delta;
                        crate::ptx::store(cp,v0);crate::ptx::store(cp+(VALUES*4) as u32,v1);
                        crate::ptx::store(cp+(2*VALUES*4) as u32,v2);crate::ptx::store(cp+(3*VALUES*4) as u32,v3);
                        sum+=v0*f32::from_bits(k[0]);sum+=v1*f32::from_bits(k[1]);
                        sum+=v2*f32::from_bits(k[2]);sum+=v3*f32::from_bits(k[3]);
                        g+=1;
                    }
                    *out.add(width*d+head*d+value+token*width)=round(sum);
                }
                j+=1;
            }
            crate::ptx::sync();
            c+=1;
        }
        unroll!(4,a,{unroll!(4,b,{unroll!(4,e,{let i=(a*4+b)*4+e;*out.add(base+i*MAX)=col[i];});});});
        let mut ki=0usize;
        while ki<SMEM_KEYS {
            *out.add(base+(REG_KEYS+ki)*MAX)=crate::ptx::load(col_hi+((ki*VALUES)*4) as u32);
            ki+=1;
        }
    }
}
