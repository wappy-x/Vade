//! The recurrent update. A block owns 32 value columns of one head, one thread each, and keeps
//! the columns and this token's vectors in shared memory, so the two folds read no global memory.
//! Every column keeps the adds and the order of `step_column`, so the bits are equal.
use crate::prompt_ops::{Arguments,Op,round};
/// Largest head size the static shared array holds.
const MAX:usize=128;
/// Threads and value columns of one block: one warp.
const VALUES:usize=32;

#[no_mangle]
pub extern "ptx-kernel" fn step_rows_block(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents, with
    // same-context contiguous owners that live through stream completion. Inputs are immutable.
    // The block index names a head and 32 values, the thread one of them, all checked here, so
    // global reads stay inside the head's rows and each output word has one writer. Shared words
    // stay in the static array: dim is a multiple of 32 and at most MAX, or the block traps, so
    // four-word loads are aligned. A thread writes only its column and its own four words of a
    // token's vectors. The vectors alternate between two stages, and the barrier after each
    // write means no reader is still on the stage being written. All threads run the same tokens.
    unsafe {
        let s=p.shape;let (m,h,d)=(s.tokens as usize,s.heads as usize,s.dim as usize);
        let block=core::arch::nvptx::_block_idx_x() as usize;let local=core::arch::nvptx::_thread_idx_x() as usize;
        if s.op!=Op::StepRows || d==0 || d%VALUES!=0 || d>MAX || core::arch::nvptx::_block_dim_x() as usize!=VALUES {core::arch::asm!("trap;");}
        let parts=d/VALUES;let (head,part)=(block/parts,block%parts);let value=part*VALUES+local;
        if head>=h {core::arch::asm!("trap;");}
        let smem=crate::ptx::shared::<{(VALUES*MAX+6*MAX)*4}>();
        let column=smem+(local*d*4) as u32;let stages=smem+(VALUES*d*4) as u32;
        let stride=4*d+1;let width=h*d;let own=d/VALUES;
        let (input,state,decay)=(p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32);let out=p.output as *mut f32;
        let base=head*d*d+value;
        let mut key=0;while key<d {crate::ptx::store(column+(key*4) as u32,*state.add(base+key*d));key+=1;}
        let mut token=0;
        while token<m {
            let at=token*h+head;let row=input.add(at*stride);
            // This token's three vectors: keys, update and decay. Each thread copies its share.
            let stage=stages+((token&1)*3*d*4) as u32;let (keys,update,decays)=(stage,stage+(d*4) as u32,stage+(2*d*4) as u32);
            let mut e=local*own;let end=e+own;
            while e<end {
                crate::ptx::store(keys+(e*4) as u32,*row.add(e));
                crate::ptx::store(update+(e*4) as u32,*row.add(d+e));
                crate::ptx::store(decays+(e*4) as u32,*decay.add(at*d+e));
                e+=1;
            }
            let (target,beta)=(*row.add(2*d+value),*row.add(4*d));
            crate::ptx::sync();
            let mut prediction=0.0f32;let mut key=0;
            while key<d {
                let at=(key*4) as u32;
                let (c,r,u)=(crate::ptx::load4(column+at),crate::ptx::load4(decays+at),crate::ptx::load4(update+at));
                let mut e=0;
                while e<4 {let v=f32::from_bits(c[e])*f32::from_bits(r[e]);crate::ptx::store(column+at+(e*4) as u32,v);prediction+=v*f32::from_bits(u[e]);e+=1;}
                key+=4;
            }
            let delta=(target-prediction)*beta;
            let mut sum=0.0f32;let mut key=0;
            while key<d {
                let at=(key*4) as u32;
                let (c,u,k)=(crate::ptx::load4(column+at),crate::ptx::load4(update+at),crate::ptx::load4(keys+at));
                let mut e=0;
                while e<4 {let v=f32::from_bits(c[e])+f32::from_bits(u[e])*delta;crate::ptx::store(column+at+(e*4) as u32,v);sum+=v*f32::from_bits(k[e]);e+=1;}
                key+=4;
            }
            *out.add(width*d+head*d+value+token*width)=round(sum);
            token+=1;
        }
        let mut key=0;while key<d {*out.add(base+key*d)=crate::ptx::load(column+(key*4) as u32);key+=1;}
    }
}
