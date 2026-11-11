//! Device versions of the wrapped per-token causal operations and the convolution rows of a prompt group. Each writes
//! the same bytes as the generic `prompt_rows` body (`prompt_ops::unit`, `causal_ops::unit`, `latent_ops::unit`): the
//! same expressions in the same order, no fused multiply-add added. The generic body gives one thread to each unit and
//! keeps every working value in memory (Route even keeps its scratch in the output row); here a warp, or a thread
//! with its values in registers, does the work. Chosen by `prompt_admission`; every entry takes the generic
//! `Arguments`, so any grid is correct (the loops stride by the grid) and the geometry only decides speed.
use core::arch::asm;
use crate::prompt_ops::{Arguments,Op};
use crate::causal_ops::{Op as Causal,round,sigmoid,root};
use crate::next_math::exp;

const THREADS:usize=crate::prompt_admission::FAST_THREADS as usize;
macro_rules! lane_square_sum {($x:expr,$d:expr,$lane:expr)=>{{
    let (x,d,lane)=($x,$d,$lane);
    let mut sum=0.0f32;let mut i=lane;while i<d {sum+=*x.add(i)* *x.add(i);i+=32;}
    let mut distance=16u32;
    while distance!=0 {
        let other:f32;asm!("shfl.sync.down.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)other,v=in(reg32)sum,d=in(reg32)distance,options(nomem,nostack));
        if lane+(distance as usize)<32 {sum=sum+other;}
        distance/=2;
    }
    let total:f32;asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)total,v=in(reg32)sum,options(nomem,nostack));
    total
}};}

/// Wrapped `Prepare`: one warp per (token, head).
#[no_mangle]
pub extern "ptx-kernel" fn prompt_prepare(p:Arguments) {
    // SAFETY: The host (prompt_admission::wrapped_kernel) checked Shape::lengths for Op::Wrapped over causal Prepare with
    // causal_ops::fast_admitted: per token, x0 holds 3*width, x1 width and x2 heads values, the weight heads+width values
    // (shared by all tokens), and each (token, head) unit owns the disjoint 4*dim+1 output words at unit*(4*dim+1);
    // owners live and disjoint through completion. Indexes stay inside (token < tokens, head < heads, j < dim). All 32
    // lanes of a warp take the same trip counts, so the shuffles are converged. A block of other than THREADS threads traps.
    unsafe {
        let s=p.shape;let inner=s.inner;
        if s.op!=Op::Wrapped || inner.op!=Causal::Prepare || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (heads,d,tokens)=(inner.heads as usize,inner.dim as usize,s.tokens as usize);let width=heads*d;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        let mut unit=core::arch::nvptx::_block_idx_x() as usize*4+tid/32;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*4;
        let w=p.weight as *const f32;
        while unit<tokens*heads {
            let (token,head)=(unit/heads,unit%heads);
            let x0=(p.input[0] as *const f32).add(token*3*width);let x1=(p.input[1] as *const f32).add(token*width);let x2=(p.input[2] as *const f32).add(token*heads);
            let base=head*d;let q=x0.add(base);let k=x0.add(width+base);
            let qnorm=root(lane_square_sum!(q,d,lane)+1e-6);let knorm=root(lane_square_sum!(k,d,lane)+1e-6);
            let scale=1.0/root(d as f32);let rate=exp(*w.add(head));
            let out=(p.output as *mut f32).add(unit*(4*d+1));
            let mut j=lane;
            while j<d {
                *out.add(j)=(*q.add(j)/qnorm)*scale;*out.add(d+j)=*k.add(j)/knorm;*out.add(2*d+j)=*x0.add(2*width+base+j);
                *out.add(3*d+j)=inner.bound*sigmoid(rate*(*x1.add(base+j)+*w.add(heads+base+j)));j+=32;
            }
            if lane==0 {*out.add(4*d)=round(sigmoid(*x2.add(head)));}
            unit+=stride;
        }
    }
}

/// Wrapped `GatedNorm`: one warp per (token, head).
#[no_mangle]
pub extern "ptx-kernel" fn prompt_gated_norm(p:Arguments) {
    // SAFETY: As prompt_prepare, for Op::Wrapped over causal GatedNorm: per token x0 and x1 hold heads*dim values, the weight
    // dim values (shared), and each (token, head) unit owns the disjoint dim output words at unit*dim.
    unsafe {
        let s=p.shape;let inner=s.inner;
        if s.op!=Op::Wrapped || inner.op!=Causal::GatedNorm || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (heads,d,tokens)=(inner.heads as usize,inner.dim as usize,s.tokens as usize);let width=heads*d;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        let mut unit=core::arch::nvptx::_block_idx_x() as usize*4+tid/32;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*4;
        let w=p.weight as *const f32;
        while unit<tokens*heads {
            let (token,head)=(unit/heads,unit%heads);
            let x0=(p.input[0] as *const f32).add(token*width);let x1=(p.input[1] as *const f32).add(token*width);
            let base=head*d;let row=x0.add(base);
            let scale=1.0/root(lane_square_sum!(row,d,lane)/d as f32+inner.epsilon);
            let out=(p.output as *mut f32).add(unit*d);
            let mut j=lane;
            while j<d {*out.add(j)=round(((*row.add(j)*scale)**w.add(j))*sigmoid(*x1.add(base+j)));j+=32;}
            unit+=stride;
        }
    }
}

/// Registers per lane that hold one token's experts: up to 16 per lane, so at most 512 experts.
const PER_LANE:usize=16;
/// Wrapped `Route` for one group (`heads == 1`, `dim == 1`): one warp per token. With one group every expert is kept, the
/// group score is never read, and the selection is `cols` rounds of "largest corrected score, lowest id on a tie".
#[no_mangle]
pub extern "ptx-kernel" fn prompt_route(p:Arguments) {
    // SAFETY: The host (prompt_admission::route_admitted) checked Shape::lengths for Op::Wrapped over causal Route with one
    // group and one kept group: per token x0 holds `rows` logits, the weight `rows` biases (shared), and the token owns
    // the disjoint 2*cols + 2*rows + 2 output words at token*(2*cols+2*rows+2); `rows` <= 32*PER_LANE and `cols` <= 64.
    // Every lane takes the same trip counts and every shuffle is converged. A register array is only indexed by loop
    // counters that unroll (selects over i), never by a data value.
    unsafe {
        let s=p.shape;let inner=s.inner;
        if s.op!=Op::Wrapped || inner.op!=Causal::Route || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (r,k,tokens)=(inner.rows as usize,inner.cols as usize,s.tokens as usize);
        let words=2*k+2*r+2*inner.heads as usize;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        let mut token=core::arch::nvptx::_block_idx_x() as usize*4+tid/32;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*4;
        let w=p.weight as *const f32;
        while token<tokens {
            let x0=(p.input[0] as *const f32).add(token*r);let out=(p.output as *mut f32).add(token*words);
            let mut score=[0.0f32;PER_LANE];let mut choice=[0.0f32;PER_LANE];
            let mut i=0;while i<PER_LANE {
                let e=lane+32*i;
                if e<r {score[i]=sigmoid(*x0.add(e));choice[i]=score[i]+*w.add(e);}
                i+=1;
            }
            let mut sum=0.0f32;
            let mut held_id=[0.0f32;2];let mut held_score=[0.0f32;2];
            let mut slot=0usize;
            while slot<k {
                // Best corrected score of this lane, lowest expert on a tie (strictly greater replaces).
                let mut best=f32::NEG_INFINITY;let mut at=u32::MAX;
                let mut i=0;while i<PER_LANE {let e=lane+32*i;if e<r && choice[i]>best {best=choice[i];at=e as u32;}i+=1;}
                let mut distance=16u32;
                while distance!=0 {
                    let (ov,oa):(f32,u32);
                    asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)ov,v=in(reg32)best,d=in(reg32)distance,options(nomem,nostack));
                    asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)oa,v=in(reg32)at,d=in(reg32)distance,options(nomem,nostack));
                    if ov>best || (ov==best && oa<at) {best=ov;at=oa;}
                    distance/=2;
                }
                // No corrected score above -inf: the generic code keeps id 0.
                let id=if at==u32::MAX {0usize} else {at as usize};
                // The score of expert `id` lives in lane id%32, register id/32; every lane gets it by shuffle.
                let mut mine=0.0f32;let mut i=0;while i<PER_LANE {if id/32==i {mine=score[i];}i+=1;}
                let chosen:f32;asm!("shfl.sync.idx.b32 {o}, {v}, {l}, 31, -1;",o=out(reg32)chosen,v=in(reg32)mine,l=in(reg32)(id%32) as u32,options(nomem,nostack));
                sum+=chosen;
                if slot%32==lane {if slot<32 {held_id[0]=id as f32;held_score[0]=chosen;} else {held_id[1]=id as f32;held_score[1]=chosen;}}
                let mut i=0;while i<PER_LANE {if lane+32*i==id {choice[i]=f32::NEG_INFINITY;}i+=1;}
                slot+=1;
            }
            let mut part=0;while part<2 {
                let slot=lane+32*part;
                if slot<k {
                    let mut v=held_score[part];
                    if inner.taps!=0 {v/=sum+1e-20;}
                    v*=inner.bound;
                    *out.add(slot)=held_id[part];*out.add(k+slot)=v;
                }
                part+=1;
            }
            let mut j=2*k+lane;while j<words {*out.add(j)=0.0;j+=32;}
            token+=stride;
        }
    }
}

/// Registers one thread holds for the expert order of a fold.
const FOLD_MAX:usize=8;
/// Wrapped `ExpertFold`: one thread per (token, feature) with the token's experts in registers. The fold visits the experts
/// in ascending id order (ties and gaps as the generic selection loop treats them), each term rounded to BF16 twice.
#[no_mangle]
pub extern "ptx-kernel" fn prompt_fold(p:Arguments) {
    // SAFETY: The host (prompt_admission::fold_admitted) checked Shape::lengths for Op::Wrapped over causal ExpertFold: per
    // token x0 and x1 hold `rows` values, x2 `rows*cols` values, x3 `cols` values when taps is 1 (otherwise p.input[3] is
    // never read); the output has one word per (token, feature) at index = token*cols+feature; `rows` <= FOLD_MAX (8, the routed experts of a token). The
    // selected slot is always below `rows` (initially 0, else a slot index of the scan), so x2 reads stay inside the
    // token's rows. Owners live and disjoint through completion. No shared memory, no barrier.
    unsafe {
        let s=p.shape;let inner=s.inner;
        if s.op!=Op::Wrapped || inner.op!=Causal::ExpertFold || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (r,k,tokens)=(inner.rows as usize,inner.cols as usize,s.tokens as usize);
        let total=tokens*k;
        let mut index=core::arch::nvptx::_block_idx_x() as usize*THREADS+core::arch::nvptx::_thread_idx_x() as usize;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*THREADS;
        while index<total {
            let (token,feature)=(index/k,index%k);
            let ids=(p.input[0] as *const f32).add(token*r);let weights=(p.input[1] as *const f32).add(token*r);let rows=(p.input[2] as *const f32).add(token*r*k);
            let mut id=[0.0f32;FOLD_MAX];let mut i=0;while i<FOLD_MAX {if i<r {id[i]=*ids.add(i);}i+=1;}
            // The visiting order, by the generic loop: smallest id above the previous one, first slot on a tie, slot 0 if none.
            let mut order=[0usize;FOLD_MAX];let mut have=false;let mut previous=0.0f32;
            let mut step=0;while step<FOLD_MAX {
                if step<r {
                    let mut selected=0usize;let mut best=f32::INFINITY;
                    let mut slot=0;while slot<FOLD_MAX {if slot<r && (!have || id[slot]>previous) && id[slot]<best {selected=slot;best=id[slot];}slot+=1;}
                    order[step]=selected;previous=best;have=true;
                }
                step+=1;
            }
            // The terms do not depend on the running value: load them all, then run the chain.
            let mut term=[0.0f32;FOLD_MAX];let mut step=0;while step<FOLD_MAX {if step<r {let selected=order[step];term[step]=round(*rows.add(selected*k+feature)**weights.add(selected));}step+=1;}
            let mut value=0.0f32;let mut step=0;while step<FOLD_MAX {if step<r {value=round(value+term[step]);}step+=1;}
            *(p.output as *mut f32).add(index)=if inner.taps!=0 {round(value+*(p.input[3] as *const f32).add(token*k+feature))} else {value};
            index+=stride;
        }
    }
}

/// Tokens one thread of `prompt_conv` walks.
const CONV_RUN:usize=crate::prompt_admission::CONV_RUN;
/// `ConvRows` for taps = TAPS: a thread owns (plane, channel) and a run of consecutive tokens; the TAPS inputs of a token
/// are the TAPS inputs of the previous token shifted by one, so each input word is loaded once.
#[inline(always)]
fn conv_run<const TAPS:usize>(p:&Arguments,plane:usize,ch:usize,first:usize,last:usize) {
    // SAFETY: See prompt_conv. Reads: the weights of (plane, ch) are TAPS words at (plane*width+ch)*TAPS; a position below
    // TAPS-1 reads the old history row (plane, position, ch), any other the plane's row position-(TAPS-1) < tokens.
    unsafe {
        let s=p.shape;let width=s.heads as usize*s.dim as usize;let keep=TAPS-1;let history=keep*width;
        let planes=[p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32];
        let current=planes[plane];let old=(p.input[3] as *const f32).add(plane*history);
        let w=(p.weight as *const f32).add((plane*width+ch)*TAPS);
        let mut weight=[0.0f32;TAPS];let mut tap=0;while tap<TAPS {weight[tap]=*w.add(tap);tap+=1;}
        let at=|position:usize|->f32 {if position<keep {*old.add(position*width+ch)} else {*current.add((position-keep)*width+ch)}};
        let mut window=[0.0f32;TAPS];let mut tap=0;while tap<TAPS {window[tap]=at(first+tap);tap+=1;}
        let mut token=first;
        while token<last {
            let mut sum=0.0f32;let mut tap=0;while tap<TAPS {sum+=window[tap]*weight[tap];tap+=1;}
            *(p.output as *mut f32).add((token*3+plane)*width+ch)=round(sum*sigmoid(sum));
            token+=1;
            if token<last {let mut tap=0;while tap+1<TAPS {window[tap]=window[tap+1];tap+=1;}window[TAPS-1]=at(token+TAPS-1);}
        }
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn prompt_conv(p:Arguments) {
    // SAFETY: The host (prompt_admission::conv_admitted) checked Shape::lengths for Op::ConvRows with 1 <= taps <= 8: x0, x1, x2
    // hold tokens*width values, x3 holds 3*(taps-1)*width, the weight 3*width*taps, the output tokens*3*width. A thread owns the
    // output words (token, plane, ch) of one (plane, ch) and tokens [run*CONV_RUN, ..): disjoint across threads. Every read is
    // at a position below taps-1 (old history row) or at token+tap-(taps-1) < tokens (the plane), inside the extents above.
    unsafe {
        let s=p.shape;
        if s.op!=Op::ConvRows || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (m,width,taps)=(s.tokens as usize,s.heads as usize*s.dim as usize,s.taps as usize);
        let columns=3*width;let runs=m.div_ceil(CONV_RUN);
        let mut index=core::arch::nvptx::_block_idx_x() as usize*THREADS+core::arch::nvptx::_thread_idx_x() as usize;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*THREADS;
        while index<columns*runs {
            let (column,run)=(index%columns,index/columns);
            let (plane,ch)=(column/width,column%width);
            let first=run*CONV_RUN;let last=(first+CONV_RUN).min(m);
            match taps {
                1=>conv_run::<1>(&p,plane,ch,first,last),2=>conv_run::<2>(&p,plane,ch,first,last),3=>conv_run::<3>(&p,plane,ch,first,last),4=>conv_run::<4>(&p,plane,ch,first,last),
                5=>conv_run::<5>(&p,plane,ch,first,last),6=>conv_run::<6>(&p,plane,ch,first,last),7=>conv_run::<7>(&p,plane,ch,first,last),8=>conv_run::<8>(&p,plane,ch,first,last),
                _=>core::arch::nvptx::trap(),
            }
            index+=stride;
        }
    }
}
