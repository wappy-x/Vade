//! Device versions of five single-token causal operations. Each writes the same bytes as the generic
//! one-thread-per-unit code (`causal_ops::unit`, `latent_ops::unit`): the same expressions in the same order,
//! no fused multiply-add added. What changes is where the working values live and which lanes hold them.
//! A warp (or a block) works one unit at a time and loops over its units, so any grid is correct; the launch
//! geometry (`causal_ops::geometry`) only decides speed. Chosen by `causal_ops::fast_admitted`; the caller
//! guarantees the block has 128 threads and every lane of every warp enters (no early exit before a barrier).
use core::arch::asm;
use crate::causal_ops::{Arguments,round,sigmoid,root};
use crate::next_math::exp;
use crate::strip_warp::unroll;

/// Lane tree of `square_sum`: 32 lanes each add their elements lane, lane+32, ... in ascending order from 0.0,
/// then distances 16, 8, 4, 2, 1 with a lane only adding a partner that exists. Every lane returns lane 0's total.
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

/// Picks one thread of `expert_fold` holds in registers.
const FOLD_MAX:usize=crate::causal_ops::FOLD_PICKS;
/// Op::ExpertFold with `dim` FOLD_ROUNDS, at most FOLD_MAX picks: the round launch of the row words followed by the fold,
/// in one launch. One thread a word, as the generic unit. A thread loads the picks' ids once and finds the visiting order
/// in registers; the generic unit reads the ids from memory in each of its picks x picks comparisons. Same order
/// (ascending id; the first slot on a tie; slot 0 when no id is left, as the generic scan gives), same chain:
/// `round(value+round(word*weight))` a pick from +0.0, then the shared word. Each row word is rounded first with
/// `bf16::rounded`, the cast of the round launch (`bf16_lanes_round`), which keeps a NaN a NaN. The fold alone (any other
/// `dim`) has no device version: it traps here and keeps the generic unit.
pub(crate) fn expert_fold(p:Arguments) {
    // SAFETY: The host checked Shape::lengths (`dim` changes no extent) and the exact same-context extents: x0 and x1 hold
    // `rows` values, x2 `rows*cols` values, x3 `cols` values when taps is 1 (otherwise p.input[3] is never read), and the
    // private output `cols` words. `causal_ops::fast_admitted` bounds rows by FOLD_MAX and asks for the FOLD_ROUNDS mark;
    // another op, pick count or `dim` traps here. A selected slot is always below `rows` (0 at first, else a slot index of the scan), so
    // every x1 and x2 read stays inside its extent. Each word index below `cols` belongs to one thread, and a thread writes
    // only its own words. Owners live and disjoint through completion. No shared memory, no barrier, no shuffle.
    unsafe {
        let s=p.shape;let (r,k)=(s.rows as usize,s.cols as usize);
        if s.op!=crate::causal_ops::Op::ExpertFold || r==0 || r>FOLD_MAX || s.dim!=crate::causal_ops::FOLD_ROUNDS {core::arch::nvptx::trap();}
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        let mut index=core::arch::nvptx::_block_idx_x() as usize*threads+core::arch::nvptx::_thread_idx_x() as usize;
        if index>=k {return;}
        let stride=core::arch::nvptx::_grid_dim_x() as usize*threads;
        let (ids,weights,rows)=(p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32);
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
        while index<k {
            // The terms do not depend on the running value: load them all, then run the chain.
            let mut term=[0.0f32;FOLD_MAX];let mut step=0;
            while step<FOLD_MAX {
                if step<r {let selected=order[step];let word=*rows.add(selected*k+index);
                    term[step]=round(crate::bf16::rounded(word)**weights.add(selected));}
                step+=1;
            }
            let mut value=0.0f32;let mut step=0;while step<FOLD_MAX {if step<r {value=round(value+term[step]);}step+=1;}
            *(p.output as *mut f32).add(index)=if s.taps!=0 {round(value+*(p.input[3] as *const f32).add(index))} else {value};
            index+=stride;
        }
    }
}

/// Op::Prepare, one warp per head.
pub(crate) fn prepare(p:Arguments) {
    prepare_heads::<false>(p)
}
pub(crate) fn prepare_one(p:Arguments) {prepare_heads::<true>(p)}
#[inline(always)]
fn prepare_heads<const ONE:bool>(p:Arguments) {
    // SAFETY: The host checked Shape::lengths and the exact same-context extents: x0 holds 3*width, x1 width and
    // x2 heads values, the weight heads+width values, the output heads*(4*dim+1) values, each head's slice
    // disjoint. Every read and write below indexes inside those extents (head < heads, j < dim). All 32 lanes of
    // a warp run the same trip counts, so the shuffles are converged. The output overlaps no input.
    unsafe {
        let s=p.shape;let (heads,d)=(s.heads as usize,s.dim as usize);let width=heads*d;
        let (x0,x1,x2,w)=(p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32,p.weight as *const f32);
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        if ONE && tid>=32 {return;}
        let mut head=if ONE {core::arch::nvptx::_block_idx_x() as usize} else {core::arch::nvptx::_block_idx_x() as usize*4+tid/32};
        let stride=core::arch::nvptx::_grid_dim_x() as usize*if ONE {1} else {4};
        while head<heads {
            let base=head*d;let q=x0.add(base);let k=x0.add(width+base);
            let qnorm=root(lane_square_sum!(q,d,lane)+1e-6);let knorm=root(lane_square_sum!(k,d,lane)+1e-6);
            let scale=1.0/root(d as f32);let rate=exp(*w.add(head));
            let out=(p.output as *mut f32).add(head*(4*d+1));
            let mut j=lane;
            while j<d {
                *out.add(j)=(*q.add(j)/qnorm)*scale;*out.add(d+j)=*k.add(j)/knorm;*out.add(2*d+j)=*x0.add(2*width+base+j);
                *out.add(3*d+j)=s.bound*sigmoid(rate*(*x1.add(base+j)+*w.add(heads+base+j)));j+=32;
            }
            if lane==0 {*out.add(4*d)=round(sigmoid(*x2.add(head)));}
            head+=stride;
        }
    }
}

/// Op::GatedNorm, one warp per head.
pub(crate) fn gated_norm(p:Arguments) {
    gated_norm_heads::<false>(p)
}
pub(crate) fn gated_norm_one(p:Arguments) {gated_norm_heads::<true>(p)}
#[inline(always)]
fn gated_norm_heads<const ONE:bool>(p:Arguments) {
    // SAFETY: Shape::lengths was checked by the host: x0 and x1 hold heads*dim values, the weight dim values, the
    // output heads*dim values with one disjoint dim-slice per head. Indexes stay inside (head < heads, j < dim).
    // All lanes of a warp take the same trip counts. The output overlaps no input.
    unsafe {
        let s=p.shape;let (heads,d)=(s.heads as usize,s.dim as usize);
        let (x0,x1,w)=(p.input[0] as *const f32,p.input[1] as *const f32,p.weight as *const f32);
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        if ONE && tid>=32 {return;}
        let mut head=if ONE {core::arch::nvptx::_block_idx_x() as usize} else {core::arch::nvptx::_block_idx_x() as usize*4+tid/32};
        let stride=core::arch::nvptx::_grid_dim_x() as usize*if ONE {1} else {4};
        while head<heads {
            let base=head*d;let row=x0.add(base);
            let scale=1.0/root(lane_square_sum!(row,d,lane)/d as f32+s.epsilon);
            let out=(p.output as *mut f32).add(head*d);
            let mut j=lane;
            while j<d {*out.add(j)=round(((*row.add(j)*scale)**w.add(j))*sigmoid(*x1.add(base+j)));j+=32;}
            head+=stride;
        }
    }
}

/// Op::PoolScores, one warp per pool row: lane = head (heads beyond 32 in further rounds), then the head terms are
/// added in ascending head order by broadcast, exactly the original `score+=` chain.
pub(crate) fn pool_scores(p:Arguments) {
    // SAFETY: Shape::lengths was checked by the host: x0 holds heads*dim values, x1 rows*dim, x2 heads; the output
    // has max(rows,1) values, one per unit. Reads stay inside (head < heads, row < rows, j < dim) and each row's
    // output word has one writer (lane 0 of its warp). All lanes run the same trip counts.
    unsafe {
        let s=p.shape;let (h,d,r)=(s.heads as usize,s.dim as usize,s.rows as usize);
        let (x0,x1,x2)=(p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32);
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        let units=r.max(1);
        let mut index=core::arch::nvptx::_block_idx_x() as usize*4+tid/32;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*4;
        while index<units {
            let mut score=0.0f32;
            if index<r {
                let mut round_start=0;
                while round_start<h {
                    let head=round_start+lane;
                    let mut term=0.0f32;
                    if head<h {
                        let mut dot=0.0f32;let a=x0.add(head*d);let b=x1.add(index*d);
                        let mut j=0;while j<d {dot+=*a.add(j)**b.add(j);j+=1;}
                        term=(*x2.add(head)*(1.0/root(h as f32)))*(dot*(1.0/root(d as f32))).max(0.0);
                    }
                    let mut source=0u32;
                    while source<32 {
                        let value:f32;asm!("shfl.sync.idx.b32 {o}, {v}, {s}, 31, -1;",o=out(reg32)value,v=in(reg32)term,s=in(reg32)source,options(nomem,nostack));
                        if round_start+(source as usize)<h {score+=value;}
                        source+=1;
                    }
                    round_start+=32;
                }
            }
            if lane==0 {*(p.output as *mut f32).add(index)=score;}
            index+=stride;
        }
    }
}

/// Op::Step, Op::StepPlanar and Op::StepColumns with dim 128: one block per head, thread = value column. The three per-key
/// vectors of the head are staged once in shared memory. The direct layouts (Step, StepPlanar) read a thread's column of the
/// state once into registers and run both folds as straight-line code; StepColumns recomputes the column from the state in
/// each pass and parks its words in shared memory for the value-major layout.
pub(crate) fn step_columns_128(p:&Arguments) {
    // SAFETY: Shape::lengths was checked by the host: x0 holds heads*513 values, x1 heads*16384, the output
    // heads*128*129 output words. Step writes key-major state then the output vector per head; StepPlanar writes the same
    // words with every head's state first and every head's output vector after all states (state at head*16384 + key*128 + value,
    // output at heads*16384 + head*128 + value); StepColumns writes value-major units of 129 words. In every layout each
    // (head, key, value) and each (head, value) output word has one writer and no input aliases the output.
    // The layout choice is uniform for the whole launch. The block has exactly 128 threads and
    // every thread reaches every barrier (the loop bounds depend only on the shape). Shared use is 3 x 128 words
    // written by the thread of the same index and read after the first barrier, then 128 rows of 9 words: row t is
    // written by thread t and read by any thread only after the next barrier, and rewritten only after the one after. Reads stay inside the head's slices.
    // Every shared read follows a write: the three per-key vectors are written by all 128 threads (the predicate demands exactly
    // 128) before the first barrier; a stage row is written by its own thread before the barrier that precedes every read of it.
    // There is no partial block: dim == 128 and the block has 128 threads.
    // The direct layouts use the three staged vectors only (read by four-word loads: the array is 16-byte aligned and each
    // vector starts at a multiple of 512 bytes); a thread's column lives in its own registers, and each state word of the
    // output is stored once by the thread of its value column.
    // The caller retains the immutable arguments through this call. No reference escapes.
    unsafe {
        const D:usize=128;
        let planar=p.shape.op==crate::causal_ops::Op::StepPlanar;let direct=planar || p.shape.op==crate::causal_ops::Op::Step;
        let heads=p.shape.heads as usize;let value=core::arch::nvptx::_thread_idx_x() as usize;
        let (x0,x1)=(p.input[0] as *const f32,p.input[1] as *const f32);
        let smem=crate::ptx::shared::<{3*D*4+D*9*4}>();let stage=smem+(3*D*4) as u32;
        let mut head=core::arch::nvptx::_block_idx_x() as usize;
        let stride=core::arch::nvptx::_grid_dim_x() as usize;
        while head<heads {
            let pp=x0.add(head*(4*D+1));let state=x1.add(head*D*D);
            crate::ptx::store(smem+(value*4) as u32,exp(*pp.add(3*D+value)));
            crate::ptx::store(smem+((D+value)*4) as u32,*pp.add(D+value));
            crate::ptx::store(smem+((2*D+value)*4) as u32,*pp.add(value));
            asm!("bar.sync 0;");
            if direct {
                // The register form: a thread's column of 128 state words is read once into 128 local values (every index
                // below is a constant, so they are registers), and both folds are straight-line code: two four-word shared
                // loads and the arithmetic of four keys a turn. The same products, the same adds, key ascending, as the two
                // passes below; the second pass's product `state*decay` is the first fold's value, kept, not computed again.
                let (decays,update,keys)=(smem,smem+(D*4) as u32,smem+(2*D*4) as u32);
                let column=state.add(value);
                let mut col=[0.0f32;D];
                unroll!(8,a,{unroll!(4,b,{unroll!(4,e,{let i=(a*4+b)*4+e;col[i]=*column.add(i*D);});});});
                let mut prediction=0.0f32;
                unroll!(8,a,{unroll!(4,b,{let at=((a*4+b)*16) as u32;
                    let (r,u)=(crate::ptx::load4(decays+at),crate::ptx::load4(update+at));
                    unroll!(4,e,{let i=(a*4+b)*4+e;let c=col[i]*f32::from_bits(r[e]);col[i]=c;prediction+=c*f32::from_bits(u[e]);});
                });});
                let delta=(*pp.add(2*D+value)-prediction)**pp.add(4*D);
                let head_out=(p.output as *mut f32).add(head*D*(D+1));
                let (state_out,vector_out)=if planar {((p.output as *mut f32).add(head*D*D),(p.output as *mut f32).add(heads*D*D+head*D))} else {(head_out,head_out.add(D*D))};
                let column_out=state_out.add(value);
                let mut sum=0.0f32;
                unroll!(8,a,{unroll!(4,b,{let at=((a*4+b)*16) as u32;
                    let (u,k)=(crate::ptx::load4(update+at),crate::ptx::load4(keys+at));
                    unroll!(4,e,{let i=(a*4+b)*4+e;let mut c=col[i];c+=f32::from_bits(u[e])*delta;*column_out.add(i*D)=c;sum+=c*f32::from_bits(k[e]);});
                });});
                *vector_out.add(value)=round(sum);
                asm!("bar.sync 0;");
                head+=stride;
                continue;
            }
            // StepColumns, the value-major layout. The column is recomputed from the immutable state in each pass (same
            // product, same bits), so no per-thread array is needed and every load of a pass is independent.
            // Keys go in groups of 16 for the loads of the first pass, and in groups of 8 for the second pass, whose results
            // are parked in shared memory (9-word rows, conflict free) and written out by the whole block, so a warp stores
            // runs of 8 words (32 bytes) instead of 32 scattered words. The shared array is small on purpose: this kernel's
            // static shared memory sets the occupancy of every causal operation, including the large copies of other lanes.
            let mut prediction=0.0f32;
            let mut group=0;while group<D/16 {
                let mut v=[0.0f32;16];let mut i=0;while i<16 {v[i]=*state.add((group*16+i)*D+value);i+=1;}
                i=0;while i<16 {let key=group*16+i;let c=v[i]*crate::ptx::load(smem+(key*4) as u32);prediction+=c*crate::ptx::load(smem+((D+key)*4) as u32);i+=1;}
                group+=1;
            }
            let delta=(*pp.add(2*D+value)-prediction)**pp.add(4*D);
            let head_out=(p.output as *mut f32).add(head*D*(D+1));
            let mut sum=0.0f32;
            group=0;while group<D/8 {
                let mut v=[0.0f32;8];let mut i=0;while i<8 {v[i]=*state.add((group*8+i)*D+value);i+=1;}
                i=0;while i<8 {
                    let key=group*8+i;let mut c=v[i]*crate::ptx::load(smem+(key*4) as u32);
                    c+=crate::ptx::load(smem+((D+key)*4) as u32)*delta;
                    crate::ptx::store(stage+((value*9+i)*4) as u32,c);
                    sum+=c*crate::ptx::load(smem+((2*D+key)*4) as u32);i+=1;
                }
                asm!("bar.sync 0;");
                i=0;while i<8 {
                    let at=i*D+value;let (u,k)=(at/8,at%8);
                    *head_out.add(u*(D+1)+group*8+k)=crate::ptx::load(stage+((u*9+k)*4) as u32);i+=1;
                }
                asm!("bar.sync 0;");
                group+=1;
            }
            *head_out.add(value*(D+1)+D)=round(sum);
            asm!("bar.sync 0;");
            head+=stride;
        }
    }
}

/// Op::StepLayout with dim 128: a pure copy of words. Block = (head, key group of 8); thread = value. The eight
/// keys of one value row are adjacent words, so the strided reads share sectors and the writes are coalesced.
pub(crate) fn step_layout_128(p:Arguments) {
    // SAFETY: Shape::lengths was checked by the host: the input holds heads*128*129 words and the output the same
    // count with one writer per word. Groups 0..15 copy keys 8g..8g+7 of value row t to out[key*128+t]; group 16
    // copies the output word (key 128) to out[16384+t]; together exactly the 16512 words of each head, once.
    unsafe {
        const D:usize=128;
        let heads=p.shape.heads as usize;let t=core::arch::nvptx::_thread_idx_x() as usize;
        let x=p.input[0] as *const f32;let out=p.output as *mut f32;
        let mut block=core::arch::nvptx::_block_idx_x() as usize;
        let stride=core::arch::nvptx::_grid_dim_x() as usize;
        while block<heads*17 {
            let (head,group)=(block/17,block%17);
            let row=x.add((head*D+t)*(D+1));let dest=out.add(head*(D*D+D));
            if group<16 {
                let mut v=[0.0f32;8];let mut k=0;while k<8 {v[k]=*row.add(group*8+k);k+=1;}
                k=0;while k<8 {*dest.add((group*8+k)*D+t)=v[k];k+=1;}
            } else {*dest.add(D*D+t)=*row.add(D);}
            block+=stride;
        }
    }
}

/// Op::Softmax, one block per head. The scores' exponentials go to the output row first (every lane, independent),
/// then one lane adds them in ascending slot order from 0.0 (the law's chain), then every lane divides its slots.
/// The maximum is taken in any order: a maximum does not depend on it, and exp(v - max) is bit-equal for either zero.
pub(crate) fn softmax<const GROUPED:bool>(p:Arguments,heads_per_token:usize,count_stride:usize) {
    // SAFETY: Shape::lengths was checked by the host: x0 holds heads*rows scores, x1 one count word, the output
    // heads*rows words with one disjoint row per head. For GROUPED the flattened head includes the token, and x1 points
    // at the first count of a checked [tokens,count_stride] selection; head/heads_per_token indexes its count. The
    // prompt host admission bounds tokens, heads, stride and all scores/selection/output extents; every block reads one count. A live count above rows would make the generic path's slice
    // panic; here it traps the same way. The block has 128 threads and every thread reaches every barrier (the loop
    // bounds depend on the shape and the count word, which all threads read alike). The only shared words are the
    // four per-warp maxima and the denominator. The output row doubles as the exponential scratch: a lane reads
    // only its own words, or, for the chain, every word after a barrier.
    // Every shared read follows a write: the four maxima are written by lane 0 of each of the four warps (the predicate demands
    // 128 threads) before the barrier that precedes the read of all four; the denominator is written by thread 0 before the next
    // barrier. With a live count of 0 nothing shared is read. The exponential scratch words are written (slots below the count) before
    // the barrier that precedes any other lane reads them.
    unsafe {
        let s=p.shape;let (heads,r)=(s.heads as usize,s.rows as usize);
        let (x0,x1)=(p.input[0] as *const f32,p.input[1] as *const f32);

        let tid=core::arch::nvptx::_thread_idx_x() as usize;
        let smem=crate::ptx::shared::<32>();
        let mut head=core::arch::nvptx::_block_idx_x() as usize;
        let stride=core::arch::nvptx::_grid_dim_x() as usize;
        while head<heads {
            let count=*x1.add(if GROUPED {(head/heads_per_token)*count_stride} else {0}) as usize;
            if count>r {core::arch::nvptx::trap();}
            let row=x0.add(head*r);let out=(p.output as *mut f32).add(head*r);
            let mut i=count+tid;while i<r {*out.add(i)=0.0;i+=128;}
            if count>0 {
                let mut m=f32::NEG_INFINITY;
                i=tid;while i<count {m=m.max(*row.add(i));i+=128;}
                let mut distance=16u32;
                while distance!=0 {
                    let other:f32;asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)other,v=in(reg32)m,d=in(reg32)distance,options(nomem,nostack));
                    m=m.max(other);distance/=2;
                }
                if tid&31==0 {crate::ptx::store(smem+((tid/32)*4) as u32,m);}
                asm!("bar.sync 0;");
                m=crate::ptx::load(smem).max(crate::ptx::load(smem+4)).max(crate::ptx::load(smem+8)).max(crate::ptx::load(smem+12));
                i=tid;while i<count {*out.add(i)=exp(*row.add(i)-m);i+=128;}
                asm!("bar.sync 0;");
                if tid==0 {
                    let mut denominator=0.0f32;let mut at=0;
                    while at+16<=count {
                        let mut v=[0.0f32;16];let mut k=0;while k<16 {v[k]=*out.add(at+k);k+=1;}
                        k=0;while k<16 {denominator+=v[k];k+=1;}
                        at+=16;
                    }
                    while at<count {denominator+=*out.add(at);at+=1;}
                    crate::ptx::store(smem+16,denominator);
                }
                asm!("bar.sync 0;");
                let denominator=crate::ptx::load(smem+16);
                i=tid;while i<count {*out.add(i)=round(*out.add(i)/denominator);i+=128;}
            }
            asm!("bar.sync 0;");
            head+=stride;
        }
    }
}

/// Op::Gather, one warp per selected row: a pure copy of words, coalesced across the warp.
pub(crate) fn gather(p:Arguments) {
    // SAFETY: Shape::lengths was checked by the host: x0 holds rows*dim words, x1 cols (>= units) ids, x2 one count
    // word, the output units*dim words with one disjoint dim-slice per unit. A live id outside the rows would make the
    // generic path's slice panic; here it traps the same way. Reads stay inside the id's row; each output word has
    // one writer. All lanes of a warp take the same branches.
    unsafe {
        let s=p.shape;let (rows,d,units)=(s.rows as usize,s.dim as usize,s.cols as usize);
        let (x0,x1,x2)=(p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32);
        let count=*x2 as usize;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        let mut index=core::arch::nvptx::_block_idx_x() as usize*4+tid/32;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*4;
        while index<units {
            let out=(p.output as *mut f32).add(index*d);
            if index<count {
                let row=*x1.add(index) as usize;
                if row>=rows {core::arch::nvptx::trap();}
                let source=x0.add(row*d);
                let mut j=lane;while j<d {*out.add(j)=*source.add(j);j+=32;}
            } else {
                let mut j=lane;while j<d {*out.add(j)=0.0;j+=32;}
            }
            index+=stride;
        }
    }
}

/// Op::PoolChannels, one warp per channel. The generic unit walks every pool group of its channel one after the other (a
/// single 128-thread block for the 128 channels of a pool, 13 ms at a depth of 8,192 slots); the groups are independent, so
/// here lane L takes groups L, L+32, ... of the channel, each group with exactly the generic expressions in the generic
/// order (maximum over its taps, the sum of exponentials in tap order, then the rounded products in tap order). The keys and
/// gates are copied by the lanes too, slot by slot; the pooled values read the inputs, never the copy, so no lane waits for another.
pub(crate) fn pool_channels(p:Arguments) {
    // SAFETY: Shape::lengths was checked by the host: x0 and x1 hold (rows-1)*dim words, x2 and x3 dim words, the weight
    // taps*dim words, the output dim units of 2*rows+rows/taps words, each unit's slice disjoint. Reads stay inside
    // (slot < rows-1 for x0/x1, channel < dim for x2/x3, tap < taps for the weight, group*taps+tap < rows) and every
    // output word of a channel has one writer (the lane that owns its slot or group). No barrier, no early exit.
    unsafe {
        let s=p.shape;let (d,t,r)=(s.dim as usize,s.taps as usize,s.rows as usize);
        let (x0,x1,x2,x3)=(p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32,p.input[3] as *const f32);
        let w=p.weight as *const f32;
        let words=2*r+r/t;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        let mut j=core::arch::nvptx::_block_idx_x() as usize*4+tid/32;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*4;
        while j<d {
            let out=(p.output as *mut f32).add(j*words);
            let key=|slot:usize|->f32 {if slot+1<r {*x0.add(slot*d+j)} else {*x2.add(j)}};
            let gate=|slot:usize|->f32 {if slot+1<r {*x1.add(slot*d+j)} else {*x3.add(j)}};
            let mut slot=lane;
            while slot<r {*out.add(slot)=key(slot);*out.add(r+slot)=gate(slot);slot+=32;}
            let mut group=lane;
            while group<r/t {
                let mut maximum=f32::NEG_INFINITY;for tap in 0..t {maximum=maximum.max(gate(group*t+tap)+*w.add(tap*d+j));}
                let mut sum=0.0f32;for tap in 0..t {sum+=exp(gate(group*t+tap)+*w.add(tap*d+j)-maximum);}
                let mut value=0.0f32;for tap in 0..t {let q=round(exp(gate(group*t+tap)+*w.add(tap*d+j)-maximum)/sum);
                    value+=round(q*key(group*t+tap));}
                *out.add(2*r+group)=round(value);
                group+=32;
            }
            j+=stride;
        }
    }
}
