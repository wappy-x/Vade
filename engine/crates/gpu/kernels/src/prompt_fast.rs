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
use crate::strip_warp::unroll;

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
#[export_name="prompt_prepare"]
pub extern "ptx-kernel" fn entry_prompt_prepare(p:Arguments) {prompt_prepare(p)}
#[inline(always)]
pub(crate) fn prompt_prepare(p:Arguments) {
    // SAFETY: The host (prompt_admission::wrapped_kernel) checked Shape::lengths for Op::Wrapped over causal Prepare with
    // causal_ops::fast_admitted: per token, x0 holds 3*width, x1 width and x2 heads values, the weight heads+width values
    // (shared by all tokens), and each (token, head) unit owns the disjoint 4*dim+1 output words at unit*(4*dim+1);
    // owners live and disjoint through completion. Indexes stay inside (token < tokens, head < heads, j < dim). All 32
    // lanes of a warp take the same trip counts, so the shuffles are converged. A block of other than THREADS threads traps.
    // StepDecay uses the same checked prepared input rows, no weight and a private tokens*heads*dim output.
    // Each thread owns its grid-strided output words. The input address follows the same token/head/key map
    // as prompt_ops::unit. Four independent reads precede their unchanged exponentials; each tail is guarded.
    unsafe {
        let s=p.shape;let inner=s.inner;
        if s.op==Op::StepDecay {
            if core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
            let d=s.dim as usize;let units=s.tokens as usize*s.heads as usize*d;
            let input=p.input[0] as *const f32;let output=p.output as *mut f32;
            let mut first=core::arch::nvptx::_block_idx_x() as usize*THREADS+core::arch::nvptx::_thread_idx_x() as usize;
            let stride=core::arch::nvptx::_grid_dim_x() as usize*THREADS;
            while first<units {
                let mut raw=[0.0f32;4];
                unroll!(4,i,{let at=first+i*stride;if at<units {raw[i]=*input.add((at/d)*(4*d+1)+3*d+at%d);}});
                unroll!(4,i,{let at=first+i*stride;if at<units {*output.add(at)=exp(raw[i]);}});
                first+=4*stride;
            }
            return;
        }
        if s.op!=Op::Wrapped || inner.op!=Causal::Prepare || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (heads,d,tokens)=(inner.heads as usize,inner.dim as usize,s.tokens as usize);let width=heads*d;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        let mut unit=core::arch::nvptx::_block_idx_x() as usize*4+tid/32;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*4;
        let fuse_exp=s.first==crate::prompt_ops::PREPARE_DECAY_EXP;
        let w=p.weight as *const f32;
        while unit<tokens*heads {
            let (token,head)=(unit/heads,unit%heads);
            let x0=(p.input[0] as *const f32).add(token*3*width);let x1=(p.input[1] as *const f32).add(token*width);let x2=(p.input[2] as *const f32).add(token*heads);
            let base=head*d;let q=x0.add(base);let k=x0.add(width+base);
            let scale=1.0/root(d as f32);
            let rate_l0=if lane==0 {exp(*w.add(head))} else {0.0f32};
            let rate:f32;
            asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)rate,v=in(reg32)rate_l0,options(nomem,nostack));
            let out=(p.output as *mut f32).add(unit*(4*d+1));
            if d==128 {
                // Cache the 4 elements per lane in registers so q and k are read once from global memory.
                let mut qv=[0.0f32;4];let mut kv=[0.0f32;4];
                let mut qsum=0.0f32;let mut ksum=0.0f32;
                unroll!(4,i,{let j=lane+32*i;let a=*q.add(j);let b=*k.add(j);qv[i]=a;kv[i]=b;qsum+=a*a;ksum+=b*b;});
                let mut distance=16u32;
                while distance!=0 {
                    let (oq,ok):(f32,f32);
                    asm!("shfl.sync.down.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)oq,v=in(reg32)qsum,d=in(reg32)distance,options(nomem,nostack));
                    asm!("shfl.sync.down.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)ok,v=in(reg32)ksum,d=in(reg32)distance,options(nomem,nostack));
                    if lane+(distance as usize)<32 {qsum+=oq;ksum+=ok;}
                    distance/=2;
                }
                let (qtot,ktot):(f32,f32);
                asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)qtot,v=in(reg32)qsum,options(nomem,nostack));
                asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)ktot,v=in(reg32)ksum,options(nomem,nostack));
                let qnorm=root(qtot+1e-6);let knorm=root(ktot+1e-6);
                unroll!(4,i,{
                    let j=lane+32*i;
                    *out.add(j)=(qv[i]/qnorm)*scale;*out.add(d+j)=kv[i]/knorm;*out.add(2*d+j)=*x0.add(2*width+base+j);
                    let decay_raw=inner.bound*sigmoid(rate*(*x1.add(base+j)+*w.add(heads+base+j)));
                    *out.add(3*d+j)=if fuse_exp {exp(decay_raw)} else {decay_raw};
                });
            } else {
                let qnorm=root(lane_square_sum!(q,d,lane)+1e-6);let knorm=root(lane_square_sum!(k,d,lane)+1e-6);
                let mut j=lane;
                while j<d {
                    *out.add(j)=(*q.add(j)/qnorm)*scale;*out.add(d+j)=*k.add(j)/knorm;*out.add(2*d+j)=*x0.add(2*width+base+j);
                    let decay_raw=inner.bound*sigmoid(rate*(*x1.add(base+j)+*w.add(heads+base+j)));
                    *out.add(3*d+j)=if fuse_exp {exp(decay_raw)} else {decay_raw};j+=32;
                }
            }
            if lane==0 {*out.add(4*d)=round(sigmoid(*x2.add(head)));}
            unit+=stride;
        }
    }
}

/// Wrapped `GatedNorm`: one warp per (token, head).
#[export_name="prompt_gated_norm"]
pub extern "ptx-kernel" fn entry_prompt_gated_norm(p:Arguments) {prompt_gated_norm(p)}
#[inline(always)]
pub(crate) fn prompt_gated_norm(p:Arguments) {
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
            let out=(p.output as *mut f32).add(unit*d);
            if d==128 {
                let mut rv=[0.0f32;4];let mut sum=0.0f32;
                unroll!(4,i,{let v=*row.add(lane+32*i);rv[i]=v;sum+=v*v;});
                let mut distance=16u32;
                while distance!=0 {
                    let other:f32;asm!("shfl.sync.down.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)other,v=in(reg32)sum,d=in(reg32)distance,options(nomem,nostack));
                    if lane+(distance as usize)<32 {sum+=other;}
                    distance/=2;
                }
                let total:f32;asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)total,v=in(reg32)sum,options(nomem,nostack));
                let scale=1.0/root(total/d as f32+inner.epsilon);
                unroll!(4,i,{let j=lane+32*i;*out.add(j)=round(((rv[i]*scale)**w.add(j))*sigmoid(*x1.add(base+j)));});
            } else {
                let scale=1.0/root(lane_square_sum!(row,d,lane)/d as f32+inner.epsilon);
                let mut j=lane;
                while j<d {*out.add(j)=round(((*row.add(j)*scale)**w.add(j))*sigmoid(*x1.add(base+j)));j+=32;}
            }
            unit+=stride;
        }
    }
}

/// Registers per lane that hold one token's experts: up to 16 per lane, so at most 512 experts.
const PER_LANE:usize=16;
/// Wrapped `Route` for one group (`heads == 1`, `dim == 1`): one warp per token. With one group every expert is kept, the
/// group score is never read, and the selection is `cols` rounds of "largest corrected score, lowest id on a tie".
#[export_name="prompt_route"]
pub extern "ptx-kernel" fn entry_prompt_route(p:Arguments) {prompt_route(p)}
#[inline(always)]
pub(crate) fn prompt_route(p:Arguments) {
    // SAFETY: The host (prompt_admission::route_admitted) checked Shape::lengths for Op::Wrapped over causal Route with one
    // group and one kept group: per token x0 holds `rows` logits, the weight `rows` biases (shared), and the token owns
    // the disjoint 2*cols + 2*rows + 2 output words at token*(2*cols+2*rows+2); `rows` <= 32*PER_LANE and `cols` <= 64.
    // ROUTE PLANES (s.first == prompt_ops::ROUTE_PLANES, admitted by prompt_admission::planes_admitted, never split into runs):
    // the token writes its k IDs at token*k and its k weights at tokens*k + token*k, and clears the words of its own row at
    // or above 2*tokens*k. Those plane words are the first 2*tokens*k of the tokens*words output (words >= 2*cols), so
    // every output word is written by exactly one lane, as in the plain form.
    // Every lane takes the same trip counts and every shuffle is converged. A register array is only indexed by loop
    // counters that unroll (selects over i), never by a data value.
    unsafe {
        let s=p.shape;let inner=s.inner;
        if s.op!=Op::Wrapped || inner.op!=Causal::Route || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (r,k,tokens)=(inner.rows as usize,inner.cols as usize,s.tokens as usize);
        let words=2*k+2*r+2*inner.heads as usize;let planes=s.first==crate::prompt_ops::ROUTE_PLANES;
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
                    let (id,weight)=if planes {let base=p.output as *mut f32;(base.add(token*k+slot),base.add((tokens+token)*k+slot))} else {(out.add(slot),out.add(k+slot))};
                    *id=held_id[part];*weight=v;
                }
                part+=1;
            }
            let clear=if planes {(2*tokens*k).saturating_sub(token*words).min(words)} else {2*k};
            let mut j=clear+lane;while j<words {*out.add(j)=0.0;j+=32;}
            token+=stride;
        }
    }
}

/// Registers one thread holds for the expert order of a fold.
const FOLD_MAX:usize=8;
/// Wrapped `ExpertFold`: one thread per (token, feature) with the token's experts in registers. The fold visits the experts
/// in ascending id order (ties and gaps as the generic selection loop treats them), each term rounded to BF16 twice.
#[export_name="prompt_fold"]
pub extern "ptx-kernel" fn entry_prompt_fold(p:Arguments) {prompt_fold(p)}
#[inline(always)]
pub(crate) fn prompt_fold(p:Arguments) {
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
        let tid=core::arch::nvptx::_thread_idx_x() as usize;
        let mut index=core::arch::nvptx::_block_idx_x() as usize*THREADS+tid;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*THREADS;
        if (k&127)==0 && ((p.input[2]|p.output)&15)==0 && (inner.taps==0 || (p.input[3]&15)==0) {
            // Vectorized 4-feature path: 1 warp owns 128 features of ONE token (4 floats/lane via v4.f32),
            // cutting warp sorts, shuffles, and memory instructions by 4x with bit-identical per-feature chains.
            let lane=tid&31;
            let k4=k/4;
            let total4=tokens*k4;
            let mut idx4=index;
            while idx4<total4 {
                let (token,f4)=(idx4/k4,idx4%k4);
                let feature=f4*4;
                let ids=(p.input[0] as *const f32).add(token*r);let weights=(p.input[1] as *const f32).add(token*r);let rows=(p.input[2] as *const f32).add(token*r*k);
                let mut packed_order=0u32;
                if lane==0 {
                    let mut id=[0.0f32;FOLD_MAX];let mut i=0;while i<FOLD_MAX {if i<r {id[i]=*ids.add(i);}i+=1;}
                    let mut have=false;let mut previous=0.0f32;
                    let mut step=0;while step<FOLD_MAX {
                        if step<r {
                            let mut selected=0usize;let mut best=f32::INFINITY;
                            let mut slot=0;while slot<FOLD_MAX {if slot<r && (!have || id[slot]>previous) && id[slot]<best {selected=slot;best=id[slot];}slot+=1;}
                            packed_order|=(selected as u32)<<(4*step);previous=best;have=true;
                        }
                        step+=1;
                    }
                }
                let order:u32;
                asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)order,v=in(reg32)packed_order,options(nomem,nostack));
                let w_lane=if lane<r {*weights.add(lane)} else {0.0f32};
                let (mut v0,mut v1,mut v2,mut v3)=(0.0f32,0.0f32,0.0f32,0.0f32);
                let mut step=0;while step<FOLD_MAX {
                    if step<r {
                        let selected=((order>>(4*step))&15) as usize;
                        let w_sel:f32;
                        asm!("shfl.sync.idx.b32 {o}, {v}, {s}, 31, -1;",o=out(reg32)w_sel,v=in(reg32)w_lane,s=in(reg32)(selected as u32),options(nomem,nostack));
                        let (a0,a1,a2,a3):(f32,f32,f32,f32);
                        asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",
                            a=out(reg32)a0,b=out(reg32)a1,c=out(reg32)a2,d=out(reg32)a3,
                            p=in(reg64)rows.add(selected*k+feature),options(nostack,readonly));
                        v0=round(v0+round(a0*w_sel));
                        v1=round(v1+round(a1*w_sel));
                        v2=round(v2+round(a2*w_sel));
                        v3=round(v3+round(a3*w_sel));
                    }
                    step+=1;
                }
                if inner.taps!=0 {
                    let (s0,s1,s2,s3):(f32,f32,f32,f32);
                    asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",
                        a=out(reg32)s0,b=out(reg32)s1,c=out(reg32)s2,d=out(reg32)s3,
                        p=in(reg64)(p.input[3] as *const f32).add(token*k+feature),options(nostack,readonly));
                    v0=round(v0+s0);v1=round(v1+s1);v2=round(v2+s2);v3=round(v3+s3);
                }
                asm!("st.global.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                    p=in(reg64)(p.output as *mut f32).add(idx4*4),
                    a=in(reg32)v0,b=in(reg32)v1,c=in(reg32)v2,d=in(reg32)v3,options(nostack));
                idx4+=stride;
            }
            return;
        }
        if (k&31)==0 {
            // Every warp stays within a single token: lane 0 sorts the `r <= 8` expert IDs once and broadcasts the packed
            // visiting order to all 32 lanes, cutting sorting and router-metadata loads by 32x.
            let lane=tid&31;
            while index<total {
                let (token,feature)=(index/k,index%k);
                let ids=(p.input[0] as *const f32).add(token*r);let weights=(p.input[1] as *const f32).add(token*r);let rows=(p.input[2] as *const f32).add(token*r*k);
                let mut packed_order=0u32;
                if lane==0 {
                    let mut id=[0.0f32;FOLD_MAX];let mut i=0;while i<FOLD_MAX {if i<r {id[i]=*ids.add(i);}i+=1;}
                    let mut have=false;let mut previous=0.0f32;
                    let mut step=0;while step<FOLD_MAX {
                        if step<r {
                            let mut selected=0usize;let mut best=f32::INFINITY;
                            let mut slot=0;while slot<FOLD_MAX {if slot<r && (!have || id[slot]>previous) && id[slot]<best {selected=slot;best=id[slot];}slot+=1;}
                            packed_order|=(selected as u32)<<(4*step);previous=best;have=true;
                        }
                        step+=1;
                    }
                }
                let order:u32;
                asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)order,v=in(reg32)packed_order,options(nomem,nostack));
                let w_lane=if lane<r {*weights.add(lane)} else {0.0f32};
                let mut term=[0.0f32;FOLD_MAX];
                let mut step=0;while step<FOLD_MAX {
                    if step<r {
                        let selected=((order>>(4*step))&15) as usize;
                        let w_sel:f32;
                        asm!("shfl.sync.idx.b32 {o}, {v}, {s}, 31, -1;",o=out(reg32)w_sel,v=in(reg32)w_lane,s=in(reg32)(selected as u32),options(nomem,nostack));
                        let val:f32;
                        asm!("ld.global.nc.f32 {v}, [{p}];",v=out(reg32)val,p=in(reg64)rows.add(selected*k+feature),options(nostack,readonly));
                        term[step]=round(val*w_sel);
                    }
                    step+=1;
                }
                let mut value=0.0f32;let mut step=0;while step<FOLD_MAX {if step<r {value=round(value+term[step]);}step+=1;}
                *(p.output as *mut f32).add(index)=if inner.taps!=0 {round(value+*(p.input[3] as *const f32).add(token*k+feature))} else {value};
                index+=stride;
            }
            return;
        }
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

/// `PoolScoresRows` (the indexer's pool scores of a group of tokens): one warp per (token, pool), lane = head (heads beyond
/// 32 in further rounds); the head terms are then added in ascending head order by broadcast, exactly the unit's `score+=`
/// chain. This is `causal_fast::pool_scores`, the one-row kernel, with the token's own query row, its own head weights and its
/// own causal pool count. The generic unit ran the 32 dots of a word one after the other in ONE thread, so a launch cost a
/// chain of heads * dim terms whatever it held (a 4-row verify pass: 349 us in each index layer).
#[export_name="prompt_pool_scores"]
pub extern "ptx-kernel" fn entry_prompt_pool_scores(p:Arguments) {prompt_pool_scores(p)}
#[inline(always)]
pub(crate) fn prompt_pool_scores(p:Arguments) {
    // SAFETY: The host (prompt_admission::pool_scores_admitted, after Shape::lengths) passes same-context contiguous owners
    // that live through stream completion: x0 holds tokens*heads*dim query words, x1 rows*dim key words, x2 tokens*heads
    // head weights; the private output has tokens*rows words, one per unit, at index = token*rows+pool, and overlaps no
    // input. Reads stay inside those extents (token < tokens, head < heads, pool < rows, j < dim). Each output word has one
    // writer (lane 0 of the unit's warp), and every unit below tokens*rows is stored, so the output needs no fill. All lanes
    // of a warp run the same trip counts and reach every `shfl` together: the unit, its token and its pool test are the same
    // on the 32 lanes. No shared memory, no barrier. Any grid is correct (the loop strides by the grid).
    unsafe {
        let s=p.shape;
        if s.op!=Op::PoolScoresRows || s.heads==0 || s.dim==0 || s.taps==0 || s.rows==0 || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (m,h,d,t,r,first)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.first as usize);
        let (x0,x1,x2)=(p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32);
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        let units=m*r;
        let mut index=core::arch::nvptx::_block_idx_x() as usize*(THREADS/32)+tid/32;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*(THREADS/32);
        while index<units {
            let (token,pool)=(index/r,index%r);
            let mut score=0.0f32;
            // A pool that is not complete at the token's context scores zero, as in the unit.
            if pool<(first+token)/t {
                let mut round_start=0;
                while round_start<h {
                    let head=round_start+lane;
                    let mut term=0.0f32;
                    if head<h {
                        let mut dot=0.0f32;let a=x0.add((token*h+head)*d);let b=x1.add(pool*d);
                        let mut j=0;while j<d {dot+=*a.add(j)**b.add(j);j+=1;}
                        term=(*x2.add(token*h+head)*(1.0/root(h as f32)))*(dot*(1.0/root(d as f32))).max(0.0);
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
#[export_name="prompt_conv"]
pub extern "ptx-kernel" fn entry_prompt_conv(p:Arguments) {prompt_conv(p)}
#[inline(always)]
pub(crate) fn prompt_conv(p:Arguments) {
    // SAFETY: The host (prompt_admission::conv_admitted) checked Shape::lengths for Op::ConvRows with 1 <= taps <= 8: x0, x1, x2
    // hold tokens*width values, x3 holds 3*(taps-1)*width, the weight 3*width*taps, the output tokens*3*width. A thread owns the
    // output words (token, plane, ch) of one (plane, ch) (or 4 consecutive channels on the aligned taps==4 vectorized path) and
    // tokens [run*CONV_RUN, ..): disjoint across threads. Every read is at a position below taps-1 (old history row) or at
    // token+tap-(taps-1) < tokens (the plane), inside the extents above.
    unsafe {
        let s=p.shape;
        if s.op!=Op::ConvRows || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (m,width,taps)=(s.tokens as usize,s.heads as usize*s.dim as usize,s.taps as usize);
        let columns=3*width;let runs=m.div_ceil(CONV_RUN);
        let mut index=core::arch::nvptx::_block_idx_x() as usize*THREADS+core::arch::nvptx::_thread_idx_x() as usize;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*THREADS;
        if taps==4 && (width&3)==0 && ((p.input[0]|p.input[1]|p.input[2]|p.input[3]|p.output|p.weight)&15)==0 {
            let cols4=columns/4;
            let total4=cols4*runs;
            let planes=[p.input[0] as *const f32,p.input[1] as *const f32,p.input[2] as *const f32];
            let out_base=p.output as *mut f32;
            let w_base=p.weight as *const f32;
            let hist_base=p.input[3] as *const f32;
            let history=3*width;
            macro_rules! ld4 {($ptr:expr)=>{{
                let (a,b,c,d):(f32,f32,f32,f32);
                asm!("ld.global.nc.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",
                    a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg64)$ptr,options(nostack,readonly));
                [a,b,c,d]
            }};}
            while index<total4 {
                let (col4,run)=(index%cols4,index/cols4);
                let (plane,ch)=(col4/(width/4),(col4%(width/4))*4);
                let first=run*CONV_RUN;let last=(first+CONV_RUN).min(m);
                let current=planes[plane];let old=hist_base.add(plane*history);
                // Weights are stored [plane*width+ch+e][tap], 4 contiguous floats per channel.
                let w0=ld4!(w_base.add((plane*width+ch)*4));
                let w1=ld4!(w_base.add((plane*width+ch+1)*4));
                let w2=ld4!(w_base.add((plane*width+ch+2)*4));
                let w3=ld4!(w_base.add((plane*width+ch+3)*4));
                let at4=|pos:usize|->[f32;4] {if pos<3 {ld4!(old.add(pos*width+ch))} else {ld4!(current.add((pos-3)*width+ch))}};
                let mut win=[at4(first),at4(first+1),at4(first+2),at4(first+3)];
                let mut token=first;
                while token<last {
                    let s0=(((0.0f32+win[0][0]*w0[0])+win[1][0]*w0[1])+win[2][0]*w0[2])+win[3][0]*w0[3];
                    let s1=(((0.0f32+win[0][1]*w1[0])+win[1][1]*w1[1])+win[2][1]*w1[2])+win[3][1]*w1[3];
                    let s2=(((0.0f32+win[0][2]*w2[0])+win[1][2]*w2[1])+win[2][2]*w2[2])+win[3][2]*w2[3];
                    let s3=(((0.0f32+win[0][3]*w3[0])+win[1][3]*w3[1])+win[2][3]*w3[2])+win[3][3]*w3[3];
                    let y0=round(s0*sigmoid(s0));let y1=round(s1*sigmoid(s1));
                    let y2=round(s2*sigmoid(s2));let y3=round(s3*sigmoid(s3));
                    asm!("st.global.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                        p=in(reg64)out_base.add((token*3+plane)*width+ch),
                        a=in(reg32)y0,b=in(reg32)y1,c=in(reg32)y2,d=in(reg32)y3,options(nostack));
                    token+=1;
                    if token<last {win[0]=win[1];win[1]=win[2];win[2]=win[3];win[3]=at4(token+3);}
                }
                index+=stride;
            }
            return;
        }
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

/// Fast warp-level `CheckSelectionRows`: 1 warp per token coalesced-checks all `count` selected IDs via `ballot`.
/// Bit-exact match with `prompt_ops::selection_validation_status`.
#[export_name="prompt_check_selection"]
pub extern "ptx-kernel" fn entry_prompt_check_selection(p:Arguments) {
    // SAFETY: The host checked `Shape::lengths` for `Op::CheckSelectionRows`: `input[0]` holds `tokens*(slots+1)`
    // FP32 words, `output` holds `tokens` FP32 words disjoint from input, and all owners stay live through completion.
    // Each warp owns one disjoint token `< tokens`; all 32 lanes execute the same `count` loop and reach `ballot`
    // uniformly. Only lane 0 writes `output[token]`. No shared memory, no CTA barrier.
    unsafe {
        let s=p.shape;
        if s.op!=Op::CheckSelectionRows || s.slots==0 || core::arch::nvptx::_block_dim_x() as usize!=THREADS {core::arch::nvptx::trap();}
        let (tokens,slots,first)=(s.tokens as usize,s.slots as usize,s.first as usize);
        let words=slots+1;
        let tid=core::arch::nvptx::_thread_idx_x() as usize;let lane=tid&31;
        let mut token=core::arch::nvptx::_block_idx_x() as usize*4+tid/32;
        let stride=core::arch::nvptx::_grid_dim_x() as usize*4;
        let base_in=p.input[0] as *const f32;
        let out=p.output as *mut f32;
        while token<tokens {
            let row=base_in.add(token*words);
            let context=first+token;
            let count_raw:f32;
            asm!("ld.global.nc.f32 {v}, [{p}];",v=out(reg32)count_raw,p=in(reg64)row.add(slots),options(nostack,readonly));
            let limit=if slots<context {slots} else {context};
            if !count_raw.is_finite() || count_raw<0.0 || count_raw!=count_raw as u32 as f32 || count_raw>limit as f32 {
                if lane==0 {*out.add(token)=2.0;}
            } else {
                let count=count_raw as usize;
                let ctx_f=context as f32;
                let mut bad=false;
                let mut i=lane;
                while i<count {
                    let id:f32;
                    asm!("ld.global.nc.f32 {v}, [{p}];",v=out(reg32)id,p=in(reg64)row.add(i),options(nostack,readonly));
                    if !id.is_finite() || id<0.0 || id!=id as u32 as f32 || id>=ctx_f {bad=true;}
                    i+=32;
                }
                let any_bad=crate::ptx::ballot(bad)!=0;
                if lane==0 {*out.add(token)=if any_bad {1.0} else {0.0};}
            }
            token+=stride;
        }
    }
}

/// Fused KDA 4-tap convolution + `Prepare` (with optional `exp(decay)` fusion): eliminates the `tokens * 3 * width`
/// convolved DRAM write, DRAM read, and buffer allocation. Bit-exact with `prompt_conv` -> `prompt_prepare`.
/// 1 block (128 threads = 4 warps) owns 1 head and `4 * CONV_PREP_RUN = 64` consecutive tokens, staging that head's
/// `conv_weight` (`6,144 B`) and `dt_bias` (`512 B`) once in shared memory (`6,656 B`).
#[no_mangle]
pub extern "ptx-kernel" fn prompt_conv_prepare(p:crate::tc_abi::ConvPrepare) {
    // SAFETY: The host (`Backend::conv_prepare_rows`) checks exact extents, contiguity, alignment, and disjointness:
    // `q`, `k`, `v`, `a` each hold `tokens * heads * 128` FP32 words; `history` holds `3 * 3 * heads * 128` words;
    // `b` holds `tokens * heads` words; `conv_weight` holds `3 * heads * 128 * 4` words; `prep_weight` holds
    // `heads + heads * 128` words; `output` holds `tokens * heads * 513` words disjoint from all inputs and weights;
    // all owners stay live through stream completion. All 128 threads of a block reach the single `bar.sync 0` after
    // loading the head's 6,656-byte shared weight table; each warp then owns a disjoint 16-token run for that head
    // and executes converged full-mask warp shuffles (`mask = -1`). Each output word has exactly one writer.
    unsafe {
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        if _block_dim_x()!=crate::tc_abi::CONV_PREP_THREADS || p.dim!=128 || p.heads==0 || p.tokens==0
            || !(p.bound<0.0 && p.bound.is_finite())
            || (p.q|p.k|p.v|p.history|p.a|p.b|p.conv_weight|p.prep_weight|p.output)&3!=0
        {
            core::arch::nvptx::trap();
        }
        let (tokens,heads)=(p.tokens as usize,p.heads as usize);
        let width=heads*128;
        const BLK_TOK:usize=4*crate::tc_abi::CONV_PREP_RUN;
        let runs_per_head=tokens.div_ceil(BLK_TOK);
        let block=_block_idx_x() as usize;
        let head=block/runs_per_head;
        let blk_run=block%runs_per_head;
        if head>=heads {return;}
        let tid=_thread_idx_x() as usize;
        let (warp,lane)=(tid/32,tid&31);
        let base=head*128;
        // Static shared memory: 3*128*4 conv weights (6,144 B) + 128 dt_bias (512 B) = 6,656 B.
        let smem=crate::ptx::shared::<6656>();
        const DT_OFF:u32=6144;
        let cw_ptr=p.conv_weight as *const f32;
        let pw_ptr=p.prep_weight as *const f32;
        {
            let dt:f32;
            asm!("ld.global.nc.f32 {v}, [{p}];",v=out(reg32)dt,p=in(reg64)pw_ptr.add(heads+base+tid),options(nostack,readonly));
            crate::ptx::store(smem+DT_OFF+(tid*4) as u32,dt);
            let mut pl=0usize;
            while pl<3 {
                let src=cw_ptr.add((pl*width+base+tid)*4);
                let dst=smem+((pl*512+tid*4)*4) as u32;
                let (w0,w1,w2,w3):(f32,f32,f32,f32);
                asm!("ld.global.nc.f32 {v}, [{p}];",v=out(reg32)w0,p=in(reg64)src,options(nostack,readonly));
                asm!("ld.global.nc.f32 {v}, [{p}];",v=out(reg32)w1,p=in(reg64)src.add(1),options(nostack,readonly));
                asm!("ld.global.nc.f32 {v}, [{p}];",v=out(reg32)w2,p=in(reg64)src.add(2),options(nostack,readonly));
                asm!("ld.global.nc.f32 {v}, [{p}];",v=out(reg32)w3,p=in(reg64)src.add(3),options(nostack,readonly));
                asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",
                    p=in(reg32)dst,a=in(reg32)w0,b=in(reg32)w1,c=in(reg32)w2,d=in(reg32)w3,options(nostack));
                pl+=1;
            }
        }
        asm!("bar.sync 0;");
        let first=blk_run*BLK_TOK+warp*crate::tc_abi::CONV_PREP_RUN;
        let last=(first+crate::tc_abi::CONV_PREP_RUN).min(tokens);
        if first>=last {return;}
        let rate_l0=if lane==0 {exp(*pw_ptr.add(head))} else {0.0f32};
        let rate:f32;
        asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)rate,v=in(reg32)rate_l0,options(nomem,nostack));
        let scale=1.0/root(128.0f32);
        let fuse_exp=p.fuse_exp!=0;
        let (q_ptr,k_ptr,v_ptr)=(p.q as *const f32,p.k as *const f32,p.v as *const f32);
        let hist_base=p.history as *const f32;
        let (old_q,old_k,old_v)=(hist_base,hist_base.add(3*width),hist_base.add(6*width));
        let (a_ptr,b_ptr,out_base)=(p.a as *const f32,p.b as *const f32,p.output as *mut f32);
        macro_rules! ld_nc {($ptr:expr)=>{{
            let v:f32;
            asm!("ld.global.nc.f32 {v}, [{p}];",v=out(reg32)v,p=in(reg64)$ptr,options(nostack,readonly));
            v
        }};}
        let at_plane=|cur:*const f32,old:*const f32,pos:usize,j:usize|->f32 {
            let ch=base+j;
            if pos<3 {ld_nc!(old.add(pos*width+ch))} else {ld_nc!(cur.add((pos-3)*width+ch))}
        };
        let mut qw=[[0.0f32;4];4];
        let mut kw=[[0.0f32;4];4];
        let mut vw=[[0.0f32;4];4];
        unroll!(4,i,{
            let j=lane+32*i;
            unroll!(4,tap,{
                qw[i][tap]=at_plane(q_ptr,old_q,first+tap,j);
                kw[i][tap]=at_plane(k_ptr,old_k,first+tap,j);
                vw[i][tap]=at_plane(v_ptr,old_v,first+tap,j);
            });
        });
        let mut token=first;
        while token<last {
            let mut qv=[0.0f32;4];let mut kv=[0.0f32;4];
            let mut qsum=0.0f32;let mut ksum=0.0f32;
            unroll!(4,i,{
                let j=lane+32*i;
                let wq=crate::ptx::load4(smem+(j*4*4) as u32);
                let sq=(((0.0f32+qw[i][0]*f32::from_bits(wq[0]))+qw[i][1]*f32::from_bits(wq[1]))+qw[i][2]*f32::from_bits(wq[2]))+qw[i][3]*f32::from_bits(wq[3]);
                let qc=round(sq*sigmoid(sq));
                qv[i]=qc;qsum+=qc*qc;
                let wk=crate::ptx::load4(smem+((512+j*4)*4) as u32);
                let sk=(((0.0f32+kw[i][0]*f32::from_bits(wk[0]))+kw[i][1]*f32::from_bits(wk[1]))+kw[i][2]*f32::from_bits(wk[2]))+kw[i][3]*f32::from_bits(wk[3]);
                let kc=round(sk*sigmoid(sk));
                kv[i]=kc;ksum+=kc*kc;
            });
            let mut distance=16u32;
            while distance!=0 {
                let (oq,ok):(f32,f32);
                asm!("shfl.sync.down.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)oq,v=in(reg32)qsum,d=in(reg32)distance,options(nomem,nostack));
                asm!("shfl.sync.down.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)ok,v=in(reg32)ksum,d=in(reg32)distance,options(nomem,nostack));
                if lane+(distance as usize)<32 {qsum+=oq;ksum+=ok;}
                distance/=2;
            }
            let (qtot,ktot):(f32,f32);
            asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)qtot,v=in(reg32)qsum,options(nomem,nostack));
            asm!("shfl.sync.idx.b32 {o}, {v}, 0, 31, -1;",o=out(reg32)ktot,v=in(reg32)ksum,options(nomem,nostack));
            let qnorm=root(qtot+1e-6);let knorm=root(ktot+1e-6);
            let out=out_base.add((token*heads+head)*513);
            unroll!(4,i,{
                let j=lane+32*i;
                let wv=crate::ptx::load4(smem+((1024+j*4)*4) as u32);
                let sv=(((0.0f32+vw[i][0]*f32::from_bits(wv[0]))+vw[i][1]*f32::from_bits(wv[1]))+vw[i][2]*f32::from_bits(wv[2]))+vw[i][3]*f32::from_bits(wv[3]);
                let vc=round(sv*sigmoid(sv));
                *out.add(j)=(qv[i]/qnorm)*scale;
                *out.add(128+j)=kv[i]/knorm;
                *out.add(256+j)=vc;
                let av=ld_nc!(a_ptr.add(token*width+base+j));
                let dtv=crate::ptx::load(smem+DT_OFF+(j*4) as u32);
                let decay_raw=p.bound*sigmoid(rate*(av+dtv));
                *out.add(384+j)=if fuse_exp {exp(decay_raw)} else {decay_raw};
            });
            if lane==0 {*out.add(512)=round(sigmoid(ld_nc!(b_ptr.add(token*heads+head))));}
            token+=1;
            if token<last {
                unroll!(4,i,{
                    let j=lane+32*i;
                    qw[i][0]=qw[i][1];qw[i][1]=qw[i][2];qw[i][2]=qw[i][3];qw[i][3]=at_plane(q_ptr,old_q,token+3,j);
                    kw[i][0]=kw[i][1];kw[i][1]=kw[i][2];kw[i][2]=kw[i][3];kw[i][3]=at_plane(k_ptr,old_k,token+3,j);
                    vw[i][0]=vw[i][1];vw[i][1]=vw[i][2];vw[i][2]=vw[i][3];vw[i][3]=at_plane(v_ptr,old_v,token+3,j);
                });
            }
        }
    }
}
