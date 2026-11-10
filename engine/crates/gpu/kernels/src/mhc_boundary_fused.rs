//! Deferred post/pre boundaries. Arithmetic follows the resident primitives.
use crate::{mhc_params::Boundary,mhc_math as math};
use core::slice;

#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_project(p:Boundary) {
    // Same arithmetic and order as the previous kernel: the RMS sum and every
    // coefficient dot are ONE ascending FP32 chain over the features, starting
    // from 0.0, with the product rounded first and no fused multiply-add. The
    // previous kernel fed each chain through 32 shuffles per 32 features on
    // one SM. Here the products are staged in shared memory by the other warps
    // and ONE warp advances up to 24 independent chains at once, one chain per
    // lane, so one add instruction moves all of them. Chunk loads run ahead of
    // the chains (cp.async ring), and the weights are warmed into L2 while the
    // RMS chain runs.
    // SAFETY: The caller validates every disjoint plane and retains all owners
    // through the ordered consumer. One block owns one token and has at least
    // two whole warps (anything else traps). Every thread reaches every
    // barrier the same number of times: all loop bounds depend only on the
    // shape. Previous planes are read only when present and have the same
    // token/stream widths. Post stores end before RMS reads. Shared memory is
    // one static array, split into the weight ring, the RMS ring and one word;
    // every staged copy and every shared read stays inside its stage, and a
    // stage is rewritten only after a barrier that follows its last read.
    // Each producer copy reads inside the validated weight or residual
    // planes. Warp zero alone owns the chains and the coefficient stores; the
    // producers alone own the normalized plane. No slice spans a barrier.
    unsafe {
        const CH:usize=128;const RG:usize=24;const STRIDE:usize=CH+4;const STAGE:usize=CH+RG*STRIDE;const S:usize=3;
        const RC:usize=128;const RS:usize=3;
        const BYTES:usize=S*STAGE*4+RS*RC*4+16;
        let token=core::arch::nvptx::_block_idx_x() as usize;let shape=p.shape;
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;
        if threads<64 || threads%32!=0 {core::arch::asm!("trap;");}
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        let mut feature=thread;
        while feature<wide {
            let value=if p.previous_input==0 {crate::bf16::rounded(*(p.input as *const f32).add(token*wide+feature))}
                else {crate::mhc_boundary_math::combine(
                    slice::from_raw_parts((p.previous_input as *const f32).add(token*wide),wide),
                    slice::from_raw_parts((p.previous_coefficients as *const f32).add(token*count),count),
                    crate::bf16::rounded(*(p.input as *const f32).add(token*hidden+feature%hidden)),
                    streams,hidden,feature/hidden,feature%hidden)};
            *(p.residual as *mut f32).add(token*wide+feature)=value;feature+=threads;
        }
        let smem=crate::ptx::shared::<BYTES>();
        let rms=smem+(S*STAGE*4) as u32;let slot=rms+(RS*RC*4) as u32;
        let producer=thread>=32;let ptid=thread.wrapping_sub(32);let nprod=threads-32;
        let residual=(p.residual as *const f32).add(token*wide);let weights=p.weights as *const f32;
        let nrc=wide.div_ceil(RC);let nkc=wide.div_ceil(CH);let total=count.div_ceil(RG)*nkc;
        // Stage one dot chunk: the residual slice, then up to RG weight rows.
        let issue_dot=|q:usize| {
            let g=q/nkc;let k0=(q%nkc)*CH;let rows=(count-g*RG).min(RG);let span=(wide-k0).min(CH);
            let stage=smem+((q%S)*STAGE*4) as u32;
            let mut e=ptid;
            while e<span {crate::ptx::copy4(stage+(e*4) as u32,residual.add(k0+e).cast());e+=nprod;}
            let mut e=ptid;
            while e<rows*CH {
                let r=e/CH;let kk=e%CH;
                if kk<span {crate::ptx::copy4(stage+((CH+r*STRIDE+kk)*4) as u32,weights.add((g*RG+r)*wide+k0+kk).cast());}
                e+=nprod;
            }
        };
        let issue_rms=|c:usize| {
            let k0=c*RC;let span=(wide-k0).min(RC);let dst=rms+((c%RS)*RC*4) as u32;
            let mut e=ptid;
            while e<span {crate::ptx::copy4(dst+(e*4) as u32,residual.add(k0+e).cast());e+=nprod;}
        };
        crate::ptx::sync();
        if producer {
            for s in 0..S-1 {if s<total {issue_dot(s);}crate::ptx::commit();}
            let mut line=ptid;
            while line*32<count*wide {core::arch::asm!("prefetch.global.L2 [{a}];",a=in(reg64)weights.add(line*32));line+=nprod;}
            for s in 0..RS-1 {if s<nrc {issue_rms(s);}crate::ptx::commit();}
        }
        // The RMS sum: thread zero folds the staged squares in ascending order.
        let mut squares=0.0f32;let mut c=0;
        while c<nrc {
            crate::ptx::wait_one();crate::ptx::sync();
            if producer {if c+RS-1<nrc {issue_rms(c+RS-1);}crate::ptx::commit();}
            if thread==0 {
                let span=(wide-c*RC).min(RC);let base=rms+((c%RS)*RC*4) as u32;let full=span/8*8;let mut j=0;
                if full>0 {
                    let mut cur=[crate::ptx::load4(base),crate::ptx::load4(base+16)];j=8;
                    while j<=full {
                        let next=if j<full {[crate::ptx::load4(base+(j*4) as u32),crate::ptx::load4(base+(j*4) as u32+16)]} else {cur};
                        for i in 0..8 {let v=f32::from_bits(cur[i/4][i%4]);let square=v*v;squares+=square;}
                        cur=next;j+=8;
                    }
                    j=full;
                }
                while j<span {let v=crate::ptx::load(base+(j*4) as u32);let square=v*v;squares+=square;j+=1;}
            }
            c+=1;
        }
        if thread==0 {crate::ptx::store(slot,1.0/crate::qsa_common::root(squares/wide as f32+shape.rms_eps));}
        crate::ptx::wait();crate::ptx::sync();
        let inverse=crate::ptx::load(slot);
        if producer {
            let mut f=ptid;
            while f<wide {*(p.mixing_normalized as *mut f32).add(token*wide+f)=*residual.add(f)*inverse;f+=nprod;}
        }
        // The dots: lane r of warp zero folds coefficient row g*RG+r.
        let mut acc=0.0f32;let mut q=0;
        while q<total {
            crate::ptx::wait_one();crate::ptx::sync();
            if producer {if q+S-1<total {issue_dot(q+S-1);}crate::ptx::commit();}
            if thread<32 {
                let g=q/nkc;let kc=q%nkc;let span=(wide-kc*CH).min(CH);let rows=(count-g*RG).min(RG);
                if kc==0 {acc=0.0;}
                if lane<rows {
                    let stage=smem+((q%S)*STAGE*4) as u32;let ws=stage+((CH+lane*STRIDE)*4) as u32;
                    let full=span/8*8;let mut j=0;
                    if full>0 {
                        let mut w=[crate::ptx::load4(ws),crate::ptx::load4(ws+16)];
                        let mut x=[crate::ptx::load4(stage),crate::ptx::load4(stage+16)];j=8;
                        while j<=full {
                            let (nw,nx)=if j<full {(
                                [crate::ptx::load4(ws+(j*4) as u32),crate::ptx::load4(ws+(j*4) as u32+16)],
                                [crate::ptx::load4(stage+(j*4) as u32),crate::ptx::load4(stage+(j*4) as u32+16)])} else {(w,x)};
                            for i in 0..8 {
                                let normalized=f32::from_bits(x[i/4][i%4])*inverse;
                                let product=f32::from_bits(w[i/4][i%4])*normalized;acc+=product;
                            }
                            w=nw;x=nx;j+=8;
                        }
                        j=full;
                    }
                    while j<span {
                        let normalized=crate::ptx::load(stage+(j*4) as u32)*inverse;
                        let product=crate::ptx::load(ws+(j*4) as u32)*normalized;acc+=product;j+=1;
                    }
                    if kc==nkc-1 {
                        let coefficient=g*RG+lane;
                        let group=if coefficient<streams {0} else if coefficient<2*streams {1} else {2};
                        *(p.coefficients as *mut f32).add(token*count+coefficient)=acc * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(coefficient);
                    }
                }
            }
            q+=1;
        }
    }
}

macro_rules! boundary_finish_body {($arg:expr)=>{{let p=$arg;
    // SAFETY: The caller supplies validated, live, disjoint planes. The preceding
    // launch on the same stream completes the raw coefficient rows. One block
    // owns each token; all threads reach every barrier and complete warps reach
    // every shuffle. Up to 1024 coefficients and 32 row/inverse scratch values
    // fit the private shared allocation; larger shapes retain the scalar fold.
    // Rows and columns synchronize after each unchanged ordered sum and divide.
    // Activation ends before collapse, collapse before partials, and partials
    // before the unchanged per-lane final norm tree. Each output has one writer;
    // all slice extents and partial indices follow the validated shape. A block
    // past the row grid is a CHAIN-SHADOW tail: it writes nothing, reaches no
    // barrier, and prefetches only inside the binder-checked live weight range.
    // A matrix that fits one warp uses all 32 lanes for each shuffle. Its row
    // and column sources name initialized matrix lanes. Only valid cells and
    // gates store outputs. The block barrier publishes them before collapse.
    unsafe {
        let token=core::arch::nvptx::_block_idx_x() as usize;let shape=p.shape;
        // CHAIN-SHADOW tail: blocks past the row grid only prefetch, then return.
        crate::chain_shadow::shadow_tail!(p,shape.rows as usize);
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        const COEFFICIENT_CAPACITY:usize=1024;
        let shared=crate::ptx::shared::<{(COEFFICIENT_CAPACITY+32)*4}>();
        macro_rules! load {($index:expr)=>{crate::ptx::load(shared+(($index)*4) as u32)}}
        macro_rules! store {($index:expr,$value:expr)=>{crate::ptx::store(shared+(($index)*4) as u32,$value)}}
        if streams*streams<=32 {
            if thread<32 {
                let coefficients=(p.coefficients as *mut f32).add(token*count);
                let pre=if lane<streams {math::sigmoid(*coefficients.add(lane))+shape.eps} else {0.0};
                let post=if lane<streams {2.0*math::sigmoid(*coefficients.add(streams+lane))} else {0.0};
                let mut value=if lane<streams*streams {*coefficients.add(2*streams+lane)} else {0.0};
                let row=(lane/streams).min(streams-1);let column=lane%streams;
                macro_rules! read_lane {($source:expr)=>{f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),(lane^($source)) as u32,31))}}
                let mut maximum=f32::NEG_INFINITY;
                for source in 0..streams {maximum=maximum.max(read_lane!(row*streams+source));}
                value=crate::qsa_common::exponential(value-maximum);
                let mut sum=0.0f32;
                for source in 0..streams {sum+=read_lane!(row*streams+source);}
                value=value/sum+shape.eps;
                let mut sum=0.0f32;
                for source in 0..streams {sum+=read_lane!(source*streams+column);}
                value/=sum+shape.eps;
                for _iteration in 1..shape.iterations {
                    let mut sum=0.0f32;
                    for source in 0..streams {sum+=read_lane!(row*streams+source);}
                    value/=sum+shape.eps;
                    let mut sum=0.0f32;
                    for source in 0..streams {sum+=read_lane!(source*streams+column);}
                    value/=sum+shape.eps;
                }
                if lane<streams {*coefficients.add(lane)=pre;*coefficients.add(streams+lane)=post;}
                if lane<streams*streams {*coefficients.add(2*streams+lane)=value;}
            }
        } else if count<=COEFFICIENT_CAPACITY {
            let mut coefficient=thread;
            while coefficient<count {store!(coefficient,*(p.coefficients as *const f32).add(token*count+coefficient));coefficient+=threads;}
            core::arch::nvptx::_syncthreads();
            if thread<streams {
                store!(thread,math::sigmoid(load!(thread))+shape.eps);
                store!(streams+thread,2.0*math::sigmoid(load!(streams+thread)));
                let mut maximum=f32::NEG_INFINITY;
                for column in 0..streams {maximum=maximum.max(load!(2*streams+thread*streams+column));}
                store!(COEFFICIENT_CAPACITY+thread,maximum);
            }
            core::arch::nvptx::_syncthreads();
            let mut cell=thread;while cell<streams*streams {let row=cell/streams;let value=load!(2*streams+cell)-load!(COEFFICIENT_CAPACITY+row);
                store!(2*streams+cell,crate::qsa_common::exponential(value));cell+=threads;}
            core::arch::nvptx::_syncthreads();
            if thread<streams {let mut sum=0.0f32;for column in 0..streams {sum+=load!(2*streams+thread*streams+column);}store!(COEFFICIENT_CAPACITY+thread,sum);}
            core::arch::nvptx::_syncthreads();
            let mut cell=thread;while cell<streams*streams {let row=cell/streams;store!(2*streams+cell,load!(2*streams+cell)/load!(COEFFICIENT_CAPACITY+row)+shape.eps);cell+=threads;}
            core::arch::nvptx::_syncthreads();
            if thread<streams {let mut sum=0.0f32;for source in 0..streams {sum+=load!(2*streams+source*streams+thread);}store!(COEFFICIENT_CAPACITY+thread,sum+shape.eps);}
            core::arch::nvptx::_syncthreads();
            let mut cell=thread;while cell<streams*streams {let column=cell%streams;store!(2*streams+cell,load!(2*streams+cell)/load!(COEFFICIENT_CAPACITY+column));cell+=threads;}
            core::arch::nvptx::_syncthreads();
            for _iteration in 1..shape.iterations {
                if thread<streams {let mut sum=0.0f32;for column in 0..streams {sum+=load!(2*streams+thread*streams+column);}store!(COEFFICIENT_CAPACITY+thread,sum+shape.eps);}
                core::arch::nvptx::_syncthreads();
                let mut cell=thread;while cell<streams*streams {let row=cell/streams;store!(2*streams+cell,load!(2*streams+cell)/load!(COEFFICIENT_CAPACITY+row));cell+=threads;}
                core::arch::nvptx::_syncthreads();
                if thread<streams {let mut sum=0.0f32;for source in 0..streams {sum+=load!(2*streams+source*streams+thread);}store!(COEFFICIENT_CAPACITY+thread,sum+shape.eps);}
                core::arch::nvptx::_syncthreads();
                let mut cell=thread;while cell<streams*streams {let column=cell%streams;store!(2*streams+cell,load!(2*streams+cell)/load!(COEFFICIENT_CAPACITY+column));cell+=threads;}
                core::arch::nvptx::_syncthreads();
            }
            let mut coefficient=thread;
            while coefficient<count {*(p.coefficients as *mut f32).add(token*count+coefficient)=load!(coefficient);coefficient+=threads;}
        } else if thread==0 {math::activate(shape,slice::from_raw_parts_mut((p.coefficients as *mut f32).add(token*count),count));}
        core::arch::nvptx::_syncthreads();
        let mut feature=thread;
        while feature<hidden {
            *(p.collapsed as *mut f32).add(token*hidden+feature)=crate::mhc_boundary_math::collapse(
                slice::from_raw_parts((p.residual as *const f32).add(token*wide),wide),
                slice::from_raw_parts((p.coefficients as *const f32).add(token*count),streams),streams,hidden,feature);
            feature+=threads;
        }
        core::arch::nvptx::_syncthreads();
        let chunks=hidden.div_ceil(128);let mut chunk=thread/32;
        while chunk<chunks {
            let mut squares=0.0f32;let mut feature=chunk*128+lane;
            while feature<(chunk*128+128).min(hidden) {
                let value=*(p.collapsed as *const f32).add(token*hidden+feature);squares+=value*value;feature+=32;
            }
            let mut distance=16;
            while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
            if lane==0 {*(p.norm_partials as *mut f32).add(token*chunks+chunk)=squares;}
            chunk+=threads/32;
        }
        core::arch::nvptx::_syncthreads();
        if thread<32 {
            let mut squares=0.0f32;let mut chunk=lane;
            while chunk<chunks {squares+=*(p.norm_partials as *const f32).add(token*chunks+chunk);chunk+=32;}
            let mut distance=16;
            while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
            store!(COEFFICIENT_CAPACITY+lane,1.0/crate::layer::sqrt(squares/hidden as f32+p.norm_epsilon.abs()));
        }
        core::arch::nvptx::_syncthreads();
        let inverse=load!(COEFFICIENT_CAPACITY+lane);
        let mut feature=thread;
        while feature<hidden {
            let normalized=crate::layer::round_bf16(*(p.collapsed as *const f32).add(token*hidden+feature)*inverse*1.0);
            let value=normalized * *(p.norm_weight as *const f32).add(feature);
            *(p.output as *mut f32).add(token*hidden+feature)=if p.norm_bf16==0 {value} else {crate::bf16::rounded(value)};
            feature+=threads;
        }
    }

}};}
#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_finish(p:Boundary) {boundary_finish_body!(p)}
/// Replay the unchanged finish body with the original grid and checked arguments.
#[inline(always)]
pub(crate) fn resident_mhc_boundary_finish(p:Boundary) {boundary_finish_body!(p)}
