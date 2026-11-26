//! Deferred post/pre boundaries. Arithmetic follows the resident primitives.
use crate::{mhc_params::Boundary,mhc_math as math};
use core::slice;

#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_project(p:Boundary) {
    // SAFETY: The caller validates disjoint planes and retains their owners.
    // A block owns one token. Each thread writes distinct residual and
    // normalized words. All threads reach the barriers before dependent reads.
    // Warp zero publishes the inverse in one shared word. Each complete warp
    // owns distinct coefficient rows. All lanes reach every full-mask shuffle,
    // including for partial strips. Only lane zero writes each coefficient.
    // Previous planes are read only when present, within their checked extents.
    unsafe {
        let token=core::arch::nvptx::_block_idx_x() as usize;let shape=p.shape;
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;
        if threads<64 || threads%32!=0 {core::arch::asm!("trap;");}
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        let residual=(p.residual as *mut f32).add(token*wide);let mut feature=thread;
        while feature<wide {
            let value=if p.previous_input==0 {crate::bf16::rounded(*(p.input as *const f32).add(token*wide+feature))}
                else {crate::mhc_boundary_math::combine(
                    slice::from_raw_parts((p.previous_input as *const f32).add(token*wide),wide),
                    slice::from_raw_parts((p.previous_coefficients as *const f32).add(token*count),count),
                    crate::bf16::rounded(*(p.input as *const f32).add(token*hidden+feature%hidden)),
                    streams,hidden,feature/hidden,feature%hidden)};
            *residual.add(feature)=value;feature+=threads;
        }
        let shared=crate::ptx::shared::<4>();
        core::arch::nvptx::_syncthreads();
        if thread<32 {
            let squares=crate::tree_reduction::warp_fold!(wide,lane,|column|{let value=*residual.add(column);value*value});
            if lane==0 {crate::ptx::store(shared,1.0/crate::qsa_common::root(squares/wide as f32+shape.rms_eps));}
        }
        core::arch::nvptx::_syncthreads();
        let inverse=crate::ptx::load(shared);let mut feature=thread;
        while feature<wide {*(p.mixing_normalized as *mut f32).add(token*wide+feature)=*residual.add(feature)*inverse;feature+=threads;}
        let mut coefficient=thread/32;
        while coefficient<count {
            let row=(p.weights as *const f32).add(coefficient*wide);
            let total=crate::tree_reduction::warp_fold!(wide,lane,|column|{let normalized=*residual.add(column)*inverse;*row.add(column)*normalized});
            if lane==0 {
                let group=if coefficient<streams {0} else if coefficient<2*streams {1} else {2};
                *(p.coefficients as *mut f32).add(token*count+coefficient)=total * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(coefficient);
            }
            coefficient+=threads/32;
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
    // With at most four full 128-feature chunks per warp, a thread retains its
    // four ascending features of each chunk in sixteen registers. That warp
    // produces the same norm partial without rereading collapsed storage. All
    // threads publish partials before warp zero joins them; the final barrier
    // publishes the inverse before the held values produce final outputs.
    {
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
                if streams==4 {
                    // The same four reads and ascending sums with the row/column masks fixed.
                    // Complete valid matrix lanes 0..15 only read initialized valid lanes.
                    macro_rules! read_mask {($mask:expr)=>{f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),$mask,31))}}
                    let along=[(column^0) as u32,(column^1) as u32,(column^2) as u32,(column^3) as u32];
                    let down=[((row^0)*4) as u32,((row^1)*4) as u32,((row^2)*4) as u32,((row^3)*4) as u32];
                    let mut maximum=f32::NEG_INFINITY;
                    crate::strip_warp::unroll!(4,source,{maximum=maximum.max(read_mask!(along[source]));});
                    value=crate::qsa_common::exponential(value-maximum);
                    let mut sum=0.0f32;
                    crate::strip_warp::unroll!(4,source,{sum+=read_mask!(along[source]);});
                    value=value/sum+shape.eps;
                    let mut sum=0.0f32;
                    crate::strip_warp::unroll!(4,source,{sum+=read_mask!(down[source]);});
                    value/=sum+shape.eps;
                    for _iteration in 1..shape.iterations {
                        let mut sum=0.0f32;crate::strip_warp::unroll!(4,source,{sum+=read_mask!(along[source]);});
                        value/=sum+shape.eps;
                        let mut sum=0.0f32;crate::strip_warp::unroll!(4,source,{sum+=read_mask!(down[source]);});
                        value/=sum+shape.eps;
                    }
                } else {
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
        let warps=threads/32;let chunks=hidden.div_ceil(128);
        if streams<=4 && hidden%128==0 && chunks<=warps*4 {
            // Assign a norm chunk to a warp. Each lane keeps exactly the four
            // features the original norm loop reads, in their original order.
            let warp=thread/32;let mut pre=[0.0f32;4];
            crate::strip_warp::unroll!(4,src,{if src<streams {pre[src]=*(p.coefficients as *const f32).add(token*count+src);}});
            let mut held=[0.0f32;16];
            crate::strip_warp::unroll!(4,wave,{let chunk=warp+wave*warps;if chunk<chunks {
                let mut x=[[0.0f32;4];4];
                crate::strip_warp::unroll!(4,src,{if src<streams {crate::strip_warp::unroll!(4,e,{
                    x[src][e]=*(p.residual as *const f32).add(token*wide+src*hidden+chunk*128+lane+32*e);
                });}});
                let mut value=[0.0f32;4];
                crate::strip_warp::unroll!(4,src,{if src<streams {crate::strip_warp::unroll!(4,e,{
                    crate::strip_warp::add!(value[e],crate::strip_warp::mul!(pre[src],x[src][e]));
                });}});
                let mut squares=0.0f32;
                crate::strip_warp::unroll!(4,e,{
                    let value=crate::bf16::rounded(value[e]);held[wave*4+e]=value;
                    *(p.collapsed as *mut f32).add(token*hidden+chunk*128+lane+32*e)=value;
                    crate::strip_warp::add!(squares,crate::strip_warp::mul!(value,value));
                });
                let mut distance=16;
                while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
                if lane==0 {*(p.norm_partials as *mut f32).add(token*chunks+chunk)=squares;}
            }});
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
            crate::strip_warp::unroll!(4,wave,{let chunk=warp+wave*warps;if chunk<chunks {
                crate::strip_warp::unroll!(4,e,{let feature=chunk*128+lane+32*e;
                    let normalized=crate::layer::round_bf16(held[wave*4+e]*inverse*1.0);
                    let value=normalized * *(p.norm_weight as *const f32).add(feature);
                    *(p.output as *mut f32).add(token*hidden+feature)=if p.norm_bf16==0 {value} else {crate::bf16::rounded(value)};
                });
            }});
            return;
        }
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
pub extern "ptx-kernel" fn mhc_boundary_finish(p:Boundary) {
    // SAFETY: The contract is the comment at the head of boundary_finish_body. The launch binder checks the arguments.
    unsafe {boundary_finish_body!(p)}
}
/// Replay the unchanged finish body with the original grid and checked arguments.
#[inline(always)]
pub(crate) fn resident_mhc_boundary_finish(p:Boundary) {
    // SAFETY: The contract is the comment at the head of boundary_finish_body. The resident program checks the arguments before the replay.
    unsafe {boundary_finish_body!(p)}
}

/// One block a token (256 threads, 8 warps): the token's whole prompt boundary in ONE launch. It was three launches
/// (`mhc_residual_rows`, `mhc_project_tile`, `mhc_boundary_finish`) that each streamed the group's residual plane from
/// memory (67 MB at 1,024 rows, larger than the L2): six reads and a write of it a boundary. Here a block makes its token's
/// residual row (64 KB), folds it and collapses it while the row is still in the caches.
/// Arithmetic, unchanged word for word: the residual is `mhc_boundary_project`'s first loop; the RMS strip sums and the
/// coefficient strips are `mhc_project_tile`'s (lane = strip, 16-byte loads, ascending features from +0.0, `warp_root`),
/// with the 24 rows split over the 8 warps, 3 a warp, and every warp folding the same RMS tree for its own inverse; the
/// rest is the finish body. A zero `mixing_normalized` address means that no caller reads the normalized plane: it is not stored.
#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_block(p:Boundary) {
    // SAFETY: The caller validates and retains disjoint planes (the binder of the three launches this replaces). A block
    // owns one token: every word it reads or writes is in that token's rows of the row-major planes, the shared parameter
    // rows, or its private shared words. In the four-stream producer each thread owns one feature in all four streams,
    // caches only read-only coefficients, and sums sources in ascending order. Other shapes keep one output per thread.
    // Each thread writes distinct residual words; all threads reach the barrier before
    // any thread reads them. Every warp is complete (256 threads), so all lanes reach every full-mask shuffle. Warp zero
    // alone stores normalized words (lane L owns the 8-feature groups L, L + 32, ...; the plane has the residual's shape,
    // so its rows start at 16-byte addresses as the residual's do); lane zero of a warp alone stores that warp's
    // coefficient rows. All threads reach the second barrier before the finish body reads the coefficients.
    unsafe {
        let token=core::arch::nvptx::_block_idx_x() as usize;let shape=p.shape;
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;let warp=thread/32;
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        const ROWS_A_WARP:usize=3;
        if threads!=256 || count==0 || count>8*ROWS_A_WARP || wide%4!=0 {core::arch::asm!("trap;");}
        use crate::strip_warp::{ld4,mul,add,unroll};
        let residual=(p.residual as *mut f32).add(token*wide);
        if p.previous_input!=0 && streams==4 {
            let coefficients=(p.previous_coefficients as *const f32).add(token*count);
            let previous=(p.previous_input as *const f32).add(token*wide);
            let mut comb=[0.0f32;16];let mut post=[0.0f32;4];
            unroll!(4,src,{unroll!(4,dst,{comb[src*4+dst]=crate::bf16::rounded(*coefficients.add(8+src*4+dst));});});
            unroll!(4,dst,{post[dst]=crate::bf16::rounded(*coefficients.add(4+dst));});
            let mut feature=thread;
            while feature<hidden {
                let mut sums=[0.0f32;4];
                unroll!(4,src,{
                    let value=*previous.add(src*hidden+feature);
                    unroll!(4,dst,{add!(sums[dst],mul!(comb[src*4+dst],value));});
                });
                let block=crate::bf16::rounded(*(p.input as *const f32).add(token*hidden+feature));
                unroll!(4,dst,{
                    let product=crate::bf16::rounded(mul!(post[dst],block));
                    *residual.add(dst*hidden+feature)=crate::bf16::rounded(crate::bf16::rounded(sums[dst])+product);
                });
                feature+=threads;
            }
        } else {
            let mut feature=thread;
            while feature<wide {
                let value=if p.previous_input==0 {crate::bf16::rounded(*(p.input as *const f32).add(token*wide+feature))}
                    else {crate::mhc_boundary_math::combine(
                        slice::from_raw_parts((p.previous_input as *const f32).add(token*wide),wide),
                        slice::from_raw_parts((p.previous_coefficients as *const f32).add(token*count),count),
                        crate::bf16::rounded(*(p.input as *const f32).add(token*hidden+feature%hidden)),
                        streams,hidden,feature/hidden,feature%hidden)};
                *residual.add(feature)=value;feature+=threads;
            }
        }
        core::arch::nvptx::_syncthreads();
        let residual=residual as *const f32;
        let mut squares=0.0f32;let mut col=lane*8;
        while col<wide {
            let halves=if col+4<wide {2} else {1};
            unroll!(2,h,{if h<halves {let x=ld4!(residual.add(col+4*h));unroll!(4,e,{add!(squares,mul!(x[e],x[e]));});}});
            col+=256;
        }
        let inverse=1.0/crate::qsa_common::root(crate::reduction::warp_root(squares)/wide as f32+shape.rms_eps);
        let weights=p.weights as *const f32;let normalized=(p.mixing_normalized as *mut f32).wrapping_add(token*wide);
        let store=warp==0 && p.mixing_normalized!=0;
        let mut acc=[0.0f32;ROWS_A_WARP];let mut rowp=[weights;ROWS_A_WARP];
        unroll!(3,r,{rowp[r]=weights.add((warp*ROWS_A_WARP+r).min(count-1)*wide);});
        let mut col=lane*8;
        while col<wide {
            let halves=if col+4<wide {2} else {1};
            // Issue both independent half-strips before consuming either. Each coefficient still adds
            // features in the same order. Only the placement of independent global loads changes.
            let mut x=[[0.0f32;4];2];let mut w=[[[0.0f32;4];ROWS_A_WARP];2];
            unroll!(2,h,{if h<halves {
                x[h]=ld4!(residual.add(col+4*h));
                unroll!(3,r,{w[h][r]=ld4!(rowp[r].add(col+4*h));});
            }});
            unroll!(2,h,{if h<halves {
                let n=[mul!(x[h][0],inverse),mul!(x[h][1],inverse),mul!(x[h][2],inverse),mul!(x[h][3],inverse)];
                if store {core::arch::asm!("st.global.v4.u32 [{at}], {{{a},{b},{c},{d}}};",a=in(reg32)n[0],b=in(reg32)n[1],c=in(reg32)n[2],d=in(reg32)n[3],at=in(reg64)normalized.wrapping_add(col+4*h),options(nostack));}
                unroll!(3,r,{unroll!(4,e,{add!(acc[r],mul!(w[h][r][e],n[e]));});});
            }});
            col+=256;
        }
        unroll!(3,r,{
            let coefficient=warp*ROWS_A_WARP+r;
            if coefficient<count {
                let total=crate::reduction::warp_root(acc[r]);
                if lane==0 {
                    let group=if coefficient<streams {0} else if coefficient<2*streams {1} else {2};
                    *(p.coefficients as *mut f32).add(token*count+coefficient)=total * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(coefficient);
                }
            }
        });
        core::arch::nvptx::_syncthreads();
        boundary_finish_body!(p)
    }
}
