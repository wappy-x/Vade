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
