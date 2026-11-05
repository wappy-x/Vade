//! Deferred post/pre boundaries. Arithmetic follows the resident primitives.
use crate::{mhc_params::Boundary,mhc_math as math};
use core::slice;

#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_project(p:Boundary) {
    // SAFETY: The caller validates every disjoint plane and retains all owners
    // through the ordered consumer. One block owns one token. Every thread
    // reaches each barrier; full warps participate in all shuffles. Previous
    // planes are read only when present and have the same token/stream widths.
    // Post stores end before RMS reads; normalization ends before complete
    // coefficient dots. No slice spans a barrier or aliases a writable plane.
    unsafe {
        let token=core::arch::nvptx::_block_idx_x() as usize;let shape=p.shape;
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;
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
        core::arch::nvptx::_syncthreads();
        if thread<32 {
            let mut squares=0.0f32;let mut first=0;let full=wide/32*32;
            while first<full {
                let value=*(p.residual as *const f32).add(token*wide+first+lane);let product=value*value;
                for source in 0..32 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,product.to_bits(),(lane^source) as u32,31));}
                first+=32;
            }
            if first<wide {
                let value=if first+lane<wide {*(p.residual as *const f32).add(token*wide+first+lane)} else {0.0};let product=value*value;
                for source in 0..32 {
                    let term=f32::from_bits(crate::shuffle_xor(u32::MAX,product.to_bits(),(lane^source) as u32,31));
                    if first+source<wide {squares+=term;}
                }
            }
            if lane==0 {*(p.coefficients as *mut f32).add(token*count)=1.0/crate::qsa_common::root(squares/wide as f32+shape.rms_eps);}
        }
        core::arch::nvptx::_syncthreads();
        let inverse=*(p.coefficients as *const f32).add(token*count);let mut feature=thread;
        while feature<wide {
            *(p.mixing_normalized as *mut f32).add(token*wide+feature)=*(p.residual as *const f32).add(token*wide+feature)*inverse;feature+=threads;
        }
        core::arch::nvptx::_syncthreads();
        let mut coefficient=thread/32;
        while coefficient<count {
            let mut total=0.0f32;let mut first=0;let full=wide/32*32;
            while first<full {
                let product=*(p.weights as *const f32).add(coefficient*wide+first+lane)*
                    *(p.mixing_normalized as *const f32).add(token*wide+first+lane);
                for source in 0..32 {total+=f32::from_bits(crate::shuffle_xor(u32::MAX,product.to_bits(),(lane^source) as u32,31));}
                first+=32;
            }
            if first<wide {
                let product=if first+lane<wide {*(p.weights as *const f32).add(coefficient*wide+first+lane)*
                    *(p.mixing_normalized as *const f32).add(token*wide+first+lane)} else {0.0};
                for source in 0..32 {
                    let term=f32::from_bits(crate::shuffle_xor(u32::MAX,product.to_bits(),(lane^source) as u32,31));
                    if first+source<wide {total+=term;}
                }
            }
            if lane==0 {
                let group=if coefficient<streams {0} else if coefficient<2*streams {1} else {2};
                *(p.coefficients as *mut f32).add(token*count+coefficient)=total * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(coefficient);
            }
            coefficient+=threads/32;
        }
    }
}

#[no_mangle]
pub extern "ptx-kernel" fn mhc_boundary_finish(p:Boundary) {
    // SAFETY: The caller supplies validated, live, disjoint planes. The preceding
    // launch on the same stream completes the raw coefficient rows. One block
    // owns each token; all threads reach every barrier and complete warps reach
    // every shuffle. Activation ends before collapse, collapse before partials,
    // and partials before the unchanged final norm tree. Each output has one
    // writer; all slice extents and partial indices follow the validated shape.
    unsafe {
        let token=core::arch::nvptx::_block_idx_x() as usize;let shape=p.shape;
        if token>=shape.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;let lane=thread%32;
        let wide=shape.wide();let count=shape.coefficients();let streams=shape.streams as usize;let hidden=shape.hidden as usize;
        if thread==0 {math::activate(shape,slice::from_raw_parts_mut((p.coefficients as *mut f32).add(token*count),count));}
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
        let mut squares=0.0f32;let mut chunk=lane;
        while chunk<chunks {squares+=*(p.norm_partials as *const f32).add(token*chunks+chunk);chunk+=32;}
        let mut distance=16;
        while distance!=0 {squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,squares.to_bits(),distance,31));distance/=2;}
        let inverse=1.0/crate::layer::sqrt(squares/hidden as f32+p.norm_epsilon.abs());
        let mut feature=thread;
        while feature<hidden {
            let normalized=crate::layer::round_bf16(*(p.collapsed as *const f32).add(token*hidden+feature)*inverse*1.0);
            let value=normalized * *(p.norm_weight as *const f32).add(feature);
            *(p.output as *mut f32).add(token*hidden+feature)=if p.norm_bf16==0 {value} else {crate::bf16::rounded(value)};
            feature+=threads;
        }
    }
}
