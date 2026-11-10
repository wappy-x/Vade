//! Logical FP32 Sinkhorn blocks. These entries do not admit a model.
use crate::{mhc_params::*, mhc_math as math};
use core::slice;

#[no_mangle]
pub extern "ptx-kernel" fn mhc_prepare(p: Prepare) {resident_mhc_prepare(p)}
#[inline(always)]
pub(crate) fn resident_mhc_prepare(p:Prepare) {
    // SAFETY: The caller validates Shape and supplies live, aligned, disjoint
    // buffers with Prepare's extents. Exactly one block owns each token and
    // all threads reach every block barrier. Full warps load independent
    // products, then broadcast them to one complete ascending FP32 fold.
    // All shuffle sources are participating lanes. Tail lanes load zero but
    // nonexistent features never enter a fold. Barriers publish the inverse,
    // normalized features, complete dots and pre gates in that order.
    // Warp zero owns the matrix slice, disjoint from other warps' immutable
    // pre gates and distinct collapsed features. No slice spans a barrier.
    unsafe {
        let token=core::arch::nvptx::_block_idx_x() as usize;let s=p.shape;
        if token>=s.rows as usize {return;}
        let thread=core::arch::nvptx::_thread_idx_x() as usize;
        let threads=core::arch::nvptx::_block_dim_x() as usize;
        let w=s.wide();let count=s.coefficients();let lane=thread%32;let full=w/32*32;
        if thread<32 {
            let mut squares=0.0f32;let mut first=0;
            while first<full {
                let value=*(p.input as *const f32).add(token*w+first+lane);let product=value*value;
                for source in 0..32 {
                    squares+=f32::from_bits(crate::shuffle_xor(u32::MAX,product.to_bits(),(lane^source) as u32,31));
                }
                first+=32;
            }
            if first<w {
                let value=if first+lane<w {*(p.input as *const f32).add(token*w+first+lane)} else {0.0};
                let product=value*value;
                for source in 0..32 {
                    let term=f32::from_bits(crate::shuffle_xor(u32::MAX,product.to_bits(),(lane^source) as u32,31));
                    if first+source<w {squares+=term;}
                }
            }
            if lane==0 {*(p.coefficients as *mut f32).add(token*count)=1.0/crate::qsa_common::root(squares/w as f32+s.rms_eps);}
        }
        core::arch::nvptx::_syncthreads();
        let inverse=*(p.coefficients as *const f32).add(token*count);
        let mut feature=thread;
        while feature<w {
            *(p.normalized as *mut f32).add(token*w+feature)=*(p.input as *const f32).add(token*w+feature)*inverse;
            feature+=threads;
        }
        core::arch::nvptx::_syncthreads();
        let mut coefficient=thread/32;
        while coefficient<count {
            let mut total=0.0f32;let mut first=0;
            while first<full {
                let product=*(p.weights as *const f32).add(coefficient*w+first+lane)*
                    *(p.normalized as *const f32).add(token*w+first+lane);
                for source in 0..32 {
                    total+=f32::from_bits(crate::shuffle_xor(u32::MAX,product.to_bits(),(lane^source) as u32,31));
                }
                first+=32;
            }
            if first<w {
                let product=if first+lane<w {
                    *(p.weights as *const f32).add(coefficient*w+first+lane)*
                    *(p.normalized as *const f32).add(token*w+first+lane)
                } else {0.0};
                for source in 0..32 {
                    let term=f32::from_bits(crate::shuffle_xor(u32::MAX,product.to_bits(),(lane^source) as u32,31));
                    if first+source<w {total+=term;}
                }
            }
            if lane==0 {
                let streams=s.streams as usize;
                let group=if coefficient<streams {0} else if coefficient<2*streams {1} else {2};
                *(p.coefficients as *mut f32).add(token*count+coefficient)=total * *(p.scale as *const f32).add(group)+*(p.base as *const f32).add(coefficient);
            }
            coefficient+=threads/32;
        }
        core::arch::nvptx::_syncthreads();
        if thread==0 {
            math::gates(s,slice::from_raw_parts_mut((p.coefficients as *mut f32).add(token*count),2*s.streams as usize));
        }
        core::arch::nvptx::_syncthreads();
        let streams=s.streams as usize;let hidden=s.hidden as usize;
        if thread<32 {
            if streams*streams<=32 {
                let cell=if lane<streams*streams {lane} else {0};let row=cell/streams;let column=cell%streams;
                let mut value=*(p.coefficients as *const f32).add(token*count+2*streams+cell);
                let mut maximum=f32::NEG_INFINITY;
                for source in 0..streams {
                    let term=f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),(lane^(row*streams+source)) as u32,31));
                    maximum=maximum.max(term);
                }
                value=crate::qsa_common::exponential(value-maximum);
                let mut total=0.0f32;
                for source in 0..streams {
                    total+=f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),(lane^(row*streams+source)) as u32,31));
                }
                value=value/total+s.eps;
                for iteration in 0..s.iterations {
                    if iteration>0 {
                        let mut total=0.0f32;
                        for source in 0..streams {
                            total+=f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),(lane^(row*streams+source)) as u32,31));
                        }
                        value/=total+s.eps;
                    }
                    let mut total=0.0f32;
                    for source in 0..streams {
                        total+=f32::from_bits(crate::shuffle_xor(u32::MAX,value.to_bits(),(lane^(source*streams+column)) as u32,31));
                    }
                    value/=total+s.eps;
                }
                if lane<streams*streams {*(p.coefficients as *mut f32).add(token*count+2*streams+lane)=value;}
            } else if thread==0 {
                math::sinkhorn(s,slice::from_raw_parts_mut((p.coefficients as *mut f32).add(token*count+2*streams),streams*streams));
            }
        } else {
            let mut feature=thread-32;
            while feature<hidden {
                *(p.collapsed as *mut f32).add(token*hidden+feature)=crate::mhc_boundary_math::collapse(
                    slice::from_raw_parts((p.input as *const f32).add(token*w),w),
                    slice::from_raw_parts((p.coefficients as *const f32).add(token*count),streams),streams,hidden,feature);
                feature+=threads-32;
            }
        }
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn mhc_collapse(p: Collapse) {
    // SAFETY: The caller supplies validated, live, aligned, disjoint buffers
    // with Collapse's extents. This guarded thread owns output[i] and reads
    // only the corresponding token's raw streams and coefficient row.
    unsafe {
        let i = crate::index(); let h = p.hidden as usize; let n = p.streams as usize;
        if i >= p.rows as usize * h { return; }
        let token = i / h; let count = n * (n + 2);
        *(p.output as *mut f32).add(i) = math::collapse(
            slice::from_raw_parts((p.input as *const f32).add(token * n * h), n * h),
            slice::from_raw_parts((p.coefficients as *const f32).add(token * count), n), n, h, i % h);
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn mhc_combine(p: Combine) {
    // SAFETY: The caller supplies validated, live, aligned, disjoint buffers
    // with Combine's extents. This guarded thread owns output[i]. Its input,
    // coefficient and block reads stay within the corresponding token.
    unsafe {
        let i = crate::index(); let h = p.hidden as usize; let n = p.streams as usize; let wide = n * h;
        if i >= p.rows as usize * wide { return; }
        let token = i / wide; let count = n * (n + 2);
        *(p.output as *mut f32).add(i) = math::combine(
            slice::from_raw_parts((p.input as *const f32).add(token * wide), wide),
            slice::from_raw_parts((p.coefficients as *const f32).add(token * count), count),
            *(p.block as *const f32).add(token * h + i % h), n, h, i % wide / h, i % h);
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn mhc_head(p: Head) {
    // SAFETY: The caller supplies validated, live, aligned, disjoint buffers
    // with Head's extents. This guarded thread owns output[i] and reads only
    // the raw streams for token i/hidden. The sum visits every stream in order.
    unsafe {
        let i = crate::index(); let h = p.hidden as usize; let n = p.streams as usize;
        if i >= p.rows as usize * h { return; }
        *(p.output as *mut f32).add(i) = math::head(
            slice::from_raw_parts((p.input as *const f32).add(i / h * n * h), n * h), n, h, i % h);
    }
}
