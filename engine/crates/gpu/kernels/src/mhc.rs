//! Logical FP32 Sinkhorn blocks. These entries do not admit a model.
use crate::{mhc_params::*, mhc_math as math};
use core::slice;

#[no_mangle]
pub extern "ptx-kernel" fn mhc_prepare(p: Prepare) {
    // SAFETY: The caller validates Shape and supplies live, aligned, disjoint
    // buffers with Prepare's extents. The host launches phases 0, 1, and 2 on
    // one ordered stream. Phase 0 owns each token's normalized row. Phase 1
    // uses lane zero of each complete warp to write an independent coefficient. Phase 2
    // owns each token's completed coefficient row. Guards bound each slice.
    // No mutable slice overlaps another live thread's mutable slice.
    unsafe {
        let index = crate::index(); let s = p.shape;
        let w = s.wide(); let count = s.coefficients();
        match p.phase {
            0=>{if index>=s.rows as usize {return;}
                math::normalize(s,slice::from_raw_parts((p.input as *const f32).add(index*w),w),
                    slice::from_raw_parts_mut((p.normalized as *mut f32).add(index*w),w));
            },
            1=>{if index%32!=0 {return;}let index=index/32;
                if index>=s.rows as usize*count {return;}let token=index/count;let coefficient=index%count;
                let value=math::project(s,coefficient,slice::from_raw_parts(p.weights as *const f32,count*w),
                    slice::from_raw_parts(p.base as *const f32,count),slice::from_raw_parts(p.scale as *const f32,3),
                    slice::from_raw_parts((p.normalized as *const f32).add(token*w),w));
                *(p.coefficients as *mut f32).add(index)=value;
            },
            2=>{if index>=s.rows as usize {return;}
                math::activate(s,slice::from_raw_parts_mut((p.coefficients as *mut f32).add(index*count),count));
            },_=>{},
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
