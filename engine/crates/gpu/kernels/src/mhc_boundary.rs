//! BF16 output boundaries. Storage uses widened, row-major FP32 lanes.
use crate::{mhc_params::{Collapse,Combine},mhc_boundary_math as math};
use core::slice;

#[no_mangle]
pub extern "ptx-kernel" fn bf16_lanes_round(p: math::LaneCast) {resident_bf16_lanes_round(p)}
#[inline(always)]
pub(crate) fn resident_bf16_lanes_round(p:math::LaneCast) {
    // SAFETY: The resident caller checks exact, disjoint FP32 spans and a
    // positive ABI-sized word count. The guard bounds both accesses. The caller
    // keeps the allocation owners live until the ordered stream completes.
    unsafe {
        let i=crate::index();if i>=p.words as usize {return;}
        *(p.output as *mut f32).add(i)=crate::bf16::rounded(*(p.input as *const f32).add(i));
    }
}

#[no_mangle]
pub extern "ptx-kernel" fn mhc_bf16_collapse(p: Collapse) {resident_mhc_bf16_collapse(p)}
#[inline(always)]
pub(crate) fn resident_mhc_bf16_collapse(p:Collapse) {
    // SAFETY: The resident caller checks positive dimensions and exact disjoint
    // FP32 extents. Each guarded thread writes one output. All token and feature
    // offsets below stay within those extents. Owners survive stream completion.
    unsafe {
        let i=crate::index();let h=p.hidden as usize;let n=p.streams as usize;
        if i>=p.rows as usize*h {return;}
        let token=i/h;let count=n*(n+2);
        *(p.output as *mut f32).add(i)=math::collapse(
            slice::from_raw_parts((p.input as *const f32).add(token*n*h),n*h),
            slice::from_raw_parts((p.coefficients as *const f32).add(token*count),n),n,h,i%h);
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn mhc_bf16_combine(p: Combine) {resident_mhc_bf16_combine(p)}
#[inline(always)]
pub(crate) fn resident_mhc_bf16_combine(p:Combine) {
    // SAFETY: The resident caller checks positive dimensions and exact disjoint
    // FP32 extents. Each guarded thread writes one output. Its raw input,
    // coefficients and block reads stay in that token. Owners survive completion.
    unsafe {
        let i=crate::index();let h=p.hidden as usize;let n=p.streams as usize;let wide=n*h;
        if i>=p.rows as usize*wide {return;}
        let token=i/wide;let count=n*(n+2);
        *(p.output as *mut f32).add(i)=math::combine(
            slice::from_raw_parts((p.input as *const f32).add(token*wide),wide),
            slice::from_raw_parts((p.coefficients as *const f32).add(token*count),count),
            *(p.block as *const f32).add(token*h+i%h),n,h,i%wide/h,i%h);
    }
}
