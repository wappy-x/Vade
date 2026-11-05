//! One thread per FP32 output row. Keep the scalar reference operation order.
#[no_mangle]
pub extern "ptx-kernel" fn project_f32_ordered(weights: *const f32,input: *const f32,output: *mut f32,rows: u32,cols: u32) {resident_project_f32_ordered(weights,input,output,rows,cols)}
#[inline(always)]
pub(crate) fn resident_project_f32_ordered(weights:*const f32,input:*const f32,output:*mut f32,rows:u32,cols:u32) {
    // SAFETY: The backend checks contiguous FP32 extents, context, alignment,
    // address overflow, and disjoint output storage before this launch.
    // The owners stay live through completion. Each valid thread owns one row.
    // The row check bounds every access. Shape checks bound row offset math.
    unsafe {
        let row=crate::index();
        if row<rows as usize {
            *output.add(row)=crate::serial_projection::plain_row!(weights.add(row*cols as usize),input,cols as usize);
        }
    }
}
