//! One thread per FP32 output row. Keep the scalar reference operation order.
#[no_mangle]
pub extern "ptx-kernel" fn project_f32_ordered(weights: *const f32,input: *const f32,output: *mut f32,rows: u32,cols: u32) {
    // SAFETY: The backend checks contiguous FP32 extents, context, alignment,
    // address overflow, and disjoint output storage before this launch.
    // The owners stay live through completion. Each valid thread owns one row.
    // The row check bounds every access. Shape checks bound row offset math.
    unsafe {
        let row=crate::index();
        if row<rows as usize {
            let mut sum=0.0f32;
            for j in 0..cols as usize {
                let a=*weights.add(row*cols as usize+j);let b=*input.add(j);let product: f32;
                core::arch::asm!("mul.rn.f32 {p}, {a}, {b};",p=out(reg32)product,a=in(reg32)a,b=in(reg32)b);
                core::arch::asm!("add.rn.f32 {s}, {s}, {p};",s=inout(reg32)sum,p=in(reg32)product);
            }
            *output.add(row)=sum;
        }
    }
}
