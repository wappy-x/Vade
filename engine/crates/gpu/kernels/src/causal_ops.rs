//! Checked single-token operations. Each thread owns one complete output unit.
use crate::causal_ops::{Arguments,unit};
#[no_mangle]
pub extern "ptx-kernel" fn causal_operation(p:Arguments) {resident_causal_operation(p)}
#[inline(always)]
pub(crate) fn resident_causal_operation(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and the exact same-context
    // allocation extents. Every input and weight is immutable through completion.
    // The host zeroes and completes the private output before this launch.
    // It overlaps no input. Each valid thread
    // creates only its own disjoint unit slice. Empty inputs use aligned dangling
    // pointers and zero lengths. All nonempty addresses are aligned and live.
    unsafe {
        let index=crate::index();let Some((lengths,weight,units,words))=p.shape.lengths() else {return;};
        if index>=units {return;}
        let input=core::array::from_fn(|i|core::slice::from_raw_parts(p.input[i] as *const f32,lengths[i]));
        let weight=core::slice::from_raw_parts(p.weight as *const f32,weight);
        let output=core::slice::from_raw_parts_mut((p.output as *mut f32).add(index*words),words);
        // Grouped plain rows use the one-row project_f32_ordered sequence for each
        // token: mul.rn.f32 of weight and input, then add.rn.f32 in ascending feature
        // order from +0. Pinned instructions keep the compiler from contracting them.
        if p.shape.op==crate::causal_ops::Op::ProjectRows {
            let cols=p.shape.cols as usize;
            for (token,out) in output.iter_mut().enumerate() {
                let mut sum=0.0f32;
                for j in 0..cols {
                    let a=weight[index*cols+j];let b=input[0][token*cols+j];let product: f32;
                    core::arch::asm!("mul.rn.f32 {p}, {a}, {b};",p=out(reg32)product,a=in(reg32)a,b=in(reg32)b);
                    core::arch::asm!("add.rn.f32 {s}, {s}, {p};",s=inout(reg32)sum,p=in(reg32)product);
                }
                *out=sum;
            }
            return;
        }
        unit(p.shape,index,input,weight,output);
    }
}
