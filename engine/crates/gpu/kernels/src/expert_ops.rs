//! Each thread owns one complete output unit. All arithmetic is shared.
use crate::expert_ops::{Arguments,unit};
#[no_mangle]
pub extern "ptx-kernel" fn expert_operation(p:Arguments) {
    // SAFETY: The checked host shape bounds all slices and output units.
    // The host retains same-context contiguous inputs and immutable raw bytes.
    // The host initializes output storage before this mutable slice exists.
    // Each thread owns a disjoint output unit. Empty spans use aligned nonnull
    // addresses. All owners live until the host completes the launch stream.
    unsafe {
        let Some(i)=crate::expert_ops::output_index(p.shape.op,crate::index()) else {return;};let Some((lengths,wb,sb,units,words))=p.shape.lengths() else {return;};if i>=units {return;}
        let x=core::array::from_fn(|n|core::slice::from_raw_parts(p.input[n] as *const f32,lengths[n]));
        let w=core::slice::from_raw_parts(p.weight as *const u8,wb);let s=core::slice::from_raw_parts(p.scales as *const u8,sb);
        let out=core::slice::from_raw_parts_mut((p.output as *mut f32).add(i*words),words);unit(p.shape,i,x,w,s,out);
    }
}
