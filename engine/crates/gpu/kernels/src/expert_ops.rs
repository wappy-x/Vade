//! Each thread owns one complete output unit. All arithmetic is shared.
use crate::expert_ops::{Arguments,LaunchArguments,unit};
#[no_mangle]
pub extern "ptx-kernel" fn expert_operation(launch:LaunchArguments) {
    // SAFETY: The checked host shape bounds all slices and output units.
    // The host retains same-context contiguous inputs and immutable raw bytes.
    // The host initializes output storage before this mutable slice exists.
    // Each thread owns a disjoint output unit. Empty spans use aligned nonnull
    // addresses. A nonzero batch points to an aligned, host-checked Arguments
    // table with count entries. Each complete warp selects one checked job;
    // its nested operands and disjoint output span have the same contract.
    // All owners live until the host completes the launch stream.
    unsafe {
        let thread=crate::index();
        let (p,index)=if launch.batch==0 {let Some(index)=crate::expert_ops::output_index(launch.single.shape.op,thread) else {return;};(launch.single,index)} else {
            let Some((job,row))=crate::expert_ops::projection_batch_index(launch.single.shape.rows,launch.count,thread) else {return;};
            ((launch.batch as *const Arguments).add(job).read(),row)
        };
        let Some((lengths,wb,sb,units,words))=p.shape.lengths() else {return;};if index>=units {return;}
        let x=core::array::from_fn(|n|core::slice::from_raw_parts(p.input[n] as *const f32,lengths[n]));
        let w=core::slice::from_raw_parts(p.weight as *const u8,wb);let s=core::slice::from_raw_parts(p.scales as *const u8,sb);
        let out=core::slice::from_raw_parts_mut((p.output as *mut f32).add(index*words),words);unit(p.shape,index,x,w,s,out);
    }
}
