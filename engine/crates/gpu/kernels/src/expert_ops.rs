//! Each thread owns one complete output unit. All arithmetic is shared.
use crate::expert_ops::{Arguments,GroupArguments,GroupEntry,LUT_WORDS,LaunchArguments,group_entry,group_run,group_unit,unit};
#[no_mangle]
pub extern "ptx-kernel" fn expert_operation(launch:LaunchArguments) {resident_expert_operation(launch)}
#[inline(always)]
pub(crate) fn resident_expert_operation(launch:LaunchArguments) {
    // SAFETY: The checked host shape bounds all slices and output units.
    // The host retains same-context contiguous inputs and immutable raw bytes.
    // The host initializes output storage before this mutable slice exists.
    // Each thread owns a disjoint output unit. Empty spans use aligned nonnull
    // addresses. A nonzero batch points to an aligned, host-checked Arguments
    // table with count entries. Each complete warp selects one checked job;
    // its nested operands and disjoint output span have the same contract.
    // All owners live until the host completes the launch stream.
    // Serial vector loads require naturally aligned complete tiles; their
    // guarded scalar fallback retains the same checked row and feature bounds.
    unsafe {
        let thread=crate::index();
        if matches!(launch.single.shape.op,crate::expert_ops::Op::Fp8Serial|crate::expert_ops::Op::PlainSerial) {crate::serial_projection::project!(launch,thread);return;}
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
#[no_mangle]
pub extern "ptx-kernel" fn expert_group(p:GroupArguments) {
    // SAFETY: The host checks every entry's shape, weight and scale owner, input
    // span and ascending unit range before launch. All owners and the table live
    // through stream completion, including the exact decode table. Weights and inputs are immutable. The output is
    // private and initialized earlier on the stream. group_unit is a bijection
    // on each entry's units, so each guarded thread owns one disjoint output unit.
    unsafe {
        let i=crate::index();if i>=p.units as usize {return;}let words=p.words as usize;
        let table=core::slice::from_raw_parts(p.table as *const GroupEntry,p.count as usize);let e=group_entry(table,i as u32);
        let x=core::slice::from_raw_parts((p.input as *const f32).add(e.first_row as usize*p.stride as usize),e.input_words as usize);
        let w=core::slice::from_raw_parts(e.weight as *const u8,e.weight_bytes as usize);let s=core::slice::from_raw_parts(e.scales as *const u8,e.scale_bytes as usize);
        let local=group_unit(e.shape,i-e.first as usize);
        let out=core::slice::from_raw_parts_mut((p.output as *mut f32).add((e.first as usize+local)*words),words);
        group_run(e.shape,local,x,w,s,&*(p.lut as *const [f32;LUT_WORDS]),out);
    }
}
