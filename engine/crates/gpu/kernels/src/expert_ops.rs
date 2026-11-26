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
    // NV loads read exactly eight bytes within a checked sixteen-feature
    // block. Its row stride is a multiple of eight; misalignment uses bytes.
    unsafe {
        if launch.single.shape.op==crate::expert_ops::Op::Quantize && core::arch::nvptx::_block_dim_x() as usize==crate::nv_quantize::THREADS {crate::nv_quantize::lanes_owned(launch);return;}
        let thread=crate::index();
        // TENSOR-CORE REBASE (block FP8): the decode tile is the entry `fp8_tile` (its own launch kind), not a body of this
        // function. An Fp8Serial launch on a row-body block (4, 8, 16 or 32 warps) does not belong here.
        if launch.single.shape.op==crate::expert_ops::Op::Fp8Serial && crate::expert_ops::fp8_row_threads(core::arch::nvptx::_block_dim_x() as u32) {core::arch::nvptx::trap();}
        if matches!(launch.single.shape.op,crate::expert_ops::Op::Fp8Serial|crate::expert_ops::Op::PlainSerial) {crate::serial_projection::project!(launch,thread);return;}
        let (p,index)=if launch.batch==0 {
            let mapped=if launch.single.shape.op==crate::expert_ops::Op::NvProject {
                crate::expert_ops::nv_projection_batch_index(launch.single.shape.rows,launch.single.shape.cols,1,thread).map(|(_,row)|row)
            } else {crate::expert_ops::output_index(launch.single.shape.op,thread)};
            let Some(index)=mapped else {return;};(launch.single,index)
        } else {
            let mapped=if launch.single.shape.op==crate::expert_ops::Op::NvProject {
                crate::expert_ops::nv_projection_batch_index(launch.single.shape.rows,launch.single.shape.cols,launch.count,thread)
            } else {crate::expert_ops::projection_batch_index(launch.single.shape.rows,launch.count,thread)};
            let Some((job,row))=mapped else {return;};
            ((launch.batch as *const Arguments).add(job).read(),row)
        };
        let Some((lengths,wb,sb,units,words))=p.shape.lengths() else {return;};if index>=units {return;}
        let x=core::array::from_fn(|n|core::slice::from_raw_parts(p.input[n] as *const f32,lengths[n]));
        let w=core::slice::from_raw_parts(p.weight as *const u8,wb);let s=core::slice::from_raw_parts(p.scales as *const u8,sb);
        let out=core::slice::from_raw_parts_mut((p.output as *mut f32).add(index*words),words);
        // The packed row reads row-major bytes. A column-tile job keeps the scalar body, which finds its bytes by layout.
        // TENSOR-CORE REBASE: every NVFP4 word of this body is `unit`'s (`expert_ops::nv_word`, contract b). The packed
        // row-major shortcut (`nv_projection::row!`) keeps the rc.8 chain and is not used.
        unit(p.shape,index,x,w,s,out);
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
        // TENSOR-CORE REBASE (block FP8): an Fp8Rows row takes `group_run` too (`expert_ops::fp8_acc`, the contract word of
        // `tc_f8`). The word-load shortcut (`fp8_rows::row!`) kept the old one-chain word and is gone.
        // TENSOR-CORE REBASE: an NVFP4 row of any layout takes `group_run` (`unit`, `expert_ops::nv_word`, the contract
        // word of the matrix role). The word-load shortcut for row-major rows kept the rc.8 chain and is gone.
        group_run(e.shape,local,x,w,s,&*(p.lut as *const [f32;LUT_WORDS]),out);
    }
}
