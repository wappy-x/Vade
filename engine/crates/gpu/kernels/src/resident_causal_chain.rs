use crate::resident_launch_abi::{Command,Kind};
#[no_mangle]
pub extern "ptx-kernel" fn resident_launch_causal_chain(table:*const Command,command_index:u64) {
    // SAFETY: The checked recorder accepts only matching Prepare, StepPlanar
    // with dimension one hundred twenty-eight, and GatedNorm commands. All
    // operands and the aligned initialized table are retained until queue drain.
    // One CTA owns one head. Warp zero runs the original lane trees before and
    // after the original one-CTA recurrence. Every thread reaches each CTA
    // barrier, so the head's produced words are complete before its consumers.
    // The original intermediate buffers and every per-element operation remain.
    unsafe {
        let command=&*table.add(command_index as usize);if command.kind!=Kind::CausalChain as u32 {core::arch::nvptx::trap();}
        let payload=command.payload.as_ptr();
        crate::causal_fast::prepare_one(payload.cast::<crate::causal_ops::Arguments>().read());
        core::arch::nvptx::_syncthreads();
        crate::causal_fast::step_columns_128(&*payload.add(256).cast::<crate::causal_ops::Arguments>());
        core::arch::nvptx::_syncthreads();
        crate::causal_fast::gated_norm_one(payload.add(512).cast::<crate::causal_ops::Arguments>().read());
    }
}
