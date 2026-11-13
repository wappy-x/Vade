use crate::resident_launch_abi::{Command,Kind};
#[no_mangle]
pub extern "ptx-kernel" fn resident_launch_norm_tail(table:*const Command,command_index:u64) {
    // SAFETY: The recorder accepts only ResidentNorm's checked single-row RMS
    // and matching pointwise weight product. The table and every allocation
    // remain retained through ordered completion. RMS partials are complete
    // before this launch. The original apply body uses its original grid and
    // lane tree. Only the same thread reads each normalized word for the weight
    // product and optional final cast. The optional cast spans exactly the same
    // row in a separate checked output. No cross-thread dependency is added.
    unsafe {
        let command=&*table.add(command_index as usize);if command.kind!=Kind::NormTail as u32 {core::arch::nvptx::trap();}
        let payload=command.payload.as_ptr();
        crate::layer::resident_model_rms_apply(payload.cast::<crate::kernel_params::ModelRms>().read());
        crate::layer::resident_model_elementwise(payload.add(64).cast::<crate::kernel_params::ModelElementwise>().read());
        let cast=payload.add(128).cast::<crate::mhc_boundary_math::LaneCast>().read();
        let index=crate::index();
        if index<cast.words as usize {(cast.output as *mut f32).add(index).write(crate::bf16::rounded((cast.input as *const f32).add(index).read()));}
    }
}
