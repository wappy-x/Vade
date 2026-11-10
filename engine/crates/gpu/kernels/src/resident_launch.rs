//! Replay the original kernel bodies with fresh, checked device-side arguments.
use crate::resident_launch_abi::{Command,Kind};
#[inline(always)]
fn run<const KIND:u32>(table:*const Command,index:u64) {
    // SAFETY: The checked recorder serializes every original ABI field and uses
    // its original launch geometry. The private aligned table covers index and
    // is uploaded before this replay on the same stream. Only registered kinds
    // are recorded. Every original operand's allocation remains retained by the
    // owned queue through submission and the real drain. No host padding is
    // read or serialized. Graph replays precede the next table upload. Thus each
    // body has exactly its direct entry's extents, aliases and initialization.
    // The causal-fast payload is an aligned, fully initialized Arguments value.
    // Its shared borrow cannot escape the call, and no replay writes the table.
    unsafe {
        let c=&*table.add(index as usize);if c.kind!=KIND {core::arch::nvptx::trap();}let p=c.payload.as_ptr();let words=p.cast::<u64>();
        match KIND {
            k if k==Kind::Causal as u32=>crate::causal_operation::resident_causal_operation(p.cast::<crate::causal_ops::Arguments>().read()),
            k if k==Kind::Expert as u32=>crate::expert_operation::resident_expert_operation(p.cast::<crate::expert_ops::LaunchArguments>().read()),
            k if k==Kind::CausalFast as u32=>crate::causal_operation::resident_causal_fast(&*p.cast::<crate::causal_ops::Arguments>()),
            k if k==Kind::NvStream as u32=>crate::nv_rows_stream::body::<64,256>(p.cast::<crate::expert_ops::LaunchArguments>().read()),
            k if k==Kind::NvTile as u32=>crate::nv_tile::project::<16,8,4,2>(p.cast::<crate::expert_ops::LaunchArguments>().read()),
            k if k==Kind::DeviceRoute as u32=>crate::device_route::build(p.cast::<crate::device_route_abi::Build>().read()),
            k if k==Kind::Plain as u32=>crate::ordered_projection::resident_project_f32_ordered(*words as _,*words.add(1) as _,*words.add(2) as _,*words.add(3) as u32,*words.add(4) as u32),
            k if k==Kind::PlainBf16 as u32=>crate::plain_bf16::resident_project_bf16_ordered(*words as _,*words.add(1) as _,*words.add(2) as _,*words.add(3) as u32,*words.add(4) as u32),
            k if k==Kind::Mx as u32=>crate::next_decode::resident_flash_next_mxfp8_m1(p.cast::<crate::next_params::MxLayer>().read()),
            k if k==Kind::Mixed as u32=>crate::next_decode::resident_flash_resident_projection_batch(p.cast::<crate::next_params::ResidentProjectionBatch>().read()),
            k if k==Kind::Round as u32=>crate::mhc_boundary::resident_bf16_lanes_round(p.cast::<crate::mhc_boundary_math::LaneCast>().read()),
            k if k==Kind::Copy as u32=>crate::flash_ops::resident_flash_hybrid_copy(p.cast::<crate::kernel_params::HybridCopy>().read()),
            k if k==Kind::Prepare as u32=>crate::mhc::resident_mhc_prepare(p.cast::<crate::mhc_params::Prepare>().read()),
            k if k==Kind::Collapse as u32=>crate::mhc_boundary::resident_mhc_bf16_collapse(p.cast::<crate::mhc_params::Collapse>().read()),
            k if k==Kind::Combine as u32=>crate::mhc_boundary::resident_mhc_bf16_combine(p.cast::<crate::mhc_params::Combine>().read()),
            k if k==Kind::MhcBoundaryRms as u32=>crate::mhc_boundary_chain::resident_mhc_boundary_rms(p.cast::<crate::mhc_params::Boundary>().read()),
            k if k==Kind::MhcBoundaryDots as u32=>crate::mhc_boundary_chain::resident_mhc_boundary_dots(p.cast::<crate::mhc_params::Boundary>().read()),
            k if k==Kind::MhcBoundaryFinish as u32=>crate::mhc_boundary_fused::resident_mhc_boundary_finish(p.cast::<crate::mhc_params::Boundary>().read()),
            k if k==Kind::RmsPartials as u32=>crate::layer::resident_model_rms_partials(p.cast::<crate::kernel_params::ModelRms>().read()),
            k if k==Kind::RmsApply as u32=>crate::layer::resident_model_rms_apply(p.cast::<crate::kernel_params::ModelRms>().read()),
            k if k==Kind::Elementwise as u32=>crate::layer::resident_model_elementwise(p.cast::<crate::kernel_params::ModelElementwise>().read()),
            k if k==Kind::Put as u32=>crate::tp_mailbox::resident_tp_mailbox_put_finite(*words as _,*words.add(1),*words.add(2),*words.add(3),*words.add(4),*words.add(5) as _,*words.add(6) as _,*words.add(7) as _,*words.add(8),*words.add(9) as _),
            k if k==Kind::PutPlain as u32=>crate::tp_mailbox::resident_tp_mailbox_put(*words as _,*words.add(1),*words.add(2),*words.add(3),*words.add(4),*words.add(5) as _,*words.add(6) as _,*words.add(7) as _,*words.add(8)),
            k if k==Kind::ShadowPrefetch as u32=>crate::chain_shadow::resident_chain_shadow_prefetch(p.cast::<crate::mhc_params::ShadowPrefetch>().read()),
            k if k==Kind::PutWide as u32=>crate::tp_mailbox::resident_tp_mailbox_put_wide(*words as _,*words.add(1),*words.add(2) as _,*words.add(3) as _,*words.add(4) as _,*words.add(5),*words.add(6) as _,*words.add(7) as _),
            k if k==Kind::Take as u32=>crate::tp_mailbox::resident_tp_mailbox_take(*words as _,*words.add(1),*words.add(2),*words.add(3),*words.add(4),*words.add(5) as _,*words.add(6) as _,*words.add(7),*words.add(8) as _,*words.add(9)),
            k if k==Kind::CopyWords as u32 || k==Kind::ZeroWords as u32=>{
                let i=crate::index();if i<*words.add(2) as usize {let out=*words.add(1) as *mut u32;*out.add(i)=if k==Kind::CopyWords as u32 {*(*words as *const u32).add(i)} else {0};}
            },
            _=>core::arch::nvptx::trap(),
        }
    }
}
macro_rules! entry {($name:ident,$kind:expr)=>{#[no_mangle] pub extern "ptx-kernel" fn $name(table:*const Command,index:u64) {run::<{$kind}>(table,index)}};}
entry!(resident_launch_causal,1);entry!(resident_launch_expert,2);entry!(resident_launch_plain,3);entry!(resident_launch_mx,4);
entry!(resident_launch_round,5);entry!(resident_launch_copy,6);entry!(resident_launch_put,7);entry!(resident_launch_take,8);
entry!(resident_launch_prepare,9);entry!(resident_launch_collapse,10);entry!(resident_launch_combine,11);
entry!(resident_launch_rms_partials,12);entry!(resident_launch_rms_apply,13);entry!(resident_launch_elementwise,14);
entry!(resident_launch_copy_words,15);entry!(resident_launch_zero_words,16);

entry!(resident_launch_mixed,17);
entry!(resident_launch_nv_stream,18);
entry!(resident_launch_causal_fast,19);
entry!(resident_launch_nv_tile,20);
entry!(resident_launch_device_route,21);
entry!(resident_launch_put_wide,23);
entry!(resident_launch_plain_bf16,26);
entry!(resident_launch_mhc_boundary_rms,0);
entry!(resident_launch_mhc_boundary_dots,24);
entry!(resident_launch_mhc_boundary_finish,25);
entry!(resident_launch_put_plain,27);
entry!(resident_launch_shadow_prefetch,28);
