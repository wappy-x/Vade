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
    // Row payloads use the original checked shape and the same launch geometry.
    // Their bodies keep every row and lane guard before an in-bounds offset.
    unsafe {
        let c=&*table.add(index as usize);if c.kind!=KIND {core::arch::nvptx::trap();}let p=c.payload.as_ptr();let words=p.cast::<u64>();
        match KIND {
            k if k==Kind::MxRows2 as u32=>crate::mx_rows_small::rows::<2,2,2>(p.cast::<crate::next_params::MxProjection>().read()),
            k if k==Kind::MxRows4 as u32=>crate::mx_rows_small::rows::<4,2,4>(p.cast::<crate::next_params::MxProjection>().read()),
            k if k==Kind::MxRows8 as u32=>crate::mx_rows_small::rows::<8,2,2>(p.cast::<crate::next_params::MxProjection>().read()),
            k if k==Kind::MxRows8w as u32=>crate::mx_rows_small::rows::<8,2,4>(p.cast::<crate::next_params::MxProjection>().read()),
            k if k==Kind::MxRowsTile as u32=>crate::mx_rows_tile::tile(p.cast::<crate::next_params::MxProjection>().read()),
            k if k==Kind::MixedRows as u32=>crate::next_decode::resident_projection_bf16_rows::<true>(p.cast::<crate::next_params::ResidentProjectionBatch>().read()),
            k if k==Kind::FoldRows as u32=>crate::next_decode::resident_projection_fold_rows::<true,false>(p.cast::<crate::next_params::ResidentProjectionBatch>().read()),
            k if k==Kind::FoldRowsF32 as u32=>crate::next_decode::resident_projection_fold_rows::<false,false>(p.cast::<crate::next_params::ResidentProjectionBatch>().read()),
            k if k==Kind::Fp8RowsSmall as u32=>crate::fp8_rows_small::fp8_rows_small(p.cast::<crate::expert_ops::GroupArguments>().read()),
            k if k==Kind::Fp8RowsMulti as u32=>crate::fp8_rows_multi::fp8_rows_multi(p.cast::<crate::expert_ops::GroupArguments>().read()),
            k if k==Kind::Fp8RowsFew as u32=>crate::fp8_rows_few::fp8_rows_few(p.cast::<crate::expert_ops::GroupArguments>().read()),
            k if k==Kind::Fp8RowsShare as u32=>crate::fp8_rows_share::fp8_rows_share(p.cast::<crate::expert_ops::GroupArguments>().read()),
            k if k==Kind::PlainRowsSmall as u32=>crate::plain_rows_small::plain_rows_small(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::PromptRows as u32=>crate::prompt_operation::prompt_rows(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::RowPrepare as u32=>crate::prompt_fast::prompt_prepare(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::RowGatedNorm as u32=>crate::prompt_fast::prompt_gated_norm(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::RowRecurrent as u32=>crate::recurrent_rows_kernel::body(p.cast::<crate::recurrent_rows::Arguments>().read()),
            k if k==Kind::RowRoute as u32=>crate::prompt_fast::prompt_route(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::RowFold as u32=>crate::prompt_fast::prompt_fold(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::RowPoolScores as u32=>crate::prompt_fast::prompt_pool_scores(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::RowPlace as u32=>crate::prompt_move::prompt_place(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::RowPermute as u32=>crate::prompt_move::prompt_permute(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::RowSelect as u32=>crate::select_fast::prompt_select_rows(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::TransposeSmem as u32=>crate::transpose_tiles::transpose_smem(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::DeviceRouteRows as u32=>crate::device_route::build_rows(p.cast::<crate::device_route_abi::BuildRows>().read()),
            k if k==Kind::DeviceRouteTcRows as u32=>crate::device_route::build_tc_rows(p.cast::<crate::device_route_abi::BuildTcRows>().read()),

            k if k==Kind::Causal as u32=>crate::causal_operation::resident_causal_operation(p.cast::<crate::causal_ops::Arguments>().read()),
            k if k==Kind::Expert as u32=>crate::expert_operation::resident_expert_operation(p.cast::<crate::expert_ops::LaunchArguments>().read()),
            k if k==Kind::CausalFast as u32=>crate::causal_operation::resident_causal_fast(&*p.cast::<crate::causal_ops::Arguments>()),
            k if k==Kind::NvStream as u32=>crate::nv_rows_stream::body::<64,256>(p.cast::<crate::expert_ops::LaunchArguments>().read()),
            // One recorded kind for the NV decode tile of either storage layout: the shape of the launch names the layout.
            k if k==Kind::NvTile as u32=>{let launch=p.cast::<crate::expert_ops::LaunchArguments>().read();
                if crate::expert_ops::nv_tiled(&launch.single.shape) {crate::nv_cols::project(launch)} else {crate::nv_tile::project::<16,8,4,2>(launch)}},
            // The block FP8 decode tile alone (the body `expert_operation` ran for these launches before).
            k if k==Kind::Fp8Tile as u32=>crate::fp8_tile::tile(p.cast::<crate::expert_ops::LaunchArguments>().read()),
            k if k==Kind::DeviceRoute as u32=>crate::device_route::build(p.cast::<crate::device_route_abi::Build>().read()),
            // The pair split. The raw quarter launch names the quarters a row gets in the word after its arguments.
            k if k==Kind::NvQuarters as u32=>{let launch=p.cast::<crate::expert_ops::LaunchArguments>().read();
                match *words.add(crate::resident_launch_abi::NV_QUARTERS_WORD) {
                    4=>crate::nv_cols_b::project::<4,true>(launch),2=>crate::nv_cols_b::project::<2,true>(launch),1=>crate::nv_cols_b::project::<1,true>(launch),
                    _=>core::arch::nvptx::trap(),
                }},
            k if k==Kind::QuarterFold as u32=>crate::pair_split::resident_nv_quarter_fold(p.cast::<crate::pair_split_abi::QuarterFold>().read()),
            k if k==Kind::ReduceTake as u32=>crate::pair_split::resident_tp_reduce_take(*words as _,*words.add(1) as _,*words.add(2),*words.add(3) as _,*words.add(4) as _,
                *words.add(5),*words.add(6) as _,*words.add(7),*words.add(8)),
            k if k==Kind::Plain as u32=>crate::ordered_projection::resident_project_f32_ordered(*words as _,*words.add(1) as _,*words.add(2) as _,*words.add(3) as u32,*words.add(4) as u32),
            k if k==Kind::AbsorbDense as u32=>crate::latent_absorption::resident_latent_absorption(p.cast::<crate::causal_ops::Arguments>().read()),
            k if k==Kind::AbsorbAttention as u32=>crate::latent_absorption::resident_latent_absorption_attention(p.cast::<crate::causal_ops::Arguments>().read()),
            // LATENT ROWS: the latent absorption of a check pass, each the direct entry's own body.
            k if k==Kind::LatentRowsDense as u32=>crate::latent_rows::latent_rows_dense(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::LatentRowsAttention as u32=>crate::latent_rows::latent_rows_attention(p.cast::<crate::prompt_ops::Arguments>().read()),
            k if k==Kind::PlainBf16 as u32=>crate::plain_bf16::resident_project_bf16_ordered(*words as _,*words.add(1) as _,*words.add(2) as _,*words.add(3) as u32,*words.add(4) as u32),
            k if k==Kind::Mx as u32=>crate::mx_rows_small::decode::<2,false>(p.cast::<crate::next_params::MxLayer>()),
            k if k==Kind::Mixed as u32=>crate::next_decode::resident_flash_resident_projection_batch::<false>(p.cast::<crate::next_params::ResidentProjectionBatch>().read()),
            k if k==Kind::MixedBf16 as u32=>crate::mx_rows_small::decode::<2,true>(p.cast::<crate::next_params::MxLayer>()),
            k if k==Kind::PlainBatch as u32=>crate::next_decode::resident_flash_resident_projection_plain_bf16(p.cast::<crate::next_params::ResidentProjectionBatch>().read()),
            k if k==Kind::Round as u32=>crate::mhc_boundary::resident_bf16_lanes_round(p.cast::<crate::mhc_boundary_math::LaneCast>().read()),
            k if k==Kind::Copy as u32=>crate::flash_ops::resident_flash_hybrid_copy(p.cast::<crate::kernel_params::HybridCopy>().read()),
            k if k==Kind::Prepare as u32=>crate::mhc::resident_mhc_prepare(p.cast::<crate::mhc_params::Prepare>().read()),
            k if k==Kind::Collapse as u32=>crate::mhc_boundary::resident_mhc_bf16_collapse(p.cast::<crate::mhc_params::Collapse>().read()),
            k if k==Kind::Combine as u32=>crate::mhc_boundary::resident_mhc_bf16_combine(p.cast::<crate::mhc_params::Combine>().read()),
            k if k==Kind::MhcBoundaryRms as u32=>crate::mhc_boundary_chain::resident_mhc_boundary_rms(p.cast::<crate::mhc_params::Boundary>().read()),
            k if k==Kind::MhcBoundaryDots as u32=>crate::mhc_boundary_chain::resident_mhc_boundary_dots(p.cast::<crate::mhc_params::Boundary>().read()),
            k if k==Kind::MhcBoundaryFinish as u32=>crate::mhc_boundary_fused::resident_mhc_boundary_finish(p.cast::<crate::mhc_params::Boundary>().read()),
            k if k==Kind::MhcBoundaryFused as u32=>crate::mhc_boundary_one::resident_mhc_boundary_fused::<0>(p.cast::<crate::mhc_params::FusedBoundary>().read()),
            k if k==Kind::MhcPhaseRms as u32=>crate::mhc_boundary_one::resident_mhc_boundary_fused::<1>(p.cast::<crate::mhc_params::FusedBoundary>().read()),
            k if k==Kind::MhcPhaseDots as u32=>crate::mhc_boundary_one::resident_mhc_boundary_fused::<2>(p.cast::<crate::mhc_params::FusedBoundary>().read()),
            k if k==Kind::MhcPhaseFinish as u32=>crate::mhc_boundary_one::resident_mhc_boundary_fused::<3>(p.cast::<crate::mhc_params::FusedBoundary>().read()),
            k if k==Kind::RmsPartials as u32=>crate::layer::resident_model_rms_partials(p.cast::<crate::kernel_params::ModelRms>().read()),
            k if k==Kind::RmsApply as u32=>crate::layer::resident_model_rms_apply(p.cast::<crate::kernel_params::ModelRms>().read()),
            k if k==Kind::Elementwise as u32=>crate::layer::resident_model_elementwise(p.cast::<crate::kernel_params::ModelElementwise>().read()),
            k if k==Kind::Put as u32=>crate::tp_mailbox::resident_tp_mailbox_put_finite(*words as _,*words.add(1),*words.add(2),*words.add(3),*words.add(4),*words.add(5) as _,*words.add(6) as _,*words.add(7) as _,*words.add(8),*words.add(9) as _),
            k if k==Kind::PutPlain as u32=>crate::tp_mailbox::resident_tp_mailbox_put(*words as _,*words.add(1),*words.add(2),*words.add(3),*words.add(4),*words.add(5) as _,*words.add(6) as _,*words.add(7) as _,*words.add(8)),
            k if k==Kind::ShadowPrefetch as u32=>crate::chain_shadow::resident_chain_shadow_prefetch(p.cast::<crate::mhc_params::ShadowPrefetch>().read()),
            // The window is the one a scatter take's tail rows read (bus_shadow::side checks it as bus_shadow::arm does): retained
            // resident weights, or a retained route table and the IDs of an earlier route build that no later command writes.
            k if k==Kind::ShadowSide as u32=>{crate::bus_shadow::side_window!(c);},
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
entry!(resident_launch_mixed_bf16,Kind::MixedBf16 as u32);
entry!(resident_launch_plain_batch,Kind::PlainBatch as u32);
entry!(resident_launch_nv_stream,18);
entry!(resident_launch_causal_fast,19);
entry!(resident_launch_nv_tile,20);
entry!(resident_launch_fp8_tile,Kind::Fp8Tile as u32);
entry!(resident_launch_device_route,21);
entry!(resident_launch_device_route_tc_rows,Kind::DeviceRouteTcRows as u32);
entry!(resident_launch_put_wide,23);
entry!(resident_launch_plain_bf16,26);
entry!(resident_launch_mhc_boundary_rms,0);
entry!(resident_launch_mhc_boundary_dots,24);
entry!(resident_launch_mhc_boundary_finish,25);
entry!(resident_launch_mhc_boundary_fused,Kind::MhcBoundaryFused as u32);
entry!(resident_launch_mhc_phase_rms,Kind::MhcPhaseRms as u32);
entry!(resident_launch_mhc_phase_dots,Kind::MhcPhaseDots as u32);
entry!(resident_launch_mhc_phase_finish,Kind::MhcPhaseFinish as u32);
entry!(resident_launch_nv_quarters,Kind::NvQuarters as u32);
entry!(resident_launch_quarter_fold,Kind::QuarterFold as u32);
entry!(resident_launch_reduce_take,Kind::ReduceTake as u32);
entry!(resident_launch_put_plain,27);
entry!(resident_launch_shadow_prefetch,28);
entry!(resident_launch_shadow_side,30);

entry!(resident_launch_absorb_dense,Kind::AbsorbDense as u32);
entry!(resident_launch_absorb_attention,Kind::AbsorbAttention as u32);
// LATENT ROWS (RB3): the rows form of a check pass, recorded.
entry!(resident_launch_latent_rows_dense,Kind::LatentRowsDense as u32);
entry!(resident_launch_latent_rows_attention,Kind::LatentRowsAttention as u32);

// Each row entry uses the checked original ABI and geometry through run.
entry!(resident_launch_mx_rows_small2,Kind::MxRows2 as u32);
entry!(resident_launch_mx_rows_small4,Kind::MxRows4 as u32);
entry!(resident_launch_mx_rows_small8,Kind::MxRows8 as u32);
entry!(resident_launch_mx_rows_small8w,Kind::MxRows8w as u32);
entry!(resident_launch_flash_resident_projection_bf16_rows,Kind::MixedRows as u32);
entry!(resident_launch_flash_resident_projection_fold_rows,Kind::FoldRows as u32);
entry!(resident_launch_flash_resident_projection_fold_rows_f32,Kind::FoldRowsF32 as u32);
entry!(resident_launch_fp8_rows_small,Kind::Fp8RowsSmall as u32);
entry!(resident_launch_fp8_rows_multi,Kind::Fp8RowsMulti as u32);
entry!(resident_launch_fp8_rows_few,Kind::Fp8RowsFew as u32);
entry!(resident_launch_fp8_rows_share,Kind::Fp8RowsShare as u32);
entry!(resident_launch_mx_rows_tile,Kind::MxRowsTile as u32);
entry!(resident_launch_plain_rows_small,Kind::PlainRowsSmall as u32);
entry!(resident_launch_prompt_rows,Kind::PromptRows as u32);
entry!(resident_launch_prompt_prepare,Kind::RowPrepare as u32);
entry!(resident_launch_prompt_gated_norm,Kind::RowGatedNorm as u32);
entry!(resident_launch_recurrent_rows,Kind::RowRecurrent as u32);
entry!(resident_launch_prompt_route,Kind::RowRoute as u32);
entry!(resident_launch_prompt_fold,Kind::RowFold as u32);
entry!(resident_launch_prompt_pool_scores,Kind::RowPoolScores as u32);
entry!(resident_launch_prompt_place,Kind::RowPlace as u32);
entry!(resident_launch_prompt_permute,Kind::RowPermute as u32);
entry!(resident_launch_prompt_select_rows,Kind::RowSelect as u32);
entry!(resident_launch_transpose_smem,Kind::TransposeSmem as u32);
entry!(resident_launch_device_route_rows,Kind::DeviceRouteRows as u32);
