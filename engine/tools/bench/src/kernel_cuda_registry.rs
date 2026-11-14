#[allow(dead_code)]
#[path = "../../../crates/gpu/src/fp4_abi.rs"]
mod fp4;
#[allow(dead_code, unsafe_op_in_unsafe_fn, clippy::manual_is_multiple_of)] // Shared with the NVPTX crate; fix it with the kernels.
#[path = "../../../crates/gpu/src/fp8_abi.rs"]
mod fp8;
#[allow(dead_code, clippy::assign_op_pattern, clippy::duplicate_mod, clippy::manual_div_ceil, clippy::manual_is_multiple_of)] // Shared with the NVPTX crate; fix it with the kernels.
#[path = "../../../crates/gpu/kernel_params.rs"]
mod params;
#[path = "kernel_qsa.rs"]
mod qsa;
#[path = "kernel_order_gate.rs"] pub(super) mod order_gate;
#[path = "kernel_head.rs"] mod head;
#[path = "kernel_nvcols.rs"] mod nvcols;
pub(super) use nvcols::run as nv_cols;
#[path = "kernel_routerows.rs"] mod routerows;
pub(super) use routerows::run as route_rows;
#[path = "kernel_fp4cohort.rs"] mod fp4cohort;
pub(super) use fp4cohort::run as fp4_cohort;
#[path = "kernel_mxquant.rs"] mod mxquant;
pub(super) use mxquant::run as mx_quant;
#[path = "kernel_poolscores.rs"] mod poolscores;
#[path = "kernel_steptile.rs"] mod steptile;
pub(super) use steptile::run as step_tile;
pub(super) use poolscores::run as pool_scores;
pub(super) fn gather_ptx() -> String { qsa::gather_ptx() }
#[path = "kernel_gdn.rs"]
mod gdn;
#[path = "kernel_routed.rs"]
mod bw_routed;
use bw_routed::{flush_init, flush_on, icflush, launch_finish, launch_layer, launch_moe_route, launch_route, routed_fixture};
#[path = "kernel_prompt.rs"]
mod prompt;
#[path = "kernel_frontend.rs"]
mod frontend;
#[path = "kernel_fp4_prompt.rs"]
mod fp4_prompt;
#[path = "kernel_fp8.rs"] pub(super) mod fp8group;
#[path = "kernel_rows.rs"] pub(super) mod rows;
#[path = "kernel_mxgroup.rs"]
mod mxgroup;
#[path = "kernel_fp8moe.rs"] mod fp8moe;
#[path = "kernel_mhc.rs"] mod mhc; #[path = "kernel_plainp.rs"] pub(super) mod plainp; #[path = "kernel_causalp.rs"] pub(super) mod causalp; #[path = "kernel_readp.rs"] pub(super) mod readp;
pub(super) use mhc::{run as mhc_bytes, project::run as mhc_project, project::sweep as mhc_sweep, mxwave::run as mx_wave, nvbatch::run as nv_batch, rowstile::run as rows_tile, transp::run as transpose_smem, scorestile::run as scores_tile, rowsmall::run as rows_small, nofill::run as no_fill, nvsmall::run as nv_small, fp8small::run as fp8_small, mxsmall::run as mx_small, fp8tile::run as fp8_tile, nvrows::run as nv_rows, mxrows::run as mx_rows, fp8rows::run as fp8_rows, attnsweep::run as attn_table};
pub(super) use fp8moe::run as fp8_moe;
use mxgroup::{all_mx, group_mx, mx_range_warps, mxgroup_fixture, mxgroup_weight, mxrows_a, mxrows_entries, mxrows_shared, mxrows_x};
