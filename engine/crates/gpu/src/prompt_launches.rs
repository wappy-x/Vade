//! How many launches each prompt row kernel took in this process, for proof logs: a model
//! proof must show that the model reached a new kernel, not only that a bench did.
use std::sync::atomic::{AtomicU64,Ordering};
const NAMES:[&str;49]=["fp8_rows_tile","prompt_rows","plain_rows_tile","scores_rows_tile","scores_rows_tile_packed","step_rows_block","transpose_tiles","nv_rows_tile","expert_group","mx_rows_tile","mx_rows_groups","plain_rows_small","plain_rows_few","prompt_rows_small","nv_rows_small","fp8_rows_small","fp8_rows_multi","fp8_rows_few","mx_rows_small",
    // Lane P's reading entries: prompt_operation records the first seven; the mHC launcher records the last two.
    "prompt_permute","prompt_place","prompt_prepare","prompt_gated_norm","prompt_route","prompt_fold","prompt_conv","mhc_residual_rows","mhc_project_tile",
    // The zero fill a launch did not need (prompt_writes): counted so a proof shows the fill-free path ran.
    "prompt_fill_skipped","prompt_softmax_rows",
    // Lane P's narrow plain rows (up to 192 rows).
    "plain_rows_narrow",
    // Lane P's shared-memory transpose (exchange upload and row transposes).
    "transpose_smem",
    // Which mailbox put each message took, for proof logs.
    "tp_put_wide","tp_put_narrow","tp_put_join",
    // The prompt row tile over column-tile NVFP4 weights.
    "nv_rows_cols",
    // A column-tile prompt group of few tokens run as decode jobs.
    "nv_cols",
    // A group past the reading kernels' swept token range: launches made as runs of tokens (prompt_chunks), and mHC row runs.
    "prompt_token_run","mhc_row_run",
    // A prompt layer's ready projections ran as one stage with shared exchanges (tp projection_stage.rs apply_matrix_stage).
    "row_stage",
    // A prompt group past 16 rows folded only the index pools it completed (tp rows/pools.rs append_window).
    "pool_window",
    // A prompt rows pass ran with launch recording paused (tp bounded_decoder rows.rs rows_pass).
    "rows_unrecorded","latent_absorption_tile","prompt_big_group",
    // ROW STAGE: a verify pass's MXFP8 and plain projections of one input as one merged launch of 2 to 4 source rows.
    "mixed_rows_stage",
    // A column-tile prompt group on the tensor-core contract (second image).
    "fp4_cohort",
    // The indexer's pool scores of a prompt group in shared-memory tiles.
    "pool_scores_tile",
    // The recurrent update of a prompt group with its tokens staged ahead in shared memory.
    "step_rows_tile",
    // A dense NVFP4 prompt group's operands in one pass from the FP32 rows (no quantizer rows, no pack).
    "nv_quantize_pack"];
static COUNTS:[AtomicU64;49]=[const {AtomicU64::new(0)};49];
pub fn record(kernel:&str) {if let Some(at)=NAMES.iter().position(|n|*n==kernel) {COUNTS[at].fetch_add(1,Ordering::Relaxed);}}
/// One text line of all counts.
pub fn line()->String {NAMES.iter().zip(&COUNTS).map(|(n,c)|format!("{n}={}",c.load(Ordering::Relaxed))).collect::<Vec<_>>().join(" ")}
#[cfg(test)]
mod tests {
    #[test]
    fn counts_named_kernels_and_ignores_others() {
        super::record("transpose_tiles");super::record("other");
        assert!(super::line().contains("transpose_tiles=1") && !super::line().contains("other"));
    }
    #[test]
    fn counts_every_reading_entry_the_prompt_and_mhc_launchers_can_choose() {
        for name in ["prompt_permute","prompt_place","prompt_prepare","prompt_gated_norm","prompt_route","prompt_fold","prompt_conv","mhc_residual_rows","mhc_project_tile"] {
            super::record(name);assert!(super::line().contains(&format!("{name}=1")),"{name}");
        }
    }
}
