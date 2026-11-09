//! How many launches each prompt row kernel took in this process, for proof logs: a model
//! proof must show that the model reached a new kernel, not only that a bench did.
use std::sync::atomic::{AtomicU64,Ordering};
const NAMES:[&str;33]=["fp8_rows_tile","prompt_rows","plain_rows_tile","scores_rows_tile","step_rows_block","transpose_tiles","nv_rows_tile","expert_group","mx_rows_tile","mx_rows_groups","plain_rows_small","plain_rows_few","prompt_rows_small","nv_rows_small","fp8_rows_small","fp8_rows_multi","fp8_rows_few","mx_rows_small",
    // Lane P's reading entries: prompt_operation records the first seven; the mHC launcher records the last two.
    "prompt_permute","prompt_place","prompt_prepare","prompt_gated_norm","prompt_route","prompt_fold","prompt_conv","mhc_residual_rows","mhc_project_tile",
    // The zero fill a launch did not need (prompt_writes): counted so a proof shows the fill-free path ran.
    "prompt_fill_skipped","prompt_softmax_rows",
    // Lane P's narrow plain rows (up to 192 rows).
    "plain_rows_narrow",
    // Lane P's shared-memory transpose (exchange upload and row transposes).
    "transpose_smem",
    // Which mailbox put each message took, for proof logs.
    "tp_put_wide","tp_put_narrow"];
static COUNTS:[AtomicU64;33]=[const {AtomicU64::new(0)};33];
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
