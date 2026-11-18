//! How many launches each prompt row kernel took in this process, for proof logs: a model
//! proof must show that the model reached a new kernel, not only that a bench did.
use std::sync::atomic::{AtomicU64,Ordering};
const COUNT:usize=90;
const NAMES:[&str;COUNT]=["fp8_rows_tile","prompt_rows","plain_rows_tile","scores_rows_tile","scores_rows_tile_packed","step_rows_block","transpose_tiles","nv_rows_tile","expert_group","mx_rows_tile","mx_rows_groups","plain_rows_small","plain_rows_few","prompt_rows_small","nv_rows_small","fp8_rows_small","fp8_rows_multi","fp8_rows_few","mx_rows_small",
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
    "nv_quantize_pack",
    // GROUP FOLD: a prompt group folded a routed layer's down rows on their rank before the exchange (tp apply_experts_fold).
    "prompt_fold_first",
    // RECEIVE FIRST: host-staged exchanges whose receive was posted before the wait for this rank's GPU (not a launch).
    "prompt_receive_first",
    // GATHERED ROWS: a dense NVFP4 prompt group's operands read through the expert order table (no PermuteRows copy of the rows).
    "nv_quantize_gather",
    "shared_plain_fold",
    // MXFP8 on the tensor-core word (next_params::MX_TC): a descriptor of a one-row launch (alone or in a merged batch), of a
    // merged row stage of 2 to 4 source rows, a row group of 1 to 8 token rows, and a prompt tile piece.
    "mx_tc_step","mx_tc_rows_stage","mx_tc_rows","mx_tc_tile","mx_tc_pack",
    // A launch of a LAYER matrix on the scalar word (any path). With the layer word on tensor cores this stays 0: the proof
    // that no served path of a layer matrix fell back. The output head (scalar on purpose) is not counted.
    "mx_scalar_layer",
    // FOLD ROWS: one fold launch for 2 to 8 source rows of plain matrices (several chats in a step, a drafting check).
    "fold_rows","prompt_query_local","prompt_spares_kept",
    // MIXED ROWS: an MXFP8 row launch that also made the stage's small plain BF16 jobs (tp mx_rows_launch).
    "plain_rows_ride","fp8_rows_share","prompt_link_ahead","prompt_selection_checked","wide_group_layer","wide_cohort",
    // A fold operation of the absorbed latent attention in the step form (one decode row), on the tensor-core word
    // (backend causal_ops.rs); and those of them that the recorder took, which run the two recorded entries
    // (resident_launch_absorb_dense, resident_launch_absorb_attention). The prompt form and the chat rows count as
    // latent_absorption_tile.
    "absorb_step","absorb_step_recorded",
    // The pool scores of a verify pass or a batch step (1 to 16 token rows), a warp a word (pool_scores_small.rs).
    "packed_middle","prompt_pool_scores","mhc_boundary_block",
    // Lossless BF16 wire representation: bit movement kernel and completed gather.
    "prompt_wire_bf16","prompt_wire_bf16_exchange",
    "prompt_route_count","prompt_route_tables","prompt_route_host","prompt_route_native","prompt_route_product",
    // PAIRED PARTS: a native cohort of several sets built one work interval an expert (backend/prompt_native.rs).
    "paired_parts",
    // DIRECT PLACE: a native down cohort stored its products in original pair order (no placement launch).
    "direct_place",
    // BLOCK FP8 on the tensor cores (rebase train R2): which body the parts of a block FP8 launch took.
    "fp8_tc_step","fp8_tc_rows","fp8_tc_tile","fp8_generic_layer",
    // The prompt tile launches of block FP8 that took the THIRD form of the tile body (`fp8_rows::third`, kernel
    // `fp8_rows_tile3`). They are counted in `fp8_tc_tile` too: the same word from another body.
    "fp8_tc_tile3",
    // The launches of latent_absorption_tile that took the third form of the tile (latent_absorb_rows::third: the same bits).
    "latent_tile3",
    // The packed rows of an MXFP8 prompt group made from the RAW rows in one launch (the quantize pass and the pack).
    "mx_tc_quantize_pack",
    // WIDE PHASE: a prompt group above 3,072 rows drained the stream at the end of a mixer run or of a layer, so no
    // buffer it dropped before stays charged (tp bounded_decoder rows_phase_end; the charge rule `wide_phase_workspace`).
    "wide_mixer_complete"];
static COUNTS:[AtomicU64;COUNT]=[const {AtomicU64::new(0)};COUNT];
pub fn record(kernel:&str) {if let Some(at)=NAMES.iter().position(|n|*n==kernel) {COUNTS[at].fetch_add(1,Ordering::Relaxed);}}
/// One text line of all counts.
pub fn line()->String {NAMES.iter().zip(&COUNTS).map(|(n,c)|format!("{n}={}",c.load(Ordering::Relaxed))).collect::<Vec<_>>().join(" ")}
/// The MXFP8 word counters alone (`mx_tc_*` and `mx_scalar_layer`), for the served proof lines `SERVED MX COUNTS`.
pub const MX_NAMES:[&str;5]=["mx_tc_step","mx_tc_rows_stage","mx_tc_rows","mx_tc_tile","mx_scalar_layer"];
pub fn mx_line()->String {NAMES.iter().zip(&COUNTS).filter(|(n,_)|MX_NAMES.contains(n)).map(|(n,c)|format!("{n}={}",c.load(Ordering::Relaxed))).collect::<Vec<_>>().join(" ")}
/// The block FP8 word counters alone (`fp8_tc_*` and `fp8_generic_layer`), for the served proof lines `SERVED FP8 COUNTS`.
pub const FP8_NAMES:[&str;5]=["fp8_tc_step","fp8_tc_rows","fp8_tc_tile","fp8_generic_layer","fp8_tc_tile3"];
pub fn fp8_line()->String {NAMES.iter().zip(&COUNTS).filter(|(n,_)|FP8_NAMES.contains(n)).map(|(n,c)|format!("{n}={}",c.load(Ordering::Relaxed))).collect::<Vec<_>>().join(" ")}
#[cfg(test)]
mod tests {
    /// Every name has its own count, the last name too (a list longer than the counts would drop it from the line).
    #[test]
    fn every_name_counts_and_the_mx_line_holds_its_five() {
        let count=|line:&str,name:&str|line.split(' ').find_map(|f|f.strip_prefix(&format!("{name}="))?.parse::<u64>().ok());
        assert_eq!(super::line().split(' ').count(),super::NAMES.len());
        for name in super::NAMES {let before=count(&super::line(),name).unwrap();super::record(name);assert!(count(&super::line(),name).unwrap()>before,"{name}");}
        let line=super::mx_line();assert_eq!(line.split(' ').count(),5);
        for name in super::MX_NAMES {assert!(count(&line,name).unwrap()>=1,"{name}");}
        let line=super::fp8_line();assert_eq!(line.split(' ').count(),5);
        for name in super::FP8_NAMES {assert!(count(&line,name).unwrap()>=1,"{name}");}
    }

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
