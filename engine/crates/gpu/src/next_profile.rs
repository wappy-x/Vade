//! Flash-Next stage attribution through the existing opt-in event collector.
use super::Backend;

impl Backend<'_> {
    /// Per-target-row GPU events under VADE_DECODE_PROFILE=gpu. Host launch
    /// and wait APIs overlap device work; wall time is reported separately.
    /// Disabled profiling takes only the optional-profiler guard.
    pub fn next_rows_profile<R>(
        &self,
        verify: bool,
        position: usize,
        rows: usize,
        run: impl FnOnce() -> Result<R, String>,
    ) -> Result<R, String> {
        let Some(p) = &self.context.decode_profile else {
            return run();
        };
        if !p.borrow().gpu_enabled() || p.borrow().prefill_active() {
            return run();
        }
        self.prefill_profile_begin(
            position,
            rows,
            0,
            &["dense", "gdn", "attention", "experts", "head", "other_gpu"],
        );
        p.borrow_mut().next_rows(verify);
        let before = *self.stats.borrow();
        let result = run();
        let host = self.stats.borrow().since(before);
        p.borrow_mut().next_rows_host([
            host.stream_wait_ns,
            host.host_launch_ns,
            host.host_profile_ns,
        ]);
        self.prefill_profile_end(result.is_ok())?;
        result
    }
}

pub(super) fn flash_kind(name: &str) -> usize {
    match name {
        "flash_bf16_input" => 10,
        "flash_norm" | "flash_norm_decode" | "flash_hc_norm" => 1,
        "flash_qsa_prepare" | "flash_qsa_prepare_batch" | "flash_qsa_prepare2_batch" | "flash_dense_prepare_batch" => 2,
        "flash_qsa_attention_scores"
        | "flash_qsa_attention_scores_k8"
        | "flash_qsa_attention_scores_t"
        | "flash_qsa_attention_values_wide"
        | "flash_qsa_attention_values_wide2"
        | "flash_qsa_attention_values_wide4"
        | "flash_qsa_attention_values_wide4t"
        | "flash_qsa_attention_scores_tc"
        | "flash_qsa_attention_scores_tc_w"
        | "flash_qsa_attention_scores_tc2"
        | "flash_qsa_attention_softmax"
        | "flash_qsa_attention_softmax_v"
        | "flash_qsa_attention_values"
        | "flash_qsa_attention_values_s16"
        | "flash_qsa_attention_values_tc"
        | "flash_qsa_attention_values_tc_m"
        | "flash_qsa_attention_values_tc_k"
        | "flash_qsa_attention_values_tc_kr"
        | "flash_qsa_attention_fused_tc"
        | "flash_qsa_attention_values_tc2"
        | "flash_dense_attention_batch"
        | "flash_dense_scores"
        | "flash_dense_values" => 3,
        "flash_silu" => 6,
        "flash_residual_bf16" => 7,
        "flash_gdn_mv"
        | "flash_bf16_mma"
        | "flash_bf16_mma1"
        | "flash_bf16_mma64"
        | "flash_bf16_mma128"
        | "flash_bf16_mma_prefill_pairs" => 0,
        name if name.starts_with("flash_bf16_mma") => 0,
        name if name.starts_with("flash_next_gdn_") => 18,
        name if name.starts_with("flash_next_head") => 13,
        "logits_argmax_finish" => 13,
        name if name.starts_with("fp4_")
            || name.starts_with("flash_moe_")
            || name.starts_with("flash_next_compact_")
            || name == "flash_next_k16_finish"
            || name == "flash_next_route_mask" =>
        {
            21
        }
        "flash_next_dense_attention" | "flash_next_qsa_slot" => 3,
        name if name.starts_with("flash_gdn_") => 18,
        name if name.starts_with("flash_qsa_") => 3,
        _ => 9,
    }
}
