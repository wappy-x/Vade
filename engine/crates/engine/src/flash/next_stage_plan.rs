//! Host-checked stage boundary of a prompt pass (flash/next_stage.rs). A stage for layer l runs
//! the attention HC mix, attention, the MLP HC mix and MoE. The MoE combine of layer l either
//! runs in its own stage, or the attention mix of stage l + 1 takes it over (fused, no extra
//! residual pass). A pass over layers [first, end) defers only inside the pass, so its last
//! stage always leaves a whole residual.

/// True when layer `li`'s MoE combine is left to layer `li + 1`'s attention mix, in a pass that
/// ends at layer `end`. Not for layer 0 (the PLE residual update at layer 1 reads the residual),
/// not for the pass's last layer (its residual leaves the pass or feeds the final mixer), not
/// when a session reads the residual between layers (`unobserved` false), and not when the
/// tensor split forms the layer's MoE output without the deferred finish (`plain`, read only
/// when the other terms allow a deferral).
pub(crate) fn defers_moe(li: usize, end: usize, unobserved: bool, plain: impl FnOnce() -> bool) -> bool {
    unobserved && li >= 1 && li + 1 < end && !plain()
}

#[cfg(test)]
mod tests {
    use super::defers_moe;

    #[test]
    fn every_pass_ends_its_last_stage_with_a_whole_residual() {
        // Whole passes, pipeline fronts [0, split) and backs [split, 48): a deferral always has
        // its taker inside the pass, and the last stage never defers.
        for first in 0..48 {
            for end in first + 1..=48 {
                let deferred: Vec<usize> = (first..end).filter(|&li| defers_moe(li, end, true, || false)).collect();
                assert!(deferred.iter().all(|&li| li >= 1 && li + 1 < end), "pass [{first}, {end})");
                assert!(!defers_moe(end - 1, end, true, || false));
                assert_eq!(deferred.len(), (first.max(1)..end.saturating_sub(1)).count());
            }
        }
    }

    #[test]
    fn observed_or_plain_layers_combine_in_their_own_stage() {
        for li in 0..48 {
            assert!(!defers_moe(li, 48, false, || false));
            assert!(!defers_moe(li, 48, true, || true));
        }
        // The tensor-split query runs only when the other terms allow a deferral.
        let mut asked = 0;
        for li in 0..48 {
            defers_moe(li, 48, true, || { asked += 1; false });
        }
        assert_eq!(asked, 46);
        assert!(!defers_moe(5, 48, false, || panic!("plain read while observed")));
    }
}
