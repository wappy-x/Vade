use super::{fork_failed, prompt_cohort_candidate, prompt_uses_cohort};

// Hybrid pooled geometry: 512-row short limit, 256-row canonical tile.
const LIMIT: usize = 512;
const CHUNK: usize = 256;

#[test]
fn lone_long_prompt_takes_long_path() {
    // One 8192-token tile-aligned prompt alone: not cohort, so the
    // dispatch long-prefill branch (which requires !shared_prompt) runs it.
    assert!(!prompt_uses_cohort(8192, 0, LIMIT, CHUNK, 1));
    // The same prompt with another prompt-ready job shares cohort chunks.
    assert!(prompt_uses_cohort(8192, 0, LIMIT, CHUNK, 2));
    assert!(prompt_uses_cohort(8192, 0, LIMIT, CHUNK, 4));
    // It is a cohort candidate either way; only the count decides.
    assert!(prompt_cohort_candidate(8192, 0, LIMIT, CHUNK));
    // Mid-prompt at a tile boundary behaves the same.
    assert!(!prompt_uses_cohort(4096, 4096, LIMIT, CHUNK, 1));
    assert!(prompt_uses_cohort(4096, 4096, LIMIT, CHUNK, 2));
}

#[test]
fn short_and_unaligned_prompts_keep_existing_paths() {
    // Short prompts share even when alone (existing behavior).
    assert!(prompt_uses_cohort(LIMIT, 0, LIMIT, CHUNK, 1));
    assert!(prompt_uses_cohort(1, 8191, LIMIT, CHUNK, 1));
    // Long prompts off a tile boundary never join the cohort.
    assert!(!prompt_uses_cohort(8192, 100, LIMIT, CHUNK, 2));
    // Engines without chunked sharing never admit long prompts.
    assert!(!prompt_uses_cohort(8192, 0, LIMIT, 0, 2));
    assert!(prompt_uses_cohort(LIMIT, 0, LIMIT, 0, 1));
}

#[cfg(target_os = "linux")]
#[test]
fn tile_constants_match_engine_schedule() {
    assert_eq!(spark_engine::hybrid::schedule::PROMPT_ROWS, CHUNK);
    assert_eq!(spark_engine::hybrid::schedule::ROUTED_GROUP, 4096);
}

#[test]
fn cohort_and_long_paths_visit_same_tiles_in_order() {
    // Engine schedule geometry (hybrid/schedule.rs; Linux-gated there).
    const PROMPT_ROWS: usize = CHUNK;
    const ROUTED_GROUP: usize = 4096;
    // Cohort: one canonical tile per tick. Long path: ROUTED_GROUP
    // passes whose attention visits PROMPT_ROWS tiles in scan order
    // (hybrid_decode.rs prompt tiling). Both must partition the same
    // per-session tile sequence, so switching paths at a tile boundary
    // cannot reorder any token's arithmetic.
    for total in [8192usize, 32768] {
        let cohort: Vec<(usize, usize)> = (0..total).step_by(PROMPT_ROWS)
            .map(|first| (first, PROMPT_ROWS.min(total - first))).collect();
        let long: Vec<(usize, usize)> = (0..total).step_by(ROUTED_GROUP)
            .flat_map(|group| {
                let rows = ROUTED_GROUP.min(total - group);
                (0..rows).step_by(PROMPT_ROWS)
                    .map(move |first| (group + first, PROMPT_ROWS.min(rows - first)))
            }).collect();
        assert_eq!(cohort, long, "tile order differs at {total} tokens");
    }
}

#[test]
fn each_passed_fork_error_is_one_event() {
    // Counts only grow, so other tests that note events at the same time cannot break this check.
    let before = engine_model::events::count("prefix_cache_fork_failed");
    fork_failed("automatic", 7, &["fork failed: a".into(), "truncate failed: b".into()]);
    fork_failed("session", 7, &[]);
    assert!(engine_model::events::count("prefix_cache_fork_failed") >= before + 2);
}
