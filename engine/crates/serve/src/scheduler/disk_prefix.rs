//! Persistent disk prefix cache bridge for the cooperative scheduler.
//! Disk I/O runs only on RAM cache misses (restore) and completed prompt-end snapshots (persist),
//! keeping RAM-hit latency untouched and leaving engines without disk persistence as zero-cost no-ops.
use crate::{
    Engine, Session,
    cache::{CachedSession, SessionCache, SessionKey},
};
use std::cell::Cell;
use std::time::Instant;

thread_local! {
    static MIN_RESTORE_TOKENS: Cell<usize> = const { Cell::new(0) };
}

struct MinTokensGuard(usize);
impl MinTokensGuard {
    fn new(min_tokens: usize) -> Self {
        let prev = MIN_RESTORE_TOKENS.with(|cell| cell.replace(min_tokens));
        Self(prev)
    }
}
impl Drop for MinTokensGuard {
    fn drop(&mut self) {
        MIN_RESTORE_TOKENS.with(|cell| cell.set(self.0));
    }
}

/// Minimum prefix length a disk restore must strictly exceed to beat the best
/// in-RAM prefix candidate for the current request (`0` when RAM missed).
pub fn min_restore_tokens() -> usize {
    MIN_RESTORE_TOKENS.with(Cell::get)
}

/// Persist a completed prompt-end snapshot to the engine's optional disk prefix cache.
pub(super) fn persist_prompt_end<'e, E: Engine>(
    engine: &'e E,
    scope: Option<&str>,
    state: &CachedSession<E::Session<'e>>,
    prompt_end: bool,
) {
    if !prompt_end {
        return;
    }
    if let Err(error) = engine.persist_prefix_snapshot(scope, &state.session, &state.tokens, &state.next_logits) {
        engine_model::events::note(
            "disk_prefix_cache_save_failed",
            format_args!("tokens={} error={error}", state.tokens.len()),
        );
    }
}

/// Attempt to restore the longest matching prefix from disk strictly longer than `min_tokens`.
pub(super) fn restore_longer<'e, E: Engine>(
    engine: &'e E,
    scope: Option<&str>,
    prompt: &[u32],
    target_tokens: usize,
    min_tokens: usize,
) -> Option<CachedSession<E::Session<'e>>> {
    let _guard = MinTokensGuard::new(min_tokens);
    match engine.restore_prefix_from_disk(scope, prompt, target_tokens) {
        Ok(restored) => restored.filter(|s| s.tokens.len() > min_tokens),
        Err(error) => {
            engine_model::events::note(
                "disk_prefix_cache_restore_failed",
                format_args!("prompt_tokens={} error={error}", prompt.len()),
            );
            None
        }
    }
}

/// Promote a disk-restored prefix into the in-RAM `SessionCache` via zero-copy CoW pages so
/// subsequent turns hit RAM directly, then clear `next_logits` on continuations (`reused < prompt_len`).
pub(super) fn promote_and_trim<S: Session>(
    cache: &mut SessionCache<S>,
    key: Option<SessionKey>,
    scope: Option<&str>,
    retain: bool,
    prompt_len: usize,
    state: &mut CachedSession<S>,
    now: Instant,
) {
    if retain && let Some(key) = key {
        cache.fund_pages(state.session.prefix_allocations(state.tokens.len()).unwrap_or_default());
        cache.snapshot_scoped(key, scope, state, now, true);
    }
    if state.tokens.len() < prompt_len {
        state.next_logits.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_tokens_guard_restores_previous_threshold_on_drop_and_unwind() {
        assert_eq!(min_restore_tokens(), 0);
        {
            let _outer = MinTokensGuard::new(4096);
            assert_eq!(min_restore_tokens(), 4096);
            let panicked = std::panic::catch_unwind(|| {
                let _inner = MinTokensGuard::new(16_384);
                assert_eq!(min_restore_tokens(), 16_384);
                panic!("unwind check");
            });
            assert!(panicked.is_err());
            assert_eq!(min_restore_tokens(), 4096);
        }
        assert_eq!(min_restore_tokens(), 0);
    }
}
