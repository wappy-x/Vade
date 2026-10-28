//! The Flash-Next prefix cache's startup switches (FlashNext reads them; flashnext builds on
//! Linux only, so they live here, host-tested) and the cache's topology guard.

/// VADE_FLASHNEXT_PREFIX_CACHE: 1 = on (requests with `cache:true` or a `prompt_cache_key`
/// retain their prompt); unset or 0 = off (no snapshots, main's path). Default off until a
/// same-session gate shows the cache ON ties main on every timed cell (36d-b gate).
pub fn prefix_cache() -> Result<bool, String> {
    switch(PREFIX_CACHE, std::env::var(PREFIX_CACHE.0))
}
/// VADE_FLASHNEXT_CACHE_YIELD: unset or 1 = retained prefixes yield to current host memory
/// pressure; 0 = the session pool alone bounds the cache.
pub fn cache_yield() -> Result<bool, String> {
    switch(CACHE_YIELD, std::env::var(CACHE_YIELD.0))
}
/// Unused reserve capacity owns no memory. Actual spare buffers already count in `room`.
/// Retiring a request moves its buffers to the reserve without a new allocation.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn cache_physical_room(reserve: Option<(i128, usize)>, slack: usize) -> Option<(i128, usize)> {
    reserve.map(|(room, _unused_capacity)| (room, slack))
}
/// Independent memories must each have room for a retained prefix.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn cache_rank_room(local: Option<(i128, usize)>, peer: Option<i128>, slack: usize) -> Option<(i128, usize)> {
    match (local, peer) {
        (Some((room, margin)), Some(peer)) => Some((room.min(peer), margin.max(slack))),
        (None, Some(peer)) => Some((peer, slack)),
        (local, None) => local,
    }
}
/// VADE_FLASHNEXT_ADOPT: 1 = a finished request's KV arenas stay with the prompt prefix it
/// published in place (no row copies), and the conversation's next turn builds its session
/// on them (next_prefix.rs); unset or 0 = copies (default). Needs the prefix cache.
pub fn adopt() -> Result<bool, String> {
    switch(ADOPT, std::env::var(ADOPT.0))
}
/// A tensor split restores a matching prefix on each rank. The layer chain still has no
/// cache contract. Check before connecting ranks and again on the configured front.
pub fn refuse_on_pipeline(prefix_cache: bool, tensor_split: bool) -> Result<(), String> {
    if prefix_cache && !tensor_split {
        return Err("VADE_FLASHNEXT_PREFIX_CACHE=1 needs local execution or the tensor split: the layer chain has no prefix restore contract; unset it or set 0 there".into());
    }
    Ok(())
}
/// The switches and their values when unset.
const PREFIX_CACHE: (&str, bool) = ("VADE_FLASHNEXT_PREFIX_CACHE", false);
const CACHE_YIELD: (&str, bool) = ("VADE_FLASHNEXT_CACHE_YIELD", true);
const ADOPT: (&str, bool) = ("VADE_FLASHNEXT_ADOPT", false);
/// A 0/1 switch: its default when unset; any other value, a non-Unicode one included, refused.
fn switch((name, unset): (&str, bool), value: Result<String, std::env::VarError>) -> Result<bool, String> {
    match value.as_deref() {
        Ok("0") => Ok(false),
        Ok("1") => Ok(true),
        Err(std::env::VarError::NotPresent) => Ok(unset),
        _ => Err(format!("{name} must be 0 or 1")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::VarError;

    #[test]
    fn physical_room_keeps_pressure_and_the_disabled_reserve() {
        assert_eq!(cache_physical_room(None, 1024), None);
        for unused in [0, usize::MAX] {
            assert_eq!(cache_physical_room(Some((-1, unused)), 1024), Some((-1, 1024)));
            assert_eq!(cache_physical_room(Some((4096, unused)), 1024), Some((4096, 1024)));
        }
        assert_eq!(cache_rank_room(None, None, 1024), None);
        assert_eq!(cache_rank_room(None, Some(-1), 1024), Some((-1, 1024)));
        assert_eq!(cache_rank_room(Some((4096, 2048)), Some(1024), 512), Some((1024, 2048)));
        assert_eq!(cache_rank_room(Some((-1, 512)), Some(4096), 1024), Some((-1, 1024)));
    }

    #[test]
    fn switches_take_zero_or_one_and_the_cache_defaults_off() {
        // Shipped defaults: the cache and adoption off; yielding on (read only with the cache on).
        assert!(!PREFIX_CACHE.1 && !ADOPT.1 && CACHE_YIELD.1);
        for s in [PREFIX_CACHE, CACHE_YIELD, ADOPT] {
            let set = |v: &str| switch(s, Ok(v.to_string()));
            assert_eq!(switch(s, Err(VarError::NotPresent)), Ok(s.1));
            assert_eq!(set("0"), Ok(false));
            assert_eq!(set("1"), Ok(true));
            let refused = Err(format!("{} must be 0 or 1", s.0));
            for bad in ["", "2", "on", "true", "01", " 1", "1 "] {
                assert_eq!(set(bad), refused);
            }
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStringExt;
                let raw = std::ffi::OsString::from_vec(vec![b'1', 0xff]);
                assert_eq!(switch(s, Err(VarError::NotUnicode(raw))), refused);
            }
        }
    }

    #[test]
    fn the_split_accepts_cache_and_the_layer_chain_refuses_it() {
        for tensor in [false, true] { assert_eq!(refuse_on_pipeline(false, tensor), Ok(())); }
        assert_eq!(refuse_on_pipeline(true, true), Ok(()));
        let refusal = refuse_on_pipeline(true, false).unwrap_err();
        assert!(refusal.contains("VADE_FLASHNEXT_PREFIX_CACHE=1") && refusal.contains("layer chain"), "{refusal}");
    }
}
