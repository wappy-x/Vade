//! The Flash-Next prefix cache's startup switches (FlashNext reads them; flashnext builds on
//! Linux only, so they live here, host-tested) and the layer pipeline's refusal of the cache.

/// VADE_FLASHNEXT_PREFIX_CACHE: 1 = on (requests with `cache:true` or a `prompt_cache_key`
/// retain their prompt); unset or 0 = off (no snapshots, main's path). Default off until a
/// same-session gate shows the cache ON ties main on every timed cell (36d-b gate).
pub fn prefix_cache() -> Result<bool, String> {
    switch(PREFIX_CACHE, std::env::var(PREFIX_CACHE.0))
}
/// VADE_FLASHNEXT_CACHE_YIELD: unset or 1 = retained prefixes yield the host memory the KV
/// reserve's spare session pages need before its pressure rule (variant b; the bound counts the
/// KV reserve's retention only); 0 = the session pool alone bounds the cache.
pub fn cache_yield() -> Result<bool, String> {
    switch(CACHE_YIELD, std::env::var(CACHE_YIELD.0))
}
/// VADE_FLASHNEXT_ADOPT: 1 = a finished request's KV arenas stay with the prompt prefix it
/// published in place (no row copies), and the conversation's next turn builds its session
/// on them (next_prefix.rs); unset or 0 = copies (default). Needs the prefix cache.
pub fn adopt() -> Result<bool, String> {
    switch(ADOPT, std::env::var(ADOPT.0))
}
/// The pipeline service (the tensor split and the layer chain) refuses VADE_FLASHNEXT_PREFIX_CACHE=1: run_pp on both ranks
/// before they link, FlashNext again on a configured front. A retained prefix, copied or
/// adopted, holds the front rank's layers [0, split) only. The back keeps no checkpoint of
/// its layers and accepts a slot by position, so a restored prompt would resume on whatever
/// state that slot last held: another prompt's KV and GDN rows. World 1 only until the back
/// checkpoints and restores its own layers.
pub fn refuse_on_pipeline(prefix_cache: bool) -> Result<(), String> {
    if prefix_cache {
        return Err("VADE_FLASHNEXT_PREFIX_CACHE=1 serves world 1 only: the back rank of two or more Sparks keeps no prefix checkpoints; unset it or set 0 there".into());
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
    fn the_pipeline_refuses_the_prefix_cache() {
        // Off: the pipeline starts as before. On: refused, naming the switch and the pipeline.
        assert_eq!(refuse_on_pipeline(false), Ok(()));
        let refusal = refuse_on_pipeline(true).unwrap_err();
        assert!(refusal.contains("VADE_FLASHNEXT_PREFIX_CACHE=1") && refusal.contains("back rank"), "{refusal}");
    }
}
