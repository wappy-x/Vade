//! Bounded target head: automatic only when removed scratch funds all its state.
use std::sync::OnceLock;
static ENABLED: OnceLock<Result<Option<bool>, String>> = OnceLock::new();
fn parse(value: Result<String, std::env::VarError>) -> Result<Option<bool>, String> {
    match value.as_deref() {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Ok("0") => Ok(Some(false)),
        Ok("1") => Ok(Some(true)),
        _ => Err("VADE_HEAD_BOUNDS must be unset, 0 or 1".into()),
    }
}
/// Validate once and report explicit opt-in before a model's funding is known.
pub fn load() -> Result<bool, String> {
    configured().map(|enabled| enabled.unwrap_or(false))
}
/// Refuse a malformed value at startup without fixing the switch: a pipeline rank pins it
/// (tp settings::PIPELINE_DEFAULTS) after argument parsing and before the model reads it.
pub fn validate() -> Result<(), String> {
    parse(std::env::var("VADE_HEAD_BOUNDS")).map(|_| ())?;
    if split_from_env()? && !crate::flash_tp_switch::from_env()? {
        return Err("VADE_W2_TP_HEAD=1 needs VADE_W2_TP=1 on both ranks".into());
    }
    Ok(())
}
/// VADE_W2_TP_HEAD (unset = 0 here; a split rank pins 1; with VADE_W2_TP=1): 1 splits the greedy
/// verify head of one-chat passes of 2..=RETAINED_ROWS rows by vocabulary halves across the two
/// ranks. Read once.
pub fn split_from_env() -> Result<bool, String> {
    static SPLIT: OnceLock<Result<bool, String>> = OnceLock::new();
    SPLIT.get_or_init(|| parse_split(std::env::var("VADE_W2_TP_HEAD").ok().as_deref())).clone()
}
fn parse_split(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err("VADE_W2_TP_HEAD must be 0 or 1".into()),
    }
}
/// Rows of rank 0's vocabulary half under VADE_W2_TP_HEAD: the lower half in whole 64-row
/// tiles (the bounded screen's unit); rank 1 takes the rest.
pub fn split_rows(vocabulary: usize) -> usize { vocabulary / 128 * 64 }
/// The head rows whose bounds this rank loads, as (start, rows): the whole head on a rank with the
/// head; under VADE_W2_TP_HEAD=1 (with VADE_W2_TP=1) the front its lower half and the headless rank 1
/// its upper half. None: no bounds (a headless pipeline rank).
pub fn bounds_rows(vocabulary: usize, headless: bool) -> Option<(usize, usize)> {
    split_bounds_rows(vocabulary, headless, split_from_env() == Ok(true) && crate::flash_tp_switch::from_env() == Ok(true))
}
fn split_bounds_rows(vocabulary: usize, headless: bool, split: bool) -> Option<(usize, usize)> {
    let half = split_rows(vocabulary);
    match (split, headless) {
        (false, true) => None,
        (false, false) => Some((0, vocabulary)),
        (true, false) => Some((0, half)),
        (true, true) => Some((half, vocabulary - half)),
    }
}
/// The split's two ranks, on its TCP link before each rank's wire thread starts: rank 1 writes its
/// VADE_W2_TP_HEAD (one byte; it cannot wait here, rank 0 reaches this only after rank 1 serves),
/// rank 0 reads it and stops at startup when it differs from its own (rank 1 then loses the link).
/// A rank that splits the head sends one mailbox message per eligible pass that the other rank
/// would never take, so a pair that disagrees must not serve.
pub fn agree_split(mut link: &std::net::TcpStream, rank: usize, on: bool, limit: std::time::Duration) -> Result<(), String> {
    use std::io::{Read, Write};
    if rank != 0 {
        return link.write_all(&[on as u8]).map_err(|e| format!("VADE_W2_TP_HEAD agreement: {e}"));
    }
    let before = link.read_timeout().map_err(|e| e.to_string())?;
    link.set_read_timeout(Some(limit)).map_err(|e| e.to_string())?;
    let mut peer = [0u8];
    let agreed = link.read_exact(&mut peer).map_err(|e| format!("VADE_W2_TP_HEAD agreement: {e}"));
    link.set_read_timeout(before).map_err(|e| e.to_string())?;
    agreed?;
    if (peer[0] != 0) != on {
        return Err(format!("VADE_W2_TP_HEAD is {} here and {} on the peer rank: set it on both ranks or neither", on as u8, peer[0]));
    }
    Ok(())
}
fn configured() -> Result<Option<bool>, String> {
    ENABLED.get_or_init(|| parse(std::env::var("VADE_HEAD_BOUNDS"))).clone()
}
/// Existing explicit switches keep their meaning; automatic mode must fund the
/// full twin, norms and retained inputs for every possible session, not just C1.
pub fn load_funded(saved: usize, twin: usize, vocabulary: usize, sessions: usize, mtp: bool) -> Result<bool, String> {
    configured().map(|enabled| enabled.unwrap_or_else(|| {
        let on = funded(saved, twin, vocabulary, sessions, mtp);
        if !on {
            engine_model::events::note("head_bounds_skipped", format_args!("reason={} saved_bytes={saved} twin_bytes={twin} vocabulary={vocabulary} sessions={sessions}",
                if !mtp { "no_draft" } else if twin == 0 { "no_twin" } else { "unfunded" }));
        }
        on
    }))
}
fn funded(saved: usize, twin: usize, vocabulary: usize, sessions: usize, mtp: bool) -> bool {
    if !mtp || twin == 0 { return false; }
    let needed = vocabulary.checked_mul(8).and_then(|n| n.checked_add(63)).map(|n| n / 64 * 64)
        .and_then(|norms| twin.checked_add(norms))
        .and_then(|n| sessions.checked_mul(RETAINED_INPUT_BYTES).and_then(|retained| n.checked_add(retained)));
    needed.is_some_and(|n| n <= saved)
}
/// Model-wide optional storage is forbidden on a pipeline rank that projects no head (the
/// VADE_W2_PP back and middle ranks); the front projects it as one Spark does.
/// No callback (and thus no allocation or log) runs when disabled or unused.
pub fn load_local<T>(enabled: bool, headless: bool, load: impl FnOnce() -> Result<T, String>) -> Result<Option<T>, String> {
    if enabled && !headless { load().map(Some) } else { Ok(None) }
}
/// Verify passes of up to eight rows (one canonical head group) take the bounded head.
pub const RETAINED_ROWS: usize = 8;
pub const RETAINED_INPUT_BYTES: usize = RETAINED_ROWS * 2560 * 4;
pub fn retained_input_bytes(enabled: bool) -> usize { if enabled { RETAINED_INPUT_BYTES } else { 0 } }
#[cfg(test)]
mod tests {
    use super::{parse, parse_split, agree_split, load_local, retained_input_bytes, funded, split_bounds_rows, split_rows};
    /// Each rank's bounds cover its own half in whole 64-row tiles, the halves tile the head, and
    /// without the switch the front keeps the whole head and a headless rank none.
    #[test]
    fn split_bounds_cover_each_ranks_half() {
        for vocabulary in [248_320usize, 151_936, 64 * 1000 + 64] {
            let half = split_rows(vocabulary);
            assert_eq!(half % 64, 0);
            assert!(half <= vocabulary - half && vocabulary - 2 * half < 128);
            assert_eq!(split_bounds_rows(vocabulary, false, true), Some((0, half)));
            assert_eq!(split_bounds_rows(vocabulary, true, true), Some((half, vocabulary - half)));
            assert_eq!(split_bounds_rows(vocabulary, false, false), Some((0, vocabulary)));
            assert_eq!(split_bounds_rows(vocabulary, true, false), None);
        }
        assert_eq!(split_rows(248_320), 124_160);
    }
    #[test]
    fn head_split_switch_is_strict_and_ranks_must_agree() {
        assert_eq!(parse_split(None), Ok(false));
        assert_eq!(parse_split(Some("0")), Ok(false));
        assert_eq!(parse_split(Some("1")), Ok(true));
        for value in ["", "2", "yes", " 1"] { assert!(parse_split(Some(value)).is_err()); }
        let limit = std::time::Duration::from_secs(5);
        for (front, back) in [(false, false), (true, true), (true, false), (false, true)] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let peer = std::thread::spawn(move || {
                let (link, _) = listener.accept().unwrap();
                // Rank 1 never waits: its byte is written before any later traffic on the link.
                agree_split(&link, 1, back, limit).map(|()| link)
            });
            let link = std::net::TcpStream::connect(address).unwrap();
            let theirs = peer.join().unwrap();
            assert!(theirs.is_ok());
            assert_eq!(agree_split(&link, 0, front, limit).is_ok(), front == back);
        }
    }
    #[test]
    fn disabled_and_headless_models_never_allocate_optional_head_state() {
        for (enabled, headless) in [(false,false),(false,true),(true,true)] {
            let value: Option<()> = load_local(enabled,headless,|| panic!("unused head allocator called")).unwrap();
            assert!(value.is_none());
            assert_eq!(retained_input_bytes(value.is_some()),0);
        }
        let calls=std::cell::Cell::new(0);
        let value=load_local(true,false,|| { calls.set(calls.get()+1); Ok(17) }).unwrap();
        assert_eq!(value,Some(17)); assert_eq!(calls.get(),1);
        assert_eq!(retained_input_bytes(value.is_some()),81920);
    }
    #[test]
    fn bounded_head_switch_is_strict_and_default_depends_on_funding() {
        assert_eq!(parse(Err(std::env::VarError::NotPresent)), Ok(None));
        assert_eq!(parse(Ok("0".into())), Ok(Some(false)));
        assert_eq!(parse(Ok("1".into())), Ok(Some(true)));
        for value in ["", "check", "true", "yes", "2", " 1", "1 "] {
            assert_eq!(parse(Ok(value.into())).unwrap_err(), "VADE_HEAD_BOUNDS must be unset, 0 or 1");
        }
        assert!(parse(Err(std::env::VarError::NotUnicode("bad".into()))).is_err());
    }
    #[test]
    fn automatic_head_never_adds_to_total_admission_for_any_eligible_tile() {
        let (twin, vocab, slots) = (655_564_800, 248_320, 16);
        let extra = twin + vocab * 8 + slots * super::RETAINED_INPUT_BYTES;
        for rows in [1, 63, 64, 65, 128, 2048, 4096, 8192] {
            let saved = crate::flash_next_workspace::saved_bytes(2560, 48, 16, rows);
            let enabled = funded(saved, twin, vocab, slots, true);
            assert_eq!(enabled, rows == 8192);
            assert!(saved >= if enabled { extra } else { 0 });
            if enabled { assert_eq!(saved - extra, 96_112_512); }
        }
        assert!(!funded(extra - 1, twin, vocab, slots, true));
        assert!(funded(extra, twin, vocab, slots, true));
        assert!(!funded(usize::MAX, twin, vocab, slots, false));
        assert!(!funded(usize::MAX, 0, vocab, slots, true));
        assert!(!funded(usize::MAX, twin, usize::MAX, slots, true));
        assert!(!funded(usize::MAX, twin, vocab, usize::MAX, true));
    }
    #[test]
    fn unfunded_or_unused_automatic_head_does_not_allocate() {
        for (saved, mtp, headless) in [(0, true, false), (usize::MAX, false, false), (usize::MAX, true, true)] {
            let enabled = parse(Err(std::env::VarError::NotPresent)).unwrap()
                .unwrap_or(funded(saved, 1024, 64, 16, mtp));
            let state: Option<()> = load_local(enabled, headless, || panic!("unfunded or unused head allocated")).unwrap();
            assert!(state.is_none());
            assert_eq!(retained_input_bytes(state.is_some()), 0);
        }
    }
}
