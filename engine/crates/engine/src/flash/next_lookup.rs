//! Exact eight-token keys; proposals never authorize output tokens.
use std::collections::HashMap;

/// VADE_MTP_LOOKUP=0 isolates model-head proposals for acceptance audits.
/// Unset preserves prompt lookup; target verification still selects every token.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_MTP_LOOKUP").as_deref() != Ok("0"))
}
/// Best completed single-chat setting named by the supplied GPU tuning receipt.
pub const DEFAULT_MTP_DEPTH: usize = 3;

/// Extra funding above resident MTP: small PLE/hidden prefixes, GDN
/// projection tapes, replacement logits and a conservative host lookup bound.
/// One entry matrix snapshot is already funded by resident MTP.
pub fn reservation_bytes(capacity: usize, vocab: usize, depth: usize, max: usize) -> usize {
    let extra = max.saturating_sub(depth);
    extra * (10240 * 40 + 36 * (10240 + 48 + 48) * 4)
        + (max + 1).max(depth + 1) * vocab * 4
        + capacity * 256
        + (1 << 20)
}

#[derive(Clone, Default)]
pub struct Lookup {
    spans: HashMap<[u32; 8], usize>,
    seen: usize,
}

#[test]
fn exact_keys_retain_early_prompt_and_clear_on_truncation() {
    let mut lookup = Lookup::default();
    let mut history: Vec<u32> = (0..9000).collect();
    history.extend(0..7);
    assert_eq!(lookup.propose(&history, 7, 15), (8..23).collect::<Vec<_>>());
    assert!(lookup.propose(&history, 9001, 15).is_empty());
    assert!(lookup.propose(&[0; 7], 7, 15).is_empty());
}
impl Lookup {
    /// History contains committed inputs; `next` completes the query key.
    pub fn propose(&mut self, history: &[u32], next: u32, max: usize) -> Vec<u32> {
        if history.len() < self.seen {
            *self = Self::default();
        }
        for end in self.seen.max(8)..history.len() {
            let mut key = [0; 8];
            key.copy_from_slice(&history[end - 8..end]);
            self.spans.entry(key).or_insert(end);
        }
        self.seen = history.len();
        if history.len() < 7 || max == 0 {
            return Vec::new();
        }
        let mut key = [next; 8];
        key[..7].copy_from_slice(&history[history.len() - 7..]);
        self.spans.get(&key).map_or_else(Vec::new, |&end| {
            history[end..end + max.min(15).min(history.len() - end)].to_vec()
        })
    }
}
