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

/// Multi-length exact suffix lookup (`8`, `6`, `4`, `3`, `2`, `1` tokens) storing the most recent
/// match position for each key. Supports both pre-draft proposals (`tail = &[anchor]`)
/// and hybrid neural+suffix extensions (`tail = &[anchor, d_0, ..., d_k]`) up to 15 tokens.
#[derive(Clone, Default)]
pub struct SuffixLookup {
    spans8: HashMap<[u32; 8], usize>,
    spans6: HashMap<[u32; 6], usize>,
    spans4: HashMap<[u32; 4], usize>,
    spans3: HashMap<[u32; 3], usize>,
    spans2: HashMap<u64, usize>,
    spans1: HashMap<u32, usize>,
    seen: usize,
}

impl SuffixLookup {
    fn key2(a: u32, b: u32) -> u64 {
        (u64::from(a) << 32) | u64::from(b)
    }

    pub fn sync(&mut self, history: &[u32]) {
        if history.len() < self.seen {
            *self = Self::default();
        }
        for end in self.seen.max(1)..history.len() {
            if end >= 8 {
                let mut k8 = [0; 8];
                k8.copy_from_slice(&history[end - 8..end]);
                self.spans8.insert(k8, end);
            }
            if end >= 6 {
                let mut k6 = [0; 6];
                k6.copy_from_slice(&history[end - 6..end]);
                self.spans6.insert(k6, end);
            }
            if end >= 4 {
                let mut k4 = [0; 4];
                k4.copy_from_slice(&history[end - 4..end]);
                self.spans4.insert(k4, end);
            }
            if end >= 3 {
                let mut k3 = [0; 3];
                k3.copy_from_slice(&history[end - 3..end]);
                self.spans3.insert(k3, end);
            }
            if end >= 2 {
                self.spans2
                    .insert(Self::key2(history[end - 2], history[end - 1]), end);
            }
            self.spans1.insert(history[end - 1], end);
        }
        self.seen = history.len();
    }

    /// Look up the longest matching suffix of `history ++ tail` (checking lengths 8, 6, 4, 3, 2, 1
    /// down to `min_match`) and return `(proposed_tokens, matched_suffix_len)` capped at `max.min(15)`.
    pub fn propose_after(
        &mut self,
        history: &[u32],
        tail: &[u32],
        max: usize,
        min_match: usize,
    ) -> (Vec<u32>, usize) {
        self.sync(history);
        let limit = max.min(15);
        let total = history.len() + tail.len();
        if limit == 0 || total < min_match.max(1) || history.is_empty() {
            return (Vec::new(), 0);
        }
        let token_at = |idx: usize| -> u32 {
            if idx < history.len() {
                history[idx]
            } else {
                tail[idx - history.len()]
            }
        };
        if min_match <= 8 && total >= 8 {
            let mut k8 = [0; 8];
            for (j, slot) in k8.iter_mut().enumerate() {
                *slot = token_at(total - 8 + j);
            }
            if let Some(&end) = self.spans8.get(&k8) {
                let take = limit.min(history.len() - end);
                if take > 0 {
                    return (history[end..end + take].to_vec(), 8);
                }
            }
        }
        if min_match <= 6 && total >= 6 {
            let mut k6 = [0; 6];
            for (j, slot) in k6.iter_mut().enumerate() {
                *slot = token_at(total - 6 + j);
            }
            if let Some(&end) = self.spans6.get(&k6) {
                let take = limit.min(history.len() - end);
                if take > 0 {
                    return (history[end..end + take].to_vec(), 6);
                }
            }
        }
        if min_match <= 4 && total >= 4 {
            let mut k4 = [0; 4];
            for (j, slot) in k4.iter_mut().enumerate() {
                *slot = token_at(total - 4 + j);
            }
            if let Some(&end) = self.spans4.get(&k4) {
                let take = limit.min(history.len() - end);
                if take > 0 {
                    return (history[end..end + take].to_vec(), 4);
                }
            }
        }
        if min_match <= 3 && total >= 3 {
            let mut k3 = [0; 3];
            for (j, slot) in k3.iter_mut().enumerate() {
                *slot = token_at(total - 3 + j);
            }
            if let Some(&end) = self.spans3.get(&k3) {
                let take = limit.min(history.len() - end);
                if take > 0 {
                    return (history[end..end + take].to_vec(), 3);
                }
            }
        }
        if min_match <= 2 && total >= 2 {
            let k2 = Self::key2(token_at(total - 2), token_at(total - 1));
            if let Some(&end) = self.spans2.get(&k2) {
                let take = limit.min(history.len() - end);
                if take > 0 {
                    return (history[end..end + take].to_vec(), 2);
                }
            }
        }
        if min_match <= 1 && total >= 1 {
            let k1 = token_at(total - 1);
            if let Some(&end) = self.spans1.get(&k1) {
                let take = limit.min(history.len() - end);
                if take > 0 {
                    return (history[end..end + take].to_vec(), 1);
                }
            }
        }
        (Vec::new(), 0)
    }
}

#[test]
fn suffix_lookup_matches_longest_latest_and_extends_hybrid_tail() {
    let mut s = SuffixLookup::default();
    let mut history: Vec<u32> = (0..100).collect();
    // Repeat 8-token prefix [10..18] followed by [900..920] so latest wins over earlier [18..33].
    history.extend(10..18);
    history.extend(900..920);
    // Query with tail &[17] after history ending with 10..17
    history.extend(10..17);
    let (toks, len) = s.propose_after(&history, &[17], 15, 2);
    assert_eq!(len, 8);
    assert_eq!(toks, (900..915).collect::<Vec<_>>());
    // Hybrid extension: neural drafted [900..907], extend from 907 using 8-token suffix.
    let (ext, ext_len) = s.propose_after(&history, &[17, 900, 901, 902, 903, 904, 905, 906], 8, 2);
    assert_eq!(ext_len, 8);
    assert_eq!(ext, (907..915).collect::<Vec<_>>());
    // Shorter 2-token suffix match fallback.
    let (short, short_len) = s.propose_after(&history, &[999, 998, 901, 902], 4, 2);
    assert_eq!(short_len, 2);
    assert_eq!(short, vec![903, 904, 905, 906]);
    // Truncation resets cleanly.
    let (empty, z) = s.propose_after(&[1], &[4], 15, 2);
    assert!(empty.is_empty() && z == 0);
}
