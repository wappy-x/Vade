//! Proposal-only copy lookup and depth control; the target still owns every token.
use std::collections::{HashMap, VecDeque};

/// Distributed sessions cannot use rank-local timing to shorten proposals.
pub fn default_depth(distributed: bool) -> usize { if distributed { 2 } else { 3 } }

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options { pub copy_first: bool, pub gated_depth: bool }
impl Options {
    /// Snapshot on the first hybrid startup/sizing call; all later users share it.
    pub fn startup() -> Self {
        static OPTIONS: std::sync::OnceLock<Options> = std::sync::OnceLock::new();
        *OPTIONS.get_or_init(|| {
            let on = |name| std::env::var(name).map_or(true, |v| v == "1");
            Self { copy_first: on("HYBRID_MTP_COPY"), gated_depth: on("HYBRID_MTP_GATE") }
        })
    }
    pub fn words(self) -> [u64; 2] { [u64::from(self.copy_first), u64::from(self.gated_depth)] }
}

const COPY_SPAN_CAP: usize = 4096;
/// Exact keys avoid hash-collision proposals. Keep only the newest capped spans;
/// the selected next token completes the query suffix of committed history.
#[derive(Default)]
pub struct CopyIndex { spans: HashMap<[u32; 8], usize>, oldest: VecDeque<([u32; 8], usize)>, seen: usize }
impl CopyIndex {
    /// The pinned std table can reach 4*cap buckets after tombstone churn:
    /// below half-full it rebuilds rather than growing again. Fund two tables
    /// (41 bytes/bucket) plus the 2*cap queue (40 bytes/entry), with slack.
    pub fn reservation_bytes(tokens: usize, options: Options) -> Option<usize> {
        if !options.copy_first { return Some(0); }
        tokens.min(2 * COPY_SPAN_CAP + 8).checked_mul(256)?.checked_add(1024)
    }
    /// Notify before replacing a committed suffix, even if it regrows before propose.
    pub fn truncate(&mut self, position: usize) {
        if position < self.seen { self.spans.clear(); self.oldest.clear(); self.seen = 0; }
    }
    pub fn propose(&mut self, history: &[u32], next: u32, limit: usize) -> Vec<u32> {
        self.truncate(history.len());
        for end in self.seen.max(8)..history.len() {
            let key = history[end-8..end].try_into().unwrap();
            self.spans.insert(key, end);
            self.oldest.push_back((key, end));
            if self.oldest.len() > COPY_SPAN_CAP {
                let (key, end) = self.oldest.pop_front().unwrap();
                if self.spans.get(&key) == Some(&end) { self.spans.remove(&key); }
            }
        }
        self.seen = history.len();
        if history.len() < 7 || limit == 0 { return Vec::new(); }
        let mut key = [0; 8];
        key[..7].copy_from_slice(&history[history.len()-7..]); key[7] = next;
        self.spans.get(&key).map_or_else(Vec::new, |&end|
            history[end..end + limit.min(history.len()-end)].to_vec())
    }
}

/// Top-16 softmax share, used only to decide whether to pay for another level.
pub fn confidence(scores: &[f32]) -> f64 {
    let Some(&best) = scores.first() else { return 0.0; };
    if scores.iter().any(|v| !v.is_finite()) { return 0.0; }
    1.0 / scores.iter().map(|&v| ((v as f64)-(best as f64)).exp()).sum::<f64>()
}

/// `rate` is measured accepted tokens/ms, so the comparison is dimensionless.
/// Missing timings warm up with the configured depth. Nonfinite evidence stops.
pub fn more(cumulative: f64, rate: f64, row_ms: f64, level_ms: f64) -> bool {
    if ![cumulative, rate, row_ms, level_ms].iter().all(|v| v.is_finite()) { return false; }
    if rate <= 0.0 || level_ms <= 0.0 { return true; }
    cumulative * cumulative >= rate * (row_ms.max(0.0) + level_ms)
}

/// Allocation bound for the existing GPU top-k reduction, one MTP row, k=16.
pub fn confidence_bytes(vocab: usize, options: Options) -> usize {
    if !options.gated_depth { return 0; }
    ((vocab.div_ceil(1024)*48).next_power_of_two()+64)*4
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn copy_index_reservation_is_capped_and_disabled_copy_costs_nothing() {
        let on = Options { copy_first: true, gated_depth: false };
        assert_eq!(CopyIndex::reservation_bytes(0, on), Some(1024));
        assert_eq!(CopyIndex::reservation_bytes(usize::MAX, Options::default()), Some(0));
        for tokens in [2*COPY_SPAN_CAP+8, 262144, usize::MAX] {
            assert_eq!(CopyIndex::reservation_bytes(tokens, on), Some((2*COPY_SPAN_CAP+8)*256+1024));
        }
    }
}

#[cfg(test)]
mod copy_bound_tests {
    use super::*;
    #[test]
    fn evicts_oldest_spans_but_keeps_refreshed_keys() {
        let mut index = CopyIndex::default();
        let mut history: Vec<u32> = (0..COPY_SPAN_CAP as u32+20).collect();
        index.propose(&history, 0, 4);
        assert_eq!(index.oldest.len(), COPY_SPAN_CAP);
        assert_eq!(index.spans.len(), COPY_SPAN_CAP);
        assert!(!index.spans.contains_key(&[0,1,2,3,4,5,6,7]));
        history.extend([0; COPY_SPAN_CAP+8]);
        index.propose(&history, 0, 4);
        assert_eq!(index.oldest.len(), COPY_SPAN_CAP);
        assert_eq!(index.spans.len(), 1);
        assert_eq!(index.spans.get(&[0;8]), Some(&(history.len()-1)));
        assert_eq!(index.propose(&history,0,4), [0]);
        index.truncate(0);
        assert!(index.spans.is_empty() && index.oldest.is_empty());
    }
}
