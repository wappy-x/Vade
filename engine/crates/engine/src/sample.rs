//! Draws are keyed by seed and absolute position; replay consumes no RNG state.
use std::{collections::hash_map::RandomState, hash::BuildHasher};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    /// Zero disables top-k.
    pub top_k: usize,
    pub top_p: f64,
    /// Keeps tokens at least this fraction of the best token's probability; zero disables it.
    pub min_p: f64,
    pub repetition_penalty: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub seed: u64,
}
impl Sampling {
    pub fn qwen(thinking: bool) -> Self {
        Self {
            temperature: if thinking { 0.6 } else { 0.7 },
            top_k: 20,
            top_p: if thinking { 0.95 } else { 0.8 },
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: RandomState::new().hash_one(()),
        }
    }
    pub const fn greedy() -> Self {
        Self { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, repetition_penalty: 1.0,
            presence_penalty: 0.0, frequency_penalty: 0.0, seed: 0 }
    }
    /// GenerationConfig defaults match the checkpoint; requests override fields later.
    /// A checkpoint without this optional file uses the standard greedy defaults.
    pub fn from_checkpoint(root: &std::path::Path) -> Result<Self, String> {
        use engine_model::json::{self, Value};
        let path = root.join("generation_config.json");
        if !path.try_exists().map_err(|e| e.to_string())? { return Ok(Self::greedy()); }
        let bytes = engine_model::safetensors::read_bounded(&path, 1024 * 1024)
            .map_err(|e| e.to_string())?;
        let value = json::parse(&bytes, json::Limits::default()).map_err(|e| e.to_string())?;
        let object = value.object().map_err(|e| e.to_string())?;
        let mut sampling = Self { temperature: 1.0, top_k: 50, ..Self::greedy() };
        for (name, target) in [("temperature", &mut sampling.temperature), ("top_p", &mut sampling.top_p),
            ("min_p", &mut sampling.min_p), ("repetition_penalty", &mut sampling.repetition_penalty)] {
            if let Some(value) = object.get(name) { *target = value.f64().map_err(|e| e.to_string())?; }
        }
        if let Some(value) = object.get("top_k") {
            sampling.top_k = usize::try_from(value.u64().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        }
        sampling.validate()?;
        if !object.get("do_sample").map(Value::boolean).transpose().map_err(|e| e.to_string())?.unwrap_or(false) {
            sampling.temperature = 0.0;
        }
        Ok(sampling)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite() || self.temperature < 0.0
            || !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0
            || !(0.0..=1.0).contains(&self.min_p)
            || !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0
            || !self.presence_penalty.is_finite() || !(-2.0..=2.0).contains(&self.presence_penalty)
            || !self.frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&self.frequency_penalty)
        {
            return Err("invalid sampling parameters".into());
        }
        Ok(())
    }
    /// Compact device selection is exact only before history-dependent penalties.
    pub fn compact_top_k(&self) -> usize {
        if self.temperature > 0.0 && (1..=64).contains(&self.top_k)
            && self.repetition_penalty == 1.0 && self.presence_penalty == 0.0 && self.frequency_penalty == 0.0 {
            self.top_k
        } else { 0 }
    }
}
impl Default for Sampling {
    fn default() -> Self { Self::qwen(true) }
}

// SplitMix64 finalizer; high 53 bits give a uniform f64 in [0, 1).
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
/// Position-keyed draw in [0, 1), shared with speculative decision guards.
pub fn uniform(seed: u64, position: usize) -> f64 {
    (mix(seed ^ mix((position as u64).wrapping_add(0x9e3779b97f4a7c15))) >> 11) as f64
        * (1.0 / 9007199254740992.0)
}

/// Filtered support in the sampler's canonical order with its unnormalized weights. A draw
/// sums them exactly as the sampler does: draw(uniform(seed, position)) is its id there.
pub struct Distribution(pub Vec<(u32, f64)>);
impl Distribution {
    pub fn draw(&self, uniform: f64) -> Result<u32, String> {
        let mass: f64 = self.0.iter().map(|v| v.1).sum();
        if !mass.is_finite() || mass <= 0.0 { return Err("empty sampling distribution".into()); }
        let draw = uniform * mass;
        let mut cumulative = 0.0;
        for &(id, value) in &self.0 {
            cumulative += value;
            if draw < cumulative { return Ok(id); }
        }
        Ok(self.0.last().unwrap().0)
    }
    pub fn memory_bytes(&self) -> usize { self.0.capacity() * std::mem::size_of::<(u32, f64)>() }
}
fn order(a: &(u32, f64), b: &(u32, f64)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0))
}
/// After top-k and top-p, as in transformers. Weights are exp((logit - best) / T), so the
/// best token weighs exactly 1 and always stays; each test reads one weight, never a sum.
fn min_p(candidates: &mut Vec<(u32, f64)>, min_p: f64) {
    if min_p > 0.0 { candidates.retain(|v| v.1 >= min_p); }
}
/// Partitions by probability mass to size the nucleus without sorting the vocabulary, then
/// sorts only that prefix. The partition's sums only guide the size; nucleus decides it.
fn nucleus_guess(candidates: &mut [(u32, f64)], mut threshold: f64) -> usize {
    let (mut lo, mut hi) = (0, candidates.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        candidates[lo..hi].select_nth_unstable_by(mid - lo, order);
        let mass: f64 = candidates[lo..mid].iter().map(|v| v.1).sum();
        if mass >= threshold { hi = mid; }
        else if mass + candidates[mid].1 >= threshold { lo = mid + 1; break; }
        else { threshold -= mass + candidates[mid].1; lo = mid + 1; }
    }
    let lo = lo.max(1);
    candidates[..lo].sort_unstable_by(order);
    lo
}
/// The smallest prefix in descending order whose running sum reaches `threshold`, else all.
/// `candidates[..guess]` holds the largest weights in order. When rounding leaves that
/// short, the next largest weights join one at a time; after eight, the rest sorts at once.
fn nucleus(candidates: &mut [(u32, f64)], guess: usize, threshold: f64) -> usize {
    let (mut sorted, mut cumulative) = (guess, 0.0);
    for i in 0..candidates.len() {
        if i == sorted {
            if i >= guess + 8 { candidates[i..].sort_unstable_by(order); sorted = candidates.len(); }
            else {
                let next = candidates[i..].iter().enumerate().min_by(|a, b| order(a.1, b.1)).map_or(0, |v| v.0);
                candidates.swap(i, i + next);
                sorted += 1;
            }
        }
        cumulative += candidates[i].1;
        if cumulative >= threshold { return i + 1; }
    }
    candidates.len()
}

/// Request-owned scratch; options are validated once, buffers reused each draw.
pub struct Sampler {
    options: Sampling,
    candidates: Vec<(u32, f64)>,
    counts: Vec<(bool, u32)>,
    profile: bool,
    draws: usize,
}
impl Sampler {
    pub fn new(options: Sampling) -> Result<Self, String> {
        options.validate()?;
        Ok(Self { options, candidates: Vec::new(), counts: Vec::new(),
            profile: std::env::var("SPARK_PROFILE").as_deref() == Ok("1"), draws: 0 })
    }
    fn distribution_from_candidates(&self, selected: u32) -> Distribution {
        if self.options.temperature == 0.0 { return Distribution(vec![(selected, 1.0)]); }
        Distribution(self.candidates.clone())
    }
    pub fn distribution(&mut self, logits: &[f32], history: &[u32], prompt_len: usize,
        allowed: impl Fn(u32) -> bool) -> Result<Distribution, String> {
        // Reuse the exact filtering and accumulation used by plain sampling.
        let selected = self.sample_inner(logits, history, prompt_len, history.len(), allowed)?;
        Ok(self.distribution_from_candidates(selected))
    }
    pub fn distribution_top_k(&mut self, logits: &[(u32, f32)],
        allowed: impl Fn(u32) -> bool) -> Option<Distribution> {
        let selected = self.sample_top_k_inner(logits, 0, allowed)?;
        Some(self.distribution_from_candidates(selected))
    }
    /// Sorted, finite device top-k. None requests the original full-logit path.
    /// Nucleus membership defers to that path within a conservative rounding bound.
    /// Elsewhere its support is identical; the final sorted inverse CDF then
    /// executes precisely the full path's f64 math.
    pub fn sample_top_k(&mut self, logits: &[(u32, f32)], position: usize,
        allowed: impl Fn(u32) -> bool) -> Option<u32> {
        let start = self.profile.then(std::time::Instant::now);
        let result = self.sample_top_k_inner(logits, position, allowed);
        if result.is_some() && self.candidates.capacity() > 64 { self.candidates.shrink_to(64); }
        self.draws += 1;
        if let Some(start) = start.filter(|_| matches!(self.draws, 8 | 64 | 120)) {
            eprintln!("sample_profile path=top_k position={position} k={} wall_us={:.3} scratch_bytes={} fallback={}",
                logits.len(), start.elapsed().as_secs_f64()*1e6,
                self.candidates.capacity()*std::mem::size_of::<(u32, f64)>(), result.is_none());
        }
        result
    }
    fn sample_top_k_inner(&mut self, logits: &[(u32, f32)], position: usize,
        allowed: impl Fn(u32) -> bool) -> Option<u32> {
        let p = self.options;
        let k = p.compact_top_k();
        if k == 0 || logits.len() < k { return None; }
        let logits = &logits[..k];
        if logits.iter().any(|&(id, value)| !value.is_finite() || !allowed(id))
            || logits.windows(2).any(|w| w[0].1 < w[1].1 || w[0].1 == w[1].1 && w[0].0 >= w[1].0) {
            return None;
        }
        let best = logits[0].1 as f64;
        let candidates = &mut self.candidates;
        candidates.clear();
        candidates.extend(logits.iter().map(|&(id, value)| (id, ((value as f64 - best)/p.temperature).exp())));
        // Distinct logits can round to equal probabilities at large temperatures.
        candidates.sort_unstable_by(order);
        if p.top_p < 1.0 {
            let total: f64 = candidates.iter().map(|v| v.1).sum();
            let threshold = total * p.top_p;
            // Each positive sum has at most k terms. Near this bound the
            // full path decides, so both paths keep one nucleus.
            let error = 64.0 * f64::EPSILON * (k*k) as f64;
            let mut cumulative = 0.0;
            let mut keep = k;
            for (i, &(_, value)) in candidates.iter().enumerate() {
                cumulative += value;
                if (cumulative - threshold).abs() <= error { return None; }
                if cumulative > threshold { keep = i+1; break; }
            }
            candidates.truncate(keep);
        }
        min_p(candidates, p.min_p);
        let mass: f64 = candidates.iter().map(|v| v.1).sum();
        let draw = uniform(p.seed, position) * mass;
        let mut cumulative = 0.0;
        for &(id, value) in candidates.iter() {
            cumulative += value;
            if draw < cumulative { return Some(id); }
        }
        Some(candidates.last()?.0)
    }
    /// Repetition uses all history; presence/frequency use history[prompt_len..].
    /// Top-k precedes nucleus filtering; ties prefer the lowest token ID.
    pub fn sample(&mut self, logits: &[f32], history: &[u32], prompt_len: usize,
        position: usize, allowed: impl Fn(u32) -> bool) -> Result<u32, String>
    {
        let start = self.profile.then(std::time::Instant::now);
        let result = self.sample_inner(logits, history, prompt_len, position, allowed);
        self.draws += 1;
        if let Some(start) = start.filter(|_| matches!(self.draws, 8 | 64 | 120)) {
            eprintln!("sample_profile path=full position={position} vocab={} top_k={} wall_us={:.3} scratch_bytes={}",
                logits.len(), self.options.top_k, start.elapsed().as_secs_f64()*1e6,
                self.candidates.capacity()*std::mem::size_of::<(u32, f64)>()
                    + self.counts.capacity()*std::mem::size_of::<(bool, u32)>());
        }
        result
    }
    fn sample_inner(&mut self, logits: &[f32], history: &[u32], prompt_len: usize,
        position: usize, allowed: impl Fn(u32) -> bool) -> Result<u32, String>
    {
        let p = &self.options;
        if logits.is_empty() || logits.len() > u32::MAX as usize || prompt_len > history.len() {
            return Err("invalid vocabulary or prompt boundary".into());
        }
        let penalize = p.repetition_penalty != 1.0 || p.presence_penalty != 0.0 || p.frequency_penalty != 0.0;
        if penalize {
            self.counts.resize(logits.len(), (false, 0));
            self.counts.fill((false, 0));
            for (index, &id) in history.iter().enumerate() {
                let count = self.counts.get_mut(id as usize).ok_or("history token outside vocabulary")?;
                count.0 = true;
                count.1 += u32::from(index >= prompt_len);
            }
        }
        let candidates = &mut self.candidates;
        candidates.clear();
        if p.temperature != 0.0 { candidates.reserve_exact(logits.len()); }
        let mut best = (u32::MAX, f64::NEG_INFINITY);
        for (id, &logit) in logits.iter().enumerate() {
            if !logit.is_finite() { return Err("nonfinite logits".into()); }
            if !allowed(id as u32) { continue; }
            let mut value = logit as f64;
            if penalize {
                let (seen, count) = self.counts[id];
                if seen {
                    value = if value < 0.0 { value * p.repetition_penalty } else { value / p.repetition_penalty };
                }
                if count > 0 { value -= p.presence_penalty + p.frequency_penalty * count as f64; }
            }
            if !value.is_finite() { return Err("penalty overflow".into()); }
            if value > best.1 { best = (id as u32, value); }
            if p.temperature != 0.0 { candidates.push((id as u32, value)); }
        }
        if best.0 == u32::MAX { return Err("no allowed model tokens".into()); }
        if p.temperature == 0.0 { return Ok(best.0); }
        if p.top_k > 0 && p.top_k < candidates.len() {
            candidates.select_nth_unstable_by(p.top_k, order);
            candidates.truncate(p.top_k);
        }
        let mut total = 0.0;
        for (_, value) in candidates.iter_mut() {
            *value = ((*value - best.1) / p.temperature).exp();
            total += *value;
        }
        // Each sum and the draw follow a fixed order, never a partition's: descending weight
        // after a top-k cut and in the nucleus, token-id order over the whole vocabulary.
        if p.top_k > 0 {
            candidates.sort_unstable_by(order);
            total = candidates.iter().fold(0.0, |sum, v| sum + v.1);
        }
        if p.top_p < 1.0 {
            let threshold = total * p.top_p;
            let guess = if p.top_k > 0 { candidates.len() } else { nucleus_guess(candidates, threshold) };
            let keep = nucleus(candidates, guess, threshold);
            candidates.truncate(keep);
        }
        min_p(candidates, p.min_p);
        let mass: f64 = candidates.iter().map(|v| v.1).sum();
        let draw = uniform(p.seed, position) * mass;
        let mut cumulative = 0.0;
        for &(id, value) in candidates.iter() {
            cumulative += value;
            if draw < cumulative { return Ok(id); }
        }
        Ok(candidates.last().unwrap().0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn penalties_filtering_and_scratch_reuse() {
        let p = Sampling { presence_penalty: 2.0, frequency_penalty: 0.5, ..Sampling::greedy() };
        let mut s = Sampler::new(p).unwrap();
        assert_eq!(s.sample(&[4., 3., 2.], &[0, 0], 2, 2, |_| true).unwrap(), 0);
        assert_eq!(s.sample(&[4., 3., 2.], &[0, 0, 0], 2, 3, |_| true).unwrap(), 1);
        assert_eq!(s.sample(&[4., 3., 2.], &[0, 0], 2, 2, |_| true).unwrap(), 0);
        let counts = s.counts.as_ptr();
        s.sample(&[4., 3., 2.], &[0], 0, 1, |_| true).unwrap();
        assert_eq!(counts, s.counts.as_ptr());
        let mut s = Sampler::new(Sampling { repetition_penalty: 2.0, ..p }).unwrap();
        assert_eq!(s.sample(&[4., 3., -2.], &[0], 1, 1, |_| true).unwrap(), 1);
        assert_eq!(s.sample(&[-0., 0.], &[], 0, 0, |_| true).unwrap(), 0);
        assert!(s.sample(&[f32::NAN], &[], 0, 0, |_| false).is_err());
        assert!(s.sample(&[], &[], 0, 0, |_| true).is_err());
        assert!(s.sample(&[0.], &[], 0, 0, |_| false).is_err());
    }
    #[test]
    fn nucleus_matches_sorted_support_and_replays() {
        for flat in [false, true] {
            let logits: Vec<_> = (0..1024).map(|i| if flat { 0. } else { ((i * 79 % 997) as f32 - 500.) / 30. }).collect();
            for top_k in [0, 1, 20] {
                for top_p in [0.01, 0.8, 0.95, 1.0] {
                    let p = Sampling { top_k, top_p, seed: 42, ..Sampling::default() };
                    let mut s = Sampler::new(p).unwrap();
                    let serial: Vec<_> = (20..40).map(|pos| s.sample(&logits, &[], 0, pos, |id| id != 1).unwrap()).collect();
                    let ptr = s.candidates.as_ptr();
                    let replay: Vec<_> = (20..40).rev().map(|pos| s.sample(&logits, &[], 0, pos, |id| id != 1).unwrap()).collect();
                    assert!(serial.iter().eq(replay.iter().rev()));
                    assert_eq!(ptr, s.candidates.as_ptr());
                    let mut expected: Vec<_> = logits.iter().enumerate().filter(|(id, _)| *id != 1).map(|(id, &v)| (id as u32, v as f64)).collect();
                    expected.sort_unstable_by(order);
                    if top_k > 0 { expected.truncate(top_k); }
                    let max = expected[0].1;
                    for (_, v) in &mut expected { *v = ((*v - max) / p.temperature).exp(); }
                    let threshold = expected.iter().map(|v| v.1).sum::<f64>() * top_p;
                    let mut mass = 0.;
                    let keep = expected.iter().position(|v| { mass += v.1; mass >= threshold }).map_or(expected.len(), |i| i + 1);
                    if top_p < 1.0 { expected.truncate(keep); }
                    expected.sort_unstable_by_key(|v| v.0);
                    s.candidates.sort_unstable_by_key(|v| v.0);
                    assert!(s.candidates == expected, "nucleus differs: top_k={top_k} top_p={top_p}");
                }
            }
        }
    }
    /// Probabilities 0.5, 0.25, 0.15, 0.1 at temperature 1: weights to the best 1, 0.5, 0.3, 0.2.
    fn known() -> Vec<f32> { [0.5f64, 0.25, 0.15, 0.1].iter().map(|p| p.ln() as f32).collect() }
    #[test]
    fn top_k_top_p_and_min_p_on_known_logits() {
        let logits = known();
        let pairs: Vec<_> = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
        let base = Sampling { temperature: 1.0, seed: 5, ..Sampling::greedy() };
        let cases: &[(usize, f64, f64, &[u32])] = &[
            (0, 1.0, 0.0, &[0, 1, 2, 3]), (2, 1.0, 0.0, &[0, 1]), (1, 1.0, 0.0, &[0]),
            (0, 0.3, 0.0, &[0]), (0, 0.7, 0.0, &[0, 1]), (0, 0.8, 0.0, &[0, 1, 2]),
            (0, 1.0, 0.1, &[0, 1, 2, 3]), (0, 1.0, 0.25, &[0, 1, 2]), (0, 1.0, 0.4, &[0, 1]), (0, 1.0, 1.0, &[0]),
            (3, 0.8, 0.4, &[0, 1]), (4, 0.95, 0.25, &[0, 1, 2]),
        ];
        for &(top_k, top_p, min_p, expected) in cases {
            let p = Sampling { top_k, top_p, min_p, ..base };
            let mut s = Sampler::new(p).unwrap();
            let mut support: Vec<_> = s.distribution(&logits, &[], 0, |_| true).unwrap().0.iter().map(|v| v.0).collect();
            support.sort_unstable();
            assert_eq!(support, expected, "top_k={top_k} top_p={top_p} min_p={min_p}");
            // Every kept token is drawn and no other: the filters shape the draws, not only the support.
            let mut seen = std::collections::BTreeSet::new();
            for position in 0..512 {
                let id = s.sample(&logits, &[], 0, position, |_| true).unwrap();
                // The compact device top-k path draws the same id at the same position.
                if top_k > 0 { assert_eq!(s.sample_top_k(&pairs, position, |_| true), Some(id)); }
                seen.insert(id);
            }
            assert!(seen.iter().copied().eq(expected.iter().copied()), "draws {seen:?} for {expected:?}");
        }
        // min_p has no effect on greedy decoding.
        for min_p in [0.0, 0.5, 1.0] {
            let mut s = Sampler::new(Sampling { min_p, ..Sampling::greedy() }).unwrap();
            assert_eq!(s.sample(&[1., 3., 2.], &[], 0, 9, |_| true).unwrap(), 1);
        }
        for min_p in [-0.01, 1.01, f64::NAN] {
            assert!(Sampler::new(Sampling { min_p, ..base }).is_err());
        }
    }
    #[test]
    fn same_seed_gives_same_ids_on_any_instance_and_in_any_order() {
        let logits: Vec<f32> = (0..4096).map(|i| ((i * 37 % 1009) as f32 - 504.) / 90.).collect();
        let p = Sampling { temperature: 0.8, top_k: 40, top_p: 0.9, min_p: 0.05, seed: 20261029, ..Sampling::greedy() };
        let run = |s: &mut Sampler| (0..128).map(|position| s.sample(&logits, &[], 0, position, |_| true).unwrap()).collect::<Vec<_>>();
        let first = run(&mut Sampler::new(p).unwrap());
        assert_eq!(first, run(&mut Sampler::new(p).unwrap()));
        assert!(first.iter().collect::<std::collections::BTreeSet<_>>().len() > 1);
        // A sampler that served another seed keeps no state between draws.
        let mut reused = Sampler::new(Sampling { seed: 1, ..p }).unwrap();
        let other = run(&mut reused);
        assert_ne!(first, other);
        reused.options.seed = p.seed;
        assert_eq!(first, run(&mut reused));
        // Replay after a rejected draft draws positions out of order: each position keeps its id.
        let mut s = Sampler::new(p).unwrap();
        let reversed: Vec<_> = (0..128).rev().map(|position| s.sample(&logits, &[], 0, position, |_| true).unwrap()).collect();
        assert!(first.iter().eq(reversed.iter().rev()));
    }
    /// Host model of the tensor split's head: rank 0 holds rows [0, start), rank 1 the rest.
    /// A greedy pass merges the halves' argmax rows; a sampled pass runs the whole head on
    /// rank 0, and rank 1 only receives the chosen id as its next input.
    #[test]
    fn rank_simulation_split_head_and_rank_zero_sampling() {
        let vocabulary = 997;
        let start = crate::flash_head_options::split_rows(vocabulary);
        assert!((1..vocabulary).contains(&start));
        let argmax = |row: &[f32]| row.iter().enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i as u32, v) } else { best });
        let p = Sampling { temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.02, seed: 77, ..Sampling::greedy() };
        let run = || {
            let (mut rank0, mut again) = (Sampler::new(p).unwrap(), Sampler::new(p).unwrap());
            let mut greedy = Sampler::new(Sampling::greedy()).unwrap();
            let mut history = vec![3u32];
            for step in 0..64 {
                let whole: Vec<f32> = (0..vocabulary).map(|i| (((i * 131 + step * 17) % 997) as f32 - 498.) / 60.).collect();
                let (own, peer) = whole.split_at(start);
                let merged = crate::flash_head_options::merge_split(&[argmax(own)], &[argmax(peer)], start as u32).unwrap();
                assert_eq!(merged[0].0, greedy.sample(&whole, &history, 1, history.len(), |_| true).unwrap());
                let id = rank0.sample(&whole, &history, 1, history.len(), |_| true).unwrap();
                // Rows rebuilt from the two halves give the same draw on any rank that samples them.
                assert_eq!(id, again.sample(&[own, peer].concat(), &history, 1, history.len(), |_| true).unwrap());
                history.push(id);
            }
            history
        };
        let first = run();
        assert_eq!(first, run(), "a second run with the same seed");
        assert!(first[1..].iter().collect::<std::collections::BTreeSet<_>>().len() > 8);
    }
    /// Ids pinned from main 586d87947 before min_p existed: greedy, the checkpoint's sampling
    /// and penalized nucleus sampling draw the same ids with min_p unset. The nucleus row
    /// (top_k 0) is pinned again since its draw follows descending weight, not a partition.
    #[test]
    fn unset_min_p_keeps_main_ids() {
        let pins: [(Sampling, [u32; 32]); 3] = [
            (Sampling::greedy(), [1998, 1474, 950, 426, 3995, 3471, 2947, 2423, 1899, 1375, 851, 327, 3896, 3372, 2848, 2324,
                1800, 1276, 752, 228, 3797, 3273, 2749, 2225, 1701, 1177, 653, 129, 3698, 3174, 2650, 2126]),
            (Sampling { temperature: 1.0, top_k: 20, top_p: 0.95, seed: 20260905, ..Sampling::greedy() },
                [1319, 892, 2754, 3646, 3219, 3277, 2656, 3936, 1705, 1181, 2461, 1840, 1801, 2790, 559, 1548,
                3313, 1276, 2265, 3545, 1702, 984, 1876, 4029, 3602, 3078, 556, 2030, 1215, 3174, 4066, 2029]),
            (Sampling { temperature: 0.8, top_p: 0.9, repetition_penalty: 1.1, presence_penalty: 0.3, seed: 11, ..Sampling::greedy() },
                [2598, 3045, 792, 2695, 464, 1762, 869, 2809, 1414, 2984, 3469, 2693, 3235, 986, 2634, 2613,
                2303, 3177, 3351, 1178, 2788, 3079, 130, 1255, 3194, 1176, 12, 2728, 476, 2805, 242, 2940]),
        ];
        for (p, pin) in pins {
            assert_eq!(p.min_p, 0.0);
            let mut s = Sampler::new(p).unwrap();
            let mut history = vec![5u32, 9];
            for step in 0..32usize {
                let logits: Vec<f32> = (0..4093usize).map(|i| (((i * 211 + step * 53) % 4093) as f32 - 2046.) / 400.).collect();
                let id = s.sample(&logits, &history, 2, history.len(), |id| id != 7).unwrap();
                history.push(id);
            }
            assert_eq!(history[2..], pin, "{p:?}");
        }
    }
    /// The sampling rule with no partition: whole-vocabulary sums run in token-id order; after
    /// a top-k cut, and for the nucleus and the draw, weights run in descending order.
    fn reference(logits: &[f32], p: Sampling, position: usize, allowed: impl Fn(u32) -> bool) -> u32 {
        let mut c: Vec<_> = logits.iter().enumerate().filter(|v| allowed(v.0 as u32)).map(|(id, &v)| (id as u32, v as f64)).collect();
        let best = c.iter().map(|v| v.1).fold(f64::NEG_INFINITY, f64::max);
        if p.temperature == 0.0 { return c.iter().find(|v| v.1 == best).unwrap().0; }
        if p.top_k > 0 { c.sort_by(order); c.truncate(p.top_k); }
        for v in &mut c { v.1 = ((v.1 - best) / p.temperature).exp(); }
        if p.top_k > 0 { c.sort_by(order); }
        let mut total = 0.0;
        for v in &c { total += v.1; }
        if p.top_p < 1.0 {
            c.sort_by(order);
            let mut cumulative = 0.0;
            let keep = c.iter().position(|v| { cumulative += v.1; cumulative >= total * p.top_p }).map_or(c.len(), |i| i + 1);
            c.truncate(keep);
        }
        c.retain(|v| v.1 >= p.min_p);
        let mut mass = 0.0;
        for v in &c { mass += v.1; }
        let draw = uniform(p.seed, position) * mass;
        let mut cumulative = 0.0;
        for v in &c { cumulative += v.1; if draw < cumulative { return v.0; } }
        c.last().unwrap().0
    }
    /// top_k 0 with top_p below 1. With top_k equal to the row length the same tokens stay, so on
    /// these rows both settings draw the same id; their top_p sums run in other orders and can
    /// round apart elsewhere. The audit's five-token row passes
    /// on main too: the partition sorts slices this short. The 68-token row exposes the gap.
    #[test]
    fn nucleus_without_top_k_draws_in_descending_order() {
        let rows: [(Vec<f32>, Sampling); 2] = [
            (vec![1.0, 5.0, 2.0, 4.0, 3.0], Sampling { temperature: 0.8, top_k: 0, top_p: 0.9, seed: 7, ..Sampling::greedy() }),
            ((0..68).map(|i| ((i * 13 + 5) % 97) as f32 / 8.0).collect(),
                Sampling { temperature: 1.0, top_k: 0, top_p: 0.9, seed: 0, ..Sampling::greedy() }),
        ];
        for (logits, p) in rows {
            let (mut s, mut cut) = (Sampler::new(p).unwrap(), Sampler::new(Sampling { top_k: logits.len(), ..p }).unwrap());
            let mismatch: Vec<_> = (0..64).filter_map(|position| {
                let (id, expected) = (s.sample(&logits, &[], 0, position, |_| true).unwrap(), reference(&logits, p, position, |_| true));
                assert_eq!(expected, cut.sample(&logits, &[], 0, position, |_| true).unwrap());
                (id != expected).then_some((position, id, expected))
            }).collect();
            assert!(mismatch.is_empty(), "{} of 64 positions differ on {} tokens, first {:?}", mismatch.len(), logits.len(), mismatch.first());
        }
    }
    /// Random rows, ties, filters and settings: the fast path draws the reference's id.
    #[test]
    fn fast_path_matches_the_sorted_reference() {
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = move || { state = mix(state.wrapping_add(0x9e3779b97f4a7c15)); state };
        let mut checked = 0;
        for case in 0..3000 {
            let n = if case % 50 == 0 { 4000 + (next() % 4000) as usize } else { 1 + (next() % 300) as usize };
            let (spread, steps) = ([0.5, 4.0, 20.0, 60.0][case % 4], [0, 3, 64][next() as usize % 3]);
            let logits: Vec<f32> = (0..n).map(|_| {
                let u = (next() >> 11) as f64 / 9007199254740992.0;
                (if steps == 0 { u * spread } else { (u * steps as f64).floor() * spread / steps as f64 }) as f32
            }).collect();
            let top_k = [0, 0, 1, 2, 7, 64, n, n + 3][next() as usize % 8];
            let p = Sampling { temperature: [0.05, 0.3, 0.8, 1.0, 1.9][next() as usize % 5], top_k,
                top_p: [0.05, 0.5, 0.9, 0.95, 0.999, 1.0][next() as usize % 6], min_p: [0.0, 0.0, 0.01, 0.3][next() as usize % 4],
                seed: next(), ..Sampling::greedy() };
            let banned = next() as u32 % (n as u32 + 1);
            let allowed = |id: u32| n == 1 || id != banned;
            let mut s = Sampler::new(p).unwrap();
            for position in [0, 1, 17, (next() % 100000) as usize] {
                let id = s.sample(&logits, &[], 0, position, allowed).unwrap();
                assert_eq!(id, reference(&logits, p, position, allowed), "case {case} position {position} {p:?} n {n}");
                checked += 1;
            }
        }
        assert_eq!(checked, 12000);
    }
    /// A sampled draft's distributions draw the sampler's own id at the same seed and position.
    #[test]
    fn distribution_draws_the_sampled_id() {
        let mut state = 5u64;
        for case in 0..400 {
            let n = 2 + case % 90;
            let logits: Vec<f32> = (0..n).map(|_| { state = mix(state.wrapping_add(1)); (state % 1000) as f32 / 97.0 }).collect();
            let pairs = { let mut v: Vec<_> = logits.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
                v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))); v };
            let p = Sampling { temperature: [0.4, 1.0, 1.7][case % 3], top_k: [0, 3, 20][case % 5 % 3], top_p: [1.0, 0.9, 0.3][case % 7 % 3],
                min_p: [0.0, 0.1][case % 2], seed: state, ..Sampling::greedy() };
            let mut s = Sampler::new(p).unwrap();
            for position in [0, 9, 4096] {
                let history = vec![1u32; position];
                let id = s.sample(&logits, &history, position, position, |_| true).unwrap();
                assert_eq!(s.distribution(&logits, &history, position, |_| true).unwrap().draw(uniform(p.seed, position)), Ok(id));
                if let Some(id) = s.sample_top_k(&pairs, position, |_| true) {
                    assert_eq!(s.distribution_top_k(&pairs, |_| true).unwrap().draw(uniform(p.seed, position)), Ok(id));
                }
            }
        }
    }
    /// A guess that falls short adds the next largest weights one at a time, then sorts the rest.
    #[test]
    fn nucleus_extends_a_short_guess() {
        let row: Vec<(u32, f64)> = (0..300u32).map(|id| (id, ((id * 37 % 101) as f64 + 1.0) / 64.0)).collect();
        let mut sorted = row.clone();
        sorted.sort_by(order);
        let total: f64 = row.iter().map(|v| v.1).sum();
        for top_p in [0.001, 0.02, 0.1, 0.5, 0.97, 1.0] {
            let mut cumulative = 0.0;
            let keep = sorted.iter().position(|v| { cumulative += v.1; cumulative >= total * top_p }).map_or(row.len(), |i| i + 1);
            for guess in [0, 1, 5, 40] {
                let mut c = row.clone();
                c[..guess].copy_from_slice(&sorted[..guess]);
                let mut rest = row.iter().filter(|v| !sorted[..guess].contains(v)).copied().collect::<Vec<_>>();
                rest.reverse();
                c[guess..].copy_from_slice(&rest);
                assert_eq!(nucleus(&mut c, guess, total * top_p), keep, "top_p {top_p} guess {guess}");
                assert!(c[..keep] == sorted[..keep]);
            }
        }
    }
    /// Greedy draws stay as on main a830f96e6, with and without penalties and a token filter.
    #[test]
    fn greedy_ids_unchanged() {
        let varying = |step: usize| (0..4093usize).map(|i| (((i * 211 + step * 53) % 4093) as f32 - 2046.) / 400.
            + if (i + step) % 97 == 0 { 0.4 } else { 0.0 }).collect::<Vec<f32>>();
        let constant = |_: usize| (0..4093usize).map(|i| (((i * 211) % 4093) as f32 - 2046.) / 400.).collect::<Vec<f32>>();
        let penalized = Sampling { repetition_penalty: 1.3, presence_penalty: 0.5, frequency_penalty: 0.25, ..Sampling::greedy() };
        let pins: [(Sampling, &dyn Fn(usize) -> Vec<f32>, [u32; 32]); 4] = [
            (Sampling::greedy(), &varying, [1998, 4073, 950, 426, 3995, 3471, 2947, 2423, 1899, 1375, 851, 327, 3896, 3372, 2848, 2324,
                1800, 1276, 752, 2226, 3797, 3273, 2749, 2225, 1701, 3175, 653, 4047, 3698, 3174, 2650, 31]),
            (penalized, &varying, [1998, 4073, 950, 426, 3995, 3471, 2947, 2423, 1899, 1375, 851, 327, 3896, 3372, 2848, 2324,
                1800, 1276, 752, 2226, 3797, 3273, 2749, 2225, 1701, 3175, 653, 4047, 3698, 3174, 2650, 31]),
            (Sampling::greedy(), &constant, [1998; 32]),
            (penalized, &constant, [1998, 3996, 1901, 3899, 1804, 3802, 1707, 3705, 1610, 1513, 3511, 1416, 3414, 1319, 3317, 1222,
                3220, 3123, 1028, 3026, 931, 2929, 834, 2832, 737, 2735, 640, 2638, 543, 2541, 446, 2444]),
        ];
        for (p, row, pin) in pins {
            let mut s = Sampler::new(p).unwrap();
            let mut history = vec![5u32, 9];
            for step in 0..32 {
                let id = s.sample(&row(step), &history, 2, history.len(), |id| id % 13 != 7).unwrap();
                history.push(id);
            }
            assert_eq!(history[2..], pin, "{p:?}");
        }
    }
    /// Without top_p below 1, the sampled paths also keep main's ids.
    #[test]
    fn uncut_and_top_k_sampling_ids_unchanged() {
        let pins: [(Sampling, [u32; 32]); 2] = [
            (Sampling { temperature: 0.9, min_p: 0.02, seed: 3, ..Sampling::greedy() },
                [424, 3200, 289, 676, 287, 2420, 2710, 1739, 2514, 1737, 3779, 3256, 2730, 1838, 1469, 4046,
                616, 170, 3447, 634, 945, 596, 672, 1991, 2242, 3815, 265, 2689, 554, 844, 3268, 3752]),
            (Sampling { temperature: 0.7, top_k: 40, min_p: 0.05, seed: 4, ..Sampling::greedy() },
                [2929, 2909, 2811, 2422, 2809, 1741, 2225, 1157, 962, 3968, 2706, 3578, 2219, 84, 1732, 82,
                539, 112, 3778, 771, 1857, 3273, 654, 1546, 2729, 1235, 459, 2010, 2106, 2590, 3559, 939]),
        ];
        for (p, pin) in pins {
            let mut s = Sampler::new(p).unwrap();
            let mut history = vec![5u32, 9];
            for step in 0..32usize {
                let logits: Vec<f32> = (0..4093usize).map(|i| (((i * 211 + step * 53) % 4093) as f32 - 2046.) / 400.
                    + if (i + step) % 97 == 0 { 6.0 } else { 0.0 }).collect();
                history.push(s.sample(&logits, &history, 2, history.len(), |id| id % 13 != 7).unwrap());
            }
            assert_eq!(history[2..], pin, "{p:?}");
        }
    }
}
