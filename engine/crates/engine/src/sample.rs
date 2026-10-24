//! Draws are keyed by seed and absolute position; replay consumes no RNG state.
use std::{collections::hash_map::RandomState, hash::BuildHasher};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    /// Zero disables top-k.
    pub top_k: usize,
    pub top_p: f64,
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
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: RandomState::new().hash_one(()),
        }
    }
    pub const fn greedy() -> Self {
        Self { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0,
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
            ("repetition_penalty", &mut sampling.repetition_penalty)] {
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
/// Independent proposal, acceptance and target streams for speculative sampling.
pub fn speculative_uniform(seed: u64, position: usize, stream: u64) -> f64 {
    uniform(mix(seed ^ (stream + 1).wrapping_mul(0xd1b54a32d192ed03)), position)
}

/// Filtered, normalized support in the sampler's canonical arithmetic order.
pub struct Distribution(pub Vec<(u32, f64)>);
impl Distribution {
    pub fn probability(&self, token: u32) -> f64 {
        self.0.iter().find(|v| v.0 == token).map_or(0.0, |v| v.1)
    }
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
    pub fn residual(&self, draft: &Self) -> Self {
        let mut draft = draft.0.clone();
        draft.sort_unstable_by_key(|v| v.0);
        Self(self.0.iter().map(|&(id, p)| {
            let q = draft.binary_search_by_key(&id, |v| v.0).map_or(0.0, |i| draft[i].1);
            (id, (p - q).max(0.0))
        })
            .filter(|v| v.1 > 0.0).collect())
    }
    pub fn memory_bytes(&self) -> usize { self.0.capacity() * std::mem::size_of::<(u32, f64)>() }
}
fn order(a: &(u32, f64), b: &(u32, f64)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0))
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
        let mass: f64 = self.candidates.iter().map(|v| v.1).sum();
        Distribution(self.candidates.iter().map(|&(id, value)| (id, value / mass)).collect())
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
    /// Only nucleus membership depends on the old partition's summation order.
    /// Away from a conservative rounding bound its support is identical; the
    /// final sorted inverse CDF then executes precisely the original f64 math.
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
            // Each positive sum has at most k terms. This also covers every
            // subtraction and partial sum in the legacy nucleus partition.
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
        // Partition by probability mass to find the exact nucleus without sorting
        // the vocabulary. Only the retained top-k set needs ordering for replay.
        if p.top_p < 1.0 {
            let mut threshold = total * p.top_p;
            let (mut lo, mut hi) = (0, candidates.len());
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                candidates[lo..hi].select_nth_unstable_by(mid - lo, order);
                let mass: f64 = candidates[lo..mid].iter().map(|v| v.1).sum();
                if mass >= threshold { hi = mid; }
                else if mass + candidates[mid].1 >= threshold { lo = mid + 1; break; }
                else { threshold -= mass + candidates[mid].1; lo = mid + 1; }
            }
            candidates.truncate(lo.max(1));
        }
        if p.top_k > 0 { candidates.sort_unstable_by(order); }
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
}
