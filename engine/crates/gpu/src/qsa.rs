//! FP32 QSA oracle for contiguous, unpadded text. Projection/RoPE tables belong
//! to the caller; visibility is the causal prefix. No BF16/FP8 quantization.
#[allow(clippy::duplicate_mod)] // the shared file is included on purpose.
#[path = "../kernels/src/qsa_common.rs"]
pub mod contract;
use contract::*;

#[derive(Clone)]
pub struct IndexCache {
    pub keys: Vec<[f32; DIM]>,
    ring: [[f32; DIM]; RATIO], first_rope: Rope, tokens: usize,
}
impl Default for IndexCache {
    fn default() -> Self { Self { keys: Vec::new(), ring: [[0.0; DIM]; RATIO], first_rope: [0.0; 64], tokens: 0 } }
}
impl IndexCache {
    pub fn tokens(&self) -> usize { self.tokens }
    /// Append pre-projected raw K. Complete groups use their first token's RoPE.
    pub fn append(&mut self, raw: &[f32; DIM], weight: &[f32; DIM], rope: &Rope, eps: f32) -> Result<(), String> {
        if self.tokens == LIMIT || !eps.is_finite() || eps <= 0.0 { return Err("invalid QSA append".into()); }
        let slot = self.tokens % RATIO;
        if slot == 0 { self.first_rope = *rope; }
        self.ring[slot] = *raw;
        self.tokens += 1;
        if self.tokens.is_multiple_of(RATIO) {
            let mut mean = [0.0; DIM];
            for k in self.ring { for i in 0..DIM { mean[i] += k[i]; } }
            for x in &mut mean { *x /= RATIO as f32; }
            self.keys.push(normalize_rotate(mean, weight, &self.first_rope, eps));
        }
        Ok(())
    }
    /// Scores/selection for a historical causal row, including its current token.
    pub fn decode(&self, query: &[f32; HEADS * DIM], position: usize) -> Result<(Vec<f32>, Vec<u32>), String> {
        if position >= self.tokens { return Err("QSA position outside cache".into()); }
        select_blocks(query, &self.keys, position + 1)
    }
    pub fn prefill(&self, queries: &[[f32; HEADS * DIM]], past: usize) -> Result<Vec<Vec<u32>>, String> {
        if past.checked_add(queries.len()).is_none_or(|n| n > self.tokens) { return Err("QSA prefill outside cache".into()); }
        queries.iter().enumerate().map(|(r, q)| self.decode(q, past + r).map(|(_, s)| s)).collect()
    }
}
pub fn select_blocks(query: &[f32; HEADS * DIM], keys: &[[f32; DIM]], visible: usize) -> Result<(Vec<f32>, Vec<u32>), String> {
    if visible == 0 || visible > LIMIT || visible / RATIO > keys.len() { return Err("invalid QSA visibility".into()); }
    let complete = visible / RATIO;
    let scores: Vec<_> = keys[..complete].iter().map(|k| score(query, k)).collect();
    if scores.iter().any(|s| !s.is_finite()) { return Err("nonfinite QSA score".into()); }
    let mut blocks: Vec<_> = (0..complete).collect();
    blocks.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    blocks.truncate(BUDGET / RATIO);
    blocks.sort_unstable();
    let selected = blocks.into_iter().flat_map(|b| b * RATIO..(b + 1) * RATIO)
        .chain(complete * RATIO..visible).map(|i| i as u32).collect();
    Ok((scores, selected))
}
pub fn prepare_query(q: &[[f32; DIM]; HEADS], weight: &[f32; DIM], rope: &Rope, eps: f32) -> [f32; HEADS * DIM] {
    let mut out = [0.0; HEADS * DIM];
    for h in 0..HEADS { out[h * DIM..(h + 1) * DIM].copy_from_slice(&normalize_rotate(q[h], weight, rope, eps)); }
    out
}
/// Normal causal GQA restricted to ascending selected positions; all sums are
/// feature/position ordered. The caller applies the main attention output gate.
pub fn sparse_attention(q: &[f32], keys: &[f32], values: &[f32], selected: &[u32]) -> Result<Vec<f32>, String> {
    if q.len() != 24 * 256 || keys.len() != values.len() || !keys.len().is_multiple_of(512)
        || selected.is_empty() || selected.len() > WIDTH || selected.windows(2).any(|s| s[0] >= s[1])
        || selected.last().is_some_and(|&p| p as usize >= keys.len() / 512) { return Err("invalid QSA attention shape".into()); }
    let mut out = vec![0.0; q.len()];
    for h in 0..24 {
        let mut weights: Vec<_> = selected.iter().map(|&p| {
            let start = p as usize * 512 + h / 12 * 256;
            dot(&q[h * 256..(h + 1) * 256], &keys[start..start + 256]) * (1.0 / 16.0)
        }).collect();
        let highest = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0;
        for w in &mut weights { *w = exponential(*w - highest); denominator += *w; }
        for w in &mut weights { *w /= denominator; }
        for (i, &p) in selected.iter().enumerate() { for d in 0..256 {
            out[h * 256 + d] += weights[i] * values[p as usize * 512 + h / 12 * 256 + d];
        } }
    }
    Ok(out)
}
#[derive(Debug, PartialEq, Eq)]
pub struct CacheBytes { pub indexer: usize, pub kv: usize, pub fixed: usize }
impl CacheBytes {
    /// Twelve layers; index representatives FP32, main K/V element size explicit.
    /// Fixed includes four raw keys, first-group RoPE and a u64 cursor per layer.
    pub fn for_context(tokens: usize, kv_element_bytes: usize) -> Result<Self, String> {
        if tokens > LIMIT || ![2, 4].contains(&kv_element_bytes) { return Err("invalid QSA memory shape".into()); }
        Ok(Self { indexer: 12 * (tokens / RATIO) * DIM * 4,
            kv: 12 * tokens * 2 * 2 * 256 * kv_element_bytes, fixed: 12 * (RATIO * DIM * 4 + 64 * 4 + 8) })
    }
    pub fn total(&self) -> usize { self.indexer + self.kv + self.fixed }
}
