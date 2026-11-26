//! Host f32 reference for a compressed-pool sparse attention indexer.
//!
//! This module does not change the model plan or serving path. Reductions visit
//! dimensions, heads and pool slots in ascending order. Products and sums are
//! separate f32 operations. See `dsa/README.md` for source and operation limits.
use crate::{Result, invalid, json::{self, Value, field}, shapes::CONFIG_BYTES};

// The public indexer constructs LayerNorm with this epsilon, not rms_norm_eps.
const K_NORM_EPS: f32 = 1e-6;

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("DSA: {message}"))) }
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| invalid("DSA: dimension product overflow"))
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| invalid("DSA: dimension sum overflow"))
}
fn buffer(elements: usize, width: usize) -> Result<()> {
    need(mul(elements, width)? <= isize::MAX as usize, "buffer exceeds the address range")
}
fn finite(values: &[f32]) -> Result<()> {
    need(values.iter().all(|v| v.is_finite()), "values must be finite")
}

/// Geometry and selection budget. No model name is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub hidden_size: usize,
    pub q_lora_rank: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub index_kpool: usize,
    pub index_kpool_always_select_tail: bool,
}
impl Config {
    /// Read a text config or a root with `text_config`.
    /// The source defines compressed pools only. Reject an explicit false flag.
    pub fn from_json(input: &[u8]) -> Result<Self> {
        let doc = json::parse(input, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let o = root.get("text_config").map(Value::object).transpose()?.unwrap_or(root);
        let dim = |name| -> Result<usize> {
            usize::try_from(field(o, name)?.u64()?).map_err(|_| invalid("DSA: dimension exceeds usize"))
        };
        if let Some(v) = o.get("index_kpool_compress") {
            need(v.boolean()?, "uncompressed pools have no operation in this reference")?;
        }
        let d = Self { hidden_size: dim("hidden_size")?, q_lora_rank: dim("q_lora_rank")?,
            index_n_heads: dim("index_n_heads")?, index_head_dim: dim("index_head_dim")?,
            index_topk: dim("index_topk")?, index_kpool: dim("index_kpool")?,
            index_kpool_always_select_tail: field(o, "index_kpool_always_select_tail")?.boolean()? };
        d.validate()?;
        Ok(d)
    }
    pub fn validate(&self) -> Result<()> {
        let q_width = self.q_lora_rank;
        need(self.hidden_size > 0 && q_width > 0 && self.index_n_heads > 0
            && self.index_head_dim > 0 && self.index_topk > 0 && self.index_kpool > 0,
            "dimensions and selection budget must be positive")?;
        for count in [mul(mul(self.index_n_heads, self.index_head_dim)?, q_width)?,
            mul(self.index_head_dim, self.hidden_size)?, mul(self.index_n_heads, self.hidden_size)?,
            mul(self.index_kpool, self.index_head_dim)?] { buffer(count, size_of::<f32>())?; }
        buffer(add(self.index_topk, self.index_kpool - 1)?, size_of::<usize>())
    }
}

/// Decoded f32 weights. Matrices use row-major `[output][input]` order.
/// APE uses `[pool slot][key channel]` order. K norm has both gain and bias.
#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    pub wq_b: &'a [f32],
    pub wk: &'a [f32],
    pub weights_proj: &'a [f32],
    pub k_norm_weight: &'a [f32],
    pub k_norm_bias: &'a [f32],
    pub index_kpool_compress_ape: &'a [f32],
    pub index_kpool_compress_gate: &'a [f32],
}
#[derive(Debug, Clone, PartialEq)]
pub struct Projected {
    /// Query heads in `[head][channel]` order.
    pub query: Vec<f32>,
    /// Per-token key after LayerNorm and its affine transform.
    pub key: Vec<f32>,
    /// Projected head weights after multiplication by `1 / sqrt(head count)`.
    pub head_weights: Vec<f32>,
    /// Projected compression logits before addition of APE.
    pub gate: Vec<f32>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Pool {
    /// Contiguous raw token positions. All positions in a pool are valid.
    pub indices: Vec<usize>,
    pub key: Vec<f32>,
}
#[derive(Debug, Clone)]
struct Cached { key: Vec<f32>, gate: Vec<f32>, valid: bool }

/// One sequence and one layer. Keep the same weights for the cache lifetime.
/// Padding retains its raw slot. Pools start at the first valid slot.
#[derive(Debug, Clone)]
pub struct Cache { config: Config, rows: Vec<Cached> }
impl Cache {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, rows: Vec::new() })
    }
    pub fn len(&self) -> usize { self.rows.len() }
    pub fn is_empty(&self) -> bool { self.rows.is_empty() }
    pub fn push(&mut self, token: &Projected, valid: bool) -> Result<()> {
        check_projected(self.config, token)?;
        buffer(add(self.len(), 1)?, size_of::<Cached>())?;
        self.rows.try_reserve(1).map_err(|_| invalid("DSA: cannot reserve cache storage"))?;
        self.rows.push(Cached { key: token.key.clone(), gate: token.gate.clone(), valid });
        Ok(())
    }
}

/// Scalar reference for projection, compression and causal token selection.
pub struct Reference<'a> { config: Config, weights: Weights<'a> }
impl<'a> Reference<'a> {
    pub fn new(config: Config, weights: Weights<'a>) -> Result<Self> {
        config.validate()?;
        let (d, w) = (config, weights);
        for (values, count) in [(w.wq_b, d.index_n_heads * d.index_head_dim * d.q_lora_rank),
            (w.wk, d.index_head_dim * d.hidden_size), (w.weights_proj, d.index_n_heads * d.hidden_size),
            (w.k_norm_weight, d.index_head_dim), (w.k_norm_bias, d.index_head_dim),
            (w.index_kpool_compress_ape, d.index_kpool * d.index_head_dim),
            (w.index_kpool_compress_gate, d.index_head_dim * d.hidden_size)] {
            need(values.len() == count, "weight shape differs from config")?;
            finite(values)?;
        }
        Ok(Self { config, weights })
    }
    /// `q_resid` is the normalized low-rank Q input from the caller's MLA path.
    /// The source applies no rotary transform in this indexer.
    pub fn project(&self, x: &[f32], q_resid: &[f32]) -> Result<Projected> {
        let (d, w) = (self.config, self.weights);
        let q_width = d.q_lora_rank;
        need(x.len() == d.hidden_size && q_resid.len() == q_width, "input width differs from config")?;
        finite(x)?; finite(q_resid)?;
        let query = matvec(w.wq_b, q_resid)?;
        let mut key = matvec(w.wk, x)?;
        layer_norm(&mut key, w.k_norm_weight, w.k_norm_bias)?;
        let gate = matvec(w.index_kpool_compress_gate, x)?;
        let mut head_weights = matvec(w.weights_proj, x)?;
        let scale = 1.0 / (d.index_n_heads as f32).sqrt();
        for weight in &mut head_weights { *weight *= scale; }
        let token = Projected { query, key, head_weights, gate };
        check_projected(d, &token)?;
        Ok(token)
    }
    /// Return complete valid pools. A hole invalidates its whole pool.
    /// Causality is applied later, to each query's pool candidates.
    pub fn pools(&self, cache: &Cache) -> Result<Vec<Pool>> {
        need(cache.config == self.config, "cache config differs from reference")?;
        let d = self.config;
        let first = cache.rows.iter().position(|row| row.valid).unwrap_or(cache.len());
        let mut pools = Vec::new();
        for group in 0..(cache.len() - first) / d.index_kpool {
            let start = first + group * d.index_kpool;
            let rows = &cache.rows[start..start + d.index_kpool];
            if rows.iter().any(|row| !row.valid) { continue; }
            let mut key = vec![0.0; d.index_head_dim];
            let mut probabilities = vec![0.0; d.index_kpool];
            for (channel, value) in key.iter_mut().enumerate() {
                for (slot, row) in rows.iter().enumerate() {
                    probabilities[slot] = row.gate[channel]
                        + self.weights.index_kpool_compress_ape[slot * d.index_head_dim + channel];
                }
                softmax(&mut probabilities)?;
                for (slot, row) in rows.iter().enumerate() { *value += probabilities[slot] * row.key[channel]; }
            }
            finite(&key)?;
            pools.push(Pool { indices: (start..start + d.index_kpool).collect(), key });
        }
        Ok(pools)
    }
    /// Return valid raw token indices. Invalid padding entries are omitted.
    /// Pools sort by descending score, then lower first token index. Tokens in
    /// each pool keep their slot order. The optional visible tail is appended.
    /// Equal signed zeros are ties. Future pools and invalid queries select none.
    /// Treat this list as a set for sparse attention. The source's tail rule can
    /// repeat an index when padding has internal holes. See `dsa/README.md`.
    #[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
    pub fn select(&self, token: &Projected, cache: &Cache, position: usize) -> Result<Vec<usize>> {
        need(cache.config == self.config, "cache config differs from reference")?;
        need(position < cache.len(), "query position is outside the cache")?;
        check_projected(self.config, token)?;
        if !cache.rows[position].valid { return Ok(Vec::new()); }
        let d = self.config;
        let pools = self.pools(cache)?;
        let scale = 1.0 / (d.index_head_dim as f32).sqrt();
        let mut candidates = Vec::new();
        for (pool_id, pool) in pools.iter().enumerate() {
            if pool.indices[d.index_kpool - 1] > position { continue; }
            let mut score = 0.0f32;
            for head in 0..d.index_n_heads {
                let q = &token.query[head * d.index_head_dim..(head + 1) * d.index_head_dim];
                let mut dot = 0.0f32;
                for channel in 0..d.index_head_dim { dot += q[channel] * pool.key[channel]; }
                let scaled = dot * scale;
                need(scaled.is_finite(), "pool dot product is not finite")?;
                score += token.head_weights[head] * scaled.max(0.0);
            }
            need(score.is_finite(), "pool score is not finite")?;
            candidates.push((pool_id, score));
        }
        candidates.sort_by(|&(a, sa), &(b, sb)| {
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                .then(pools[a].indices[0].cmp(&pools[b].indices[0]))
        });
        candidates.truncate(d.index_topk / d.index_kpool);
        let mut selected = Vec::new();
        for (pool, _) in candidates { selected.extend_from_slice(&pools[pool].indices); }
        if d.index_kpool_always_select_tail {
            let first = cache.rows.iter().position(|row| row.valid).unwrap_or(cache.len());
            let visible = cache.rows[..=position].iter().filter(|row| row.valid).count();
            let tail = visible % d.index_kpool;
            let start = add(first, visible - tail)?;
            for index in start..add(start, tail)? {
                if index <= position && cache.rows[index].valid { selected.push(index); }
            }
        }
        Ok(selected)
    }
    /// Append a whole chunk, then select each row with its causal position.
    /// Inputs are row-major. Empty chunks are valid. An error restores the cache.
    pub fn run_chunk(&self, cache: &mut Cache, hidden: &[f32], q_resid: &[f32], valid: &[bool]) -> Result<Vec<Vec<usize>>> {
        need(cache.config == self.config, "cache config differs from reference")?;
        let d = self.config;
        need(hidden.len() == mul(valid.len(), d.hidden_size)?
            && q_resid.len() == mul(valid.len(), d.q_lora_rank)?, "chunk shape differs from config")?;
        buffer(add(cache.len(), valid.len())?, size_of::<Cached>())?;
        let tokens = hidden.chunks_exact(d.hidden_size).zip(q_resid.chunks_exact(d.q_lora_rank))
            .map(|(x, q)| self.project(x, q)).collect::<Result<Vec<_>>>()?;
        let prior = cache.len();
        let result: Result<Vec<Vec<usize>>> = (|| {
            for (token, &mask) in tokens.iter().zip(valid) { cache.push(token, mask)?; }
            tokens.iter().enumerate().map(|(row, token)| self.select(token, cache, prior + row)).collect()
        })();
        if result.is_err() { cache.rows.truncate(prior); }
        result
    }
}

fn check_projected(d: Config, token: &Projected) -> Result<()> {
    need(token.query.len() == d.index_n_heads * d.index_head_dim && token.key.len() == d.index_head_dim
        && token.gate.len() == d.index_head_dim && token.head_weights.len() == d.index_n_heads,
        "projected token shape differs from config")?;
    finite(&token.query)?; finite(&token.key)?; finite(&token.gate)?; finite(&token.head_weights)
}
fn matvec(matrix: &[f32], x: &[f32]) -> Result<Vec<f32>> {
    let output: Vec<_> = matrix.chunks_exact(x.len()).map(|row| {
        let mut sum = 0.0f32;
        for i in 0..x.len() { sum += row[i] * x[i]; }
        sum
    }).collect();
    finite(&output)?;
    Ok(output)
}
fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32]) -> Result<()> {
    let mut sum = 0.0f32;
    for &value in x.iter() { sum += value; }
    need(sum.is_finite(), "LayerNorm mean sum is not finite")?;
    let mean = sum / x.len() as f32;
    let mut variance = 0.0f32;
    for &value in x.iter() { let delta = value - mean; variance += delta * delta; }
    need(variance.is_finite(), "LayerNorm variance sum is not finite")?;
    let inverse = 1.0 / (variance / x.len() as f32 + K_NORM_EPS).sqrt();
    for (i, value) in x.iter_mut().enumerate() { *value = ((*value - mean) * inverse) * weight[i] + bias[i]; }
    finite(x)
}
fn softmax(values: &mut [f32]) -> Result<()> {
    finite(values)?;
    let mut max = f32::NEG_INFINITY;
    for &value in values.iter() { max = max.max(value); }
    let mut sum = 0.0f32;
    for value in values.iter_mut() { *value = (*value - max).exp(); sum += *value; }
    need(sum.is_finite() && sum > 0.0, "softmax sum is not positive and finite")?;
    for value in values { *value /= sum; }
    Ok(())
}

#[cfg(test)]
mod tests;
