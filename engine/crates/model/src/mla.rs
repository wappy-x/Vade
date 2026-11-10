//! Host f32 reference for multi-head latent attention (MLA).
//!
//! This module does not admit a checkpoint or change the serving path. Each
//! reduction visits dimensions or tokens in ascending order. Products and sums
//! are separate f32 operations. The host math library supplies sqrt and exp.
//! Equal bits here do not establish equal bits on another target or on a GPU.
use crate::{Result, invalid, json::{self, Value, field}, shapes::CONFIG_BYTES};

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("MLA: {message}"))) }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| invalid("MLA: dimension sum overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| invalid("MLA: dimension product overflow"))
}
fn bytes(elements: usize) -> Result<usize> {
    let n = mul(elements, std::mem::size_of::<f32>())?;
    need(n <= isize::MAX as usize, "buffer exceeds the address range")?;
    Ok(n)
}
fn finite(x: &[f32]) -> Result<()> {
    need(x.iter().all(|v| v.is_finite()), "values must be finite")
}

/// Dimensions of the low-rank Q and KV projections. No model name is needed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub num_attention_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rms_norm_eps: f32,
}
/// Logical f32 payload per token, per attention layer. Allocator overhead is excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheBytes { pub latent: usize, pub expanded: usize }
impl Config {
    /// Read geometry from a text config or from its enclosing `text_config`.
    /// This is not a model plan parser. RoPE tables and decoded weights are inputs
    /// to the reference. Other checkpoint features are not enabled by this call.
    pub fn from_json(input: &[u8]) -> Result<Self> {
        let doc = json::parse(input, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let o = root.get("text_config").map(Value::object).transpose()?.unwrap_or(root);
        let dim = |name| -> Result<usize> {
            usize::try_from(field(o, name)?.u64()?).map_err(|_| invalid("MLA: dimension exceeds usize"))
        };
        let d = Self { hidden_size: dim("hidden_size")?, q_lora_rank: dim("q_lora_rank")?,
            kv_lora_rank: dim("kv_lora_rank")?, num_attention_heads: dim("num_attention_heads")?,
            qk_nope_head_dim: dim("qk_nope_head_dim")?, qk_rope_head_dim: dim("qk_rope_head_dim")?,
            v_head_dim: dim("v_head_dim")?, rms_norm_eps: field(o, "rms_norm_eps")?.f64()? as f32 };
        d.validate()?;
        if let Some(v) = o.get("qk_head_dim") {
            need(v.u64()? == d.qk_dim() as u64, "qk_head_dim differs from the head split")?;
        }
        if let Some(v) = o.get("attention_bias") {
            need(!v.boolean()?, "projection bias is not supported")?;
        }
        Ok(d)
    }
    pub fn validate(&self) -> Result<()> {
        let (q_width, kv_width) = (self.q_lora_rank, self.kv_lora_rank);
        need(self.hidden_size > 0 && q_width > 0 && kv_width > 0
            && self.num_attention_heads > 0 && self.v_head_dim > 0, "dimensions must be positive")?;
        let qk = add(self.qk_nope_head_dim, self.qk_rope_head_dim)?;
        need(qk > 0 && self.qk_rope_head_dim.is_multiple_of(2), "invalid head split")?;
        need(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0, "invalid RMS norm epsilon")?;
        let latent = add(kv_width, self.qk_rope_head_dim)?;
        let kv = add(self.qk_nope_head_dim, self.v_head_dim)?;
        for count in [mul(q_width, self.hidden_size)?,
            mul(mul(self.num_attention_heads, qk)?, q_width)?,
            mul(latent, self.hidden_size)?, mul(mul(self.num_attention_heads, kv)?, kv_width)?,
            mul(self.num_attention_heads, add(qk, self.v_head_dim)?)?] { bytes(count)?; }
        Ok(())
    }
    fn qk_dim(&self) -> usize { self.qk_nope_head_dim + self.qk_rope_head_dim }
    fn kv_dim(&self) -> usize { self.qk_nope_head_dim + self.v_head_dim }
    fn latent_dim(&self) -> usize { self.kv_lora_rank + self.qk_rope_head_dim }
    pub fn cache_bytes_per_token(&self) -> Result<CacheBytes> {
        self.validate()?;
        Ok(CacheBytes { latent: bytes(self.latent_dim())?,
            expanded: bytes(self.num_attention_heads * (self.qk_dim() + self.v_head_dim))? })
    }
}

/// Borrowed f32 weights. Matrices use row-major `[output][input]` order.
/// Norm weights multiply RMS-normalized values. There is no norm bias.
#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    pub q_a_proj: &'a [f32],
    pub q_a_layernorm: &'a [f32],
    pub q_b_proj: &'a [f32],
    pub kv_a_proj_with_mqa: &'a [f32],
    pub kv_a_layernorm: &'a [f32],
    pub kv_b_proj: &'a [f32],
}
/// Rotation for one token position. Each entry rotates an adjacent pair in the
/// RoPE part of Q and the shared K. Tables must have `qk_rope_head_dim / 2`
/// entries. The caller supplies frequencies, position, and any RoPE scaling.
/// Empty tables select the zero-width RoPE case from the config.
#[derive(Debug, Clone, Copy)]
pub struct Rope<'a> { pub cos: &'a [f32], pub sin: &'a [f32] }
#[derive(Debug, Clone, PartialEq)]
pub struct Projected {
    /// Each head stores `[Q_nope, Q_rope]`.
    pub query: Vec<f32>,
    /// RMS-normalized compressed KV. This is the input to `kv_b_proj`.
    pub latent: Vec<f32>,
    /// Rotated shared K. It does not pass through the latent RMS norm.
    pub rope_key: Vec<f32>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ExpandedKv {
    /// Each head stores `[K_nope, K_rope]`.
    pub key: Vec<f32>,
    /// Each head stores V.
    pub value: Vec<f32>,
}

/// One layer and one sequence. Each row contains only `[latent, rope_key]`.
/// Use the same layer weights for the lifetime of this cache.
#[derive(Debug, Clone)]
pub struct LatentCache { config: Config, data: Vec<f32> }
impl LatentCache {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, data: Vec::new() })
    }
    pub fn len(&self) -> usize { self.data.len() / self.config.latent_dim() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    /// Used payload bytes. Reserved capacity and the cache object are excluded.
    pub fn payload_bytes(&self) -> usize { self.data.len() * std::mem::size_of::<f32>() }
    pub fn push(&mut self, token: &Projected) -> Result<()> {
        check_projected(self.config, token)?;
        let width = self.config.latent_dim();
        bytes(add(self.data.len(), width)?)?;
        self.data.try_reserve(width).map_err(|_| invalid("MLA: cannot reserve cache storage"))?;
        self.data.extend_from_slice(&token.latent);
        self.data.extend_from_slice(&token.rope_key);
        Ok(())
    }
    fn row(&self, token: usize) -> (&[f32], &[f32]) {
        let width = self.config.latent_dim();
        self.data[token * width..(token + 1) * width].split_at(self.config.kv_lora_rank)
    }
}

/// A scalar host reference. It returns attention head outputs before `o_proj`.
pub struct Reference<'a> { config: Config, weights: Weights<'a> }
impl<'a> Reference<'a> {
    pub fn new(config: Config, weights: Weights<'a>) -> Result<Self> {
        config.validate()?;
        let d = config;
        for (w, count) in [(weights.q_a_proj, d.q_lora_rank * d.hidden_size),
            (weights.q_a_layernorm, d.q_lora_rank),
            (weights.q_b_proj, d.num_attention_heads * d.qk_dim() * d.q_lora_rank),
            (weights.kv_a_proj_with_mqa, d.latent_dim() * d.hidden_size),
            (weights.kv_a_layernorm, d.kv_lora_rank),
            (weights.kv_b_proj, d.num_attention_heads * d.kv_dim() * d.kv_lora_rank)] {
            need(w.len() == count, "weight shape differs from config")?;
            finite(w)?;
        }
        Ok(Self { config, weights })
    }
    pub fn project(&self, x: &[f32], rope: Rope<'_>) -> Result<Projected> {
        let (d, w) = (self.config, self.weights);
        need(x.len() == d.hidden_size, "input width differs from config")?;
        need(rope.cos.len() == d.qk_rope_head_dim / 2 && rope.sin.len() == rope.cos.len(), "RoPE table width differs from config")?;
        finite(x)?; finite(rope.cos)?; finite(rope.sin)?;
        let mut low_q = matvec(w.q_a_proj, x);
        rms_norm(&mut low_q, w.q_a_layernorm, d.rms_norm_eps)?;
        let mut query = matvec(w.q_b_proj, &low_q);
        let mut latent = matvec(w.kv_a_proj_with_mqa, x);
        let mut rope_key = latent.split_off(d.kv_lora_rank);
        rms_norm(&mut latent, w.kv_a_layernorm, d.rms_norm_eps)?;
        for q in query.chunks_exact_mut(d.qk_dim()) { rotate(&mut q[d.qk_nope_head_dim..], rope); }
        rotate(&mut rope_key, rope);
        let out = Projected { query, latent, rope_key };
        check_projected(d, &out)?;
        Ok(out)
    }
    /// Expand K and V for one token. `kv_b_proj` stores `[K_nope, V]` per head.
    pub fn expand(&self, token: &Projected) -> Result<ExpandedKv> {
        let d = self.config;
        check_projected(d, token)?;
        let mut key = Vec::with_capacity(d.num_attention_heads * d.qk_dim());
        let mut value = Vec::with_capacity(d.num_attention_heads * d.v_head_dim);
        for head in 0..d.num_attention_heads {
            let kv = self.expand_head(&token.latent, head)?;
            key.extend_from_slice(&kv[..d.qk_nope_head_dim]);
            key.extend_from_slice(&token.rope_key);
            value.extend_from_slice(&kv[d.qk_nope_head_dim..]);
        }
        Ok(ExpandedKv { key, value })
    }
    fn expand_head(&self, latent: &[f32], head: usize) -> Result<Vec<f32>> {
        let d = self.config;
        let width = d.kv_dim() * d.kv_lora_rank;
        let kv = matvec(&self.weights.kv_b_proj[head * width..(head + 1) * width], latent);
        finite(&kv)?;
        Ok(kv)
    }
    /// Attend to every token in a nonempty causal prefix. The caller must not
    /// include future tokens. Scores use `(Q dot K) * (1 / sqrt(Q head width))`.
    /// K and V are expanded on read. Expanded rows are never stored in the cache.
    pub fn attend(&self, query: &[f32], cache: &LatentCache) -> Result<Vec<f32>> {
        let d = self.config;
        need(cache.config == d, "cache config differs from reference")?;
        need(!cache.is_empty(), "attention needs a nonempty prefix")?;
        need(query.len() == d.num_attention_heads * d.qk_dim(), "query width differs from config")?;
        finite(query)?;
        let scale = 1.0 / (d.qk_dim() as f32).sqrt();
        let mut output = vec![0.0; d.num_attention_heads * d.v_head_dim];
        let mut scores = vec![0.0; cache.len()];
        for head in 0..d.num_attention_heads {
            let q = &query[head * d.qk_dim()..(head + 1) * d.qk_dim()];
            for (token, score) in scores.iter_mut().enumerate() {
                let (latent, rope_key) = cache.row(token);
                let kv = self.expand_head(latent, head)?;
                let mut sum = 0.0f32;
                for i in 0..d.qk_nope_head_dim { sum += q[i] * kv[i]; }
                for i in 0..d.qk_rope_head_dim { sum += q[d.qk_nope_head_dim + i] * rope_key[i]; }
                *score = sum * scale;
            }
            softmax(&mut scores)?;
            let out = &mut output[head * d.v_head_dim..(head + 1) * d.v_head_dim];
            for (token, &probability) in scores.iter().enumerate() {
                let kv = self.expand_head(cache.row(token).0, head)?;
                for i in 0..d.v_head_dim { out[i] += probability * kv[d.qk_nope_head_dim + i]; }
            }
        }
        finite(&output)?;
        Ok(output)
    }
    /// Project, append, and attend to the prefix including this token.
    /// On an error, the cache keeps its prior contents.
    pub fn step(&self, cache: &mut LatentCache, x: &[f32], rope: Rope<'_>) -> Result<Vec<f32>> {
        need(cache.config == self.config, "cache config differs from reference")?;
        let token = self.project(x, rope)?;
        let prior = cache.data.len();
        cache.push(&token)?;
        let result = self.attend(&token.query, cache);
        if result.is_err() { cache.data.truncate(prior); }
        result
    }
}

fn check_projected(d: Config, token: &Projected) -> Result<()> {
    let latent_width = d.kv_lora_rank;
    need(token.query.len() == d.num_attention_heads * d.qk_dim() && token.latent.len() == latent_width
        && token.rope_key.len() == d.qk_rope_head_dim, "projected token shape differs from config")?;
    finite(&token.query)?; finite(&token.latent)?; finite(&token.rope_key)
}
fn matvec(matrix: &[f32], x: &[f32]) -> Vec<f32> {
    matrix.chunks_exact(x.len()).map(|row| {
        let mut sum = 0.0f32;
        for i in 0..x.len() { sum += row[i] * x[i]; }
        sum
    }).collect()
}
fn rms_norm(x: &mut [f32], weights: &[f32], eps: f32) -> Result<()> {
    let mut sum = 0.0f32;
    for &v in x.iter() { sum += v * v; }
    need(sum.is_finite(), "RMS norm sum is not finite")?;
    let inv = 1.0 / (sum / x.len() as f32 + eps).sqrt();
    for (v, &weight) in x.iter_mut().zip(weights) { *v = (*v * inv) * weight; }
    finite(x)
}
fn rotate(x: &mut [f32], rope: Rope<'_>) {
    for (i, pair) in x.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let (a, b) = (pair[0], pair[1]);
        pair[0] = a * rope.cos[i] - b * rope.sin[i];
        pair[1] = b * rope.cos[i] + a * rope.sin[i];
    }
}
fn softmax(scores: &mut [f32]) -> Result<()> {
    finite(scores)?;
    let mut max = f32::NEG_INFINITY;
    for &score in scores.iter() { max = max.max(score); }
    let mut sum = 0.0f32;
    for score in scores.iter_mut() { *score = (*score - max).exp(); sum += *score; }
    need(sum.is_finite() && sum > 0.0, "invalid softmax sum")?;
    for score in scores { *score /= sum; }
    Ok(())
}

#[cfg(test)]
mod tests;
