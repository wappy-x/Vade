//! Host f32 reference for manifold-constrained hyper-connections (mHC).
//! This module does not change model admission or device execution.
//! See `mhc/README.md` for the public source, tensor layout, and arithmetic order.
use crate::{Result, invalid, json::{self, Value, field}, shapes::{CONFIG_BYTES, MAX_SINKHORN_ITERS}};

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("mHC: {message}"))) }
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| invalid("mHC: dimension product overflow"))
}
fn finite(x: &[f32]) -> Result<()> {
    need(x.iter().all(|v| v.is_finite()), "values must be finite")
}

/// Dimensions and normalization settings. Each field comes from the text config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub rms_norm_eps: f32,
    pub hc_eps: f32,
}
impl Config {
    /// Bind the deferred Sinkhorn block description. No model name is read.
    pub fn from_description(d: &crate::plan::TextDescription) -> Result<Self> {
        let count = |n| usize::try_from(n).map_err(|_| invalid("mHC: count exceeds usize"));
        let config = Self { hidden_size: count(d.hidden)?, hc_mult: count(d.residual.streams)?,
            hc_sinkhorn_iters: count(d.residual.iterations)?, rms_norm_eps: d.epsilon as f32,
            hc_eps: d.residual.epsilon as f32 };
        config.validate()?; Ok(config)
    }
    /// Read a text config or its enclosing `text_config` object.
    /// All operation fields are required. A model name does not select behavior.
    pub fn from_json(input: &[u8]) -> Result<Self> {
        let doc = json::parse(input, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let o = root.get("text_config").map(Value::object).transpose()?.unwrap_or(root);
        if let Some(v) = o.get("mhc") { need(v.boolean()?, "mhc must be enabled")?; }
        let count = |name| -> Result<usize> {
            usize::try_from(field(o, name)?.u64()?).map_err(|_| invalid("mHC: count exceeds usize"))
        };
        let config = Self { hidden_size: count("hidden_size")?, hc_mult: count("hc_mult")?,
            hc_sinkhorn_iters: count("hc_sinkhorn_iters")?, rms_norm_eps: field(o, "rms_norm_eps")?.f64()? as f32,
            hc_eps: field(o, "hc_eps")?.f64()? as f32 };
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        need(self.hidden_size > 0 && self.hc_mult > 0, "dimensions must be positive")?;
        need(self.hc_sinkhorn_iters > 0 && self.hc_sinkhorn_iters <= MAX_SINKHORN_ITERS as usize,
            "Sinkhorn iteration count is outside its limit")?;
        need(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0, "invalid RMS norm epsilon")?;
        need(self.hc_eps.is_finite() && self.hc_eps > 0.0, "invalid HC epsilon")?;
        let streams_plus_two = self.hc_mult.checked_add(2).ok_or_else(|| invalid("mHC: dimension sum overflow"))?;
        let rows = mul(streams_plus_two, self.hc_mult)?;
        let wide = mul(self.hc_mult, self.hidden_size)?;
        let bytes = mul(mul(rows, wide)?, std::mem::size_of::<f32>())?;
        need(bytes <= isize::MAX as usize, "projection exceeds the address range")
    }
    fn wide(&self) -> usize { self.hc_mult * self.hidden_size }
    fn rows(&self) -> usize { (self.hc_mult + 2) * self.hc_mult }
    /// Unweighted final stream mean. The caller applies the final RMS norm next.
    pub fn mean_head(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.validate()?;
        need(x.len() == self.wide(), "head shape mismatch")?; finite(x)?;
        let mut out = vec![0.0f32; self.hidden_size];
        for src in 0..self.hc_mult {
            for d in 0..self.hidden_size { out[d] += x[src * self.hidden_size + d]; }
        }
        for v in &mut out { *v /= self.hc_mult as f32; }
        finite(&out)?; Ok(out)
    }
}

/// Borrowed logical f32 tensors for either the attention or FFN connection.
/// `projection` is the `fn` tensor: row-major `[N*(N+2), N*D]`.
/// `base` stores `[pre[N], post[N], comb[N*N]]`. `scale` stores `[pre, post, comb]`.
#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    pub projection: &'a [f32],
    pub base: &'a [f32],
    pub scale: &'a [f32],
}
/// Coefficients and block input for one token. `comb` is `[source][destination]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mix {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
    pub collapsed: Vec<f32>,
}
pub struct Reference<'a> { config: Config, weights: Weights<'a> }
impl<'a> Reference<'a> {
    pub fn new(config: Config, weights: Weights<'a>) -> Result<Self> {
        config.validate()?;
        need(weights.projection.len() == config.rows() * config.wide()
            && weights.base.len() == config.rows() && weights.scale.len() == 3, "weight shape mismatch")?;
        for values in [weights.projection, weights.base, weights.scale] { finite(values)?; }
        Ok(Self { config, weights })
    }
    /// Compute coefficients and collapse raw `[stream][hidden]` residuals.
    /// Each reduction visits its fixed tensor dimension in ascending order.
    pub fn mix(&self, x: &[f32]) -> Result<Mix> {
        let c = self.config;
        let n = c.hc_mult;
        need(x.len() == c.wide(), "input shape mismatch")?;
        finite(x)?;
        let mut squares = 0.0f32;
        for &v in x { squares += v * v; }
        let variance = squares / c.wide() as f32 + c.rms_norm_eps;
        need(variance.is_finite(), "RMS norm exceeds f32 range")?;
        let inverse = 1.0 / variance.sqrt();
        let normalized: Vec<f32> = x.iter().map(|&v| v * inverse).collect();
        finite(&normalized)?;
        let mut logits = vec![0.0f32; c.rows()];
        for (row, logit) in logits.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (col, &value) in normalized.iter().enumerate() {
                sum += self.weights.projection[row * c.wide() + col] * value;
            }
            need(sum.is_finite(), "projection exceeds f32 range")?;
            let group = if row < n { 0 } else if row < 2 * n { 1 } else { 2 };
            *logit = sum * self.weights.scale[group] + self.weights.base[row];
        }
        finite(&logits)?;
        let pre: Vec<f32> = logits[..n].iter().map(|&v| sigmoid(v) + c.hc_eps).collect();
        let post: Vec<f32> = logits[n..2 * n].iter().map(|&v| 2.0 * sigmoid(v)).collect();
        let comb = sinkhorn(&logits[2 * n..], n, c.hc_sinkhorn_iters, c.hc_eps)?;
        let mut collapsed = vec![0.0f32; c.hidden_size];
        for src in 0..n {
            for d in 0..c.hidden_size { collapsed[d] += pre[src] * x[src * c.hidden_size + d]; }
        }
        finite(&pre)?;
        finite(&collapsed)?;
        Ok(Mix { pre, post, comb, collapsed })
    }
    /// Return `post[dst]*block[d] + sum_src comb[src,dst]*x[src,d]`.
    /// Use the raw residual and the coefficients from its preceding `mix` call.
    /// The attention or FFN block, including its input norm, is supplied by the caller.
    pub fn combine(&self, x: &[f32], mixing: &Mix, block: &[f32]) -> Result<Vec<f32>> {
        let c = self.config;
        let n = c.hc_mult;
        need(x.len() == c.wide() && block.len() == c.hidden_size
            && mixing.post.len() == n && mixing.comb.len() == n * n, "combine shape mismatch")?;
        for values in [x, block, &mixing.post, &mixing.comb] { finite(values)?; }
        let mut out = vec![0.0f32; c.wide()];
        for dst in 0..n {
            for d in 0..c.hidden_size {
                let mut residual = 0.0f32;
                for src in 0..n { residual += mixing.comb[src * n + dst] * x[src * c.hidden_size + d]; }
                out[dst * c.hidden_size + d] = mixing.post[dst] * block[d] + residual;
            }
        }
        finite(&out)?;
        Ok(out)
    }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// The first iteration is row softmax plus epsilon, then column normalization.
// Each later iteration normalizes rows, then columns. Every divisor adds epsilon.
fn sinkhorn(logits: &[f32], n: usize, iterations: usize, eps: f32) -> Result<Vec<f32>> {
    let mut comb = logits.to_vec();
    for row in comb.chunks_exact_mut(n) {
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter() { max = max.max(v); }
        let mut sum = 0.0f32;
        for v in row.iter_mut() { *v = (*v - max).exp(); sum += *v; }
        for v in row { *v = *v / sum + eps; }
    }
    normalize_columns(&mut comb, n, eps)?;
    for _ in 1..iterations {
        for row in comb.chunks_exact_mut(n) {
            let mut sum = 0.0f32;
            for &v in row.iter() { sum += v; }
            let divisor = sum + eps;
            need(divisor.is_finite() && divisor > 0.0, "Sinkhorn row divisor exceeds f32 range")?;
            for v in row { *v /= divisor; }
        }
        normalize_columns(&mut comb, n, eps)?;
    }
    Ok(comb)
}
fn normalize_columns(comb: &mut [f32], n: usize, eps: f32) -> Result<()> {
    for dst in 0..n {
        let mut sum = 0.0f32;
        for src in 0..n { sum += comb[src * n + dst]; }
        let divisor = sum + eps;
        need(divisor.is_finite() && divisor > 0.0, "Sinkhorn column divisor exceeds f32 range")?;
        for src in 0..n { comb[src * n + dst] /= divisor; }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
