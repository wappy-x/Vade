//! Fixed-order f32 host reference for Kimi Delta Attention (KDA).
//! All dimensions come from config data. This module does not enable a model
//! plan or change serving. See `kda/README.md` for the public source and scope.
//! Reductions visit dimensions in ascending order. Chunk and step calls use
//! the same recurrence. Products and sums are separate f32 operations.
use crate::{Result, invalid, json::{self, Value, field}, shapes::CONFIG_BYTES};

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("KDA: {message}"))) }
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| invalid("KDA: dimension product overflow"))
}
fn buffer_size(count: usize) -> Result<()> {
    need(mul(count, std::mem::size_of::<f32>())? <= isize::MAX as usize, "buffer exceeds the address range")
}
fn finite(x: &[f32]) -> Result<()> { need(x.iter().all(|v| v.is_finite()), "values must be finite") }
fn zeros(count: usize) -> Result<Vec<f32>> {
    let mut data = Vec::new();
    data.try_reserve_exact(count).map_err(|_| invalid("KDA: cannot reserve state storage"))?;
    data.resize(count, 0.0);
    Ok(data)
}

/// Equal Q, K and V head widths, with a configured causal convolution width.
/// The low-rank forget and output gates each use `head_dim` as their rank.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub conv_kernel_size: usize,
    pub gate_lower_bound: Option<f32>,
    pub rms_norm_eps: f32,
}
impl Config {
    /// Read an explicit text config, or its `text_config` envelope.
    /// Accept nested `linear_attn_config` data or explicit `linear_*` fields.
    /// No model name or default dimensions are used. A null lower bound selects
    /// the source's softplus formula. This does not validate checkpoint storage.
    pub fn from_json(input: &[u8]) -> Result<Self> {
        let doc = json::parse(input, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let o = root.get("text_config").map(Value::object).transpose()?.unwrap_or(root);
        need(field(o, "hidden_act")?.string()? == "silu", "only the SiLU convolution activation is supported")?;
        let dim = |value: &Value| -> Result<usize> {
            usize::try_from(value.u64()?).map_err(|_| invalid("KDA: dimension exceeds usize"))
        };
        let (heads, head, conv, lower) = if let Some(v) = o.get("linear_attn_config") {
            need(!["linear_num_heads", "linear_head_dim", "linear_conv_kernel_dim", "linear_lower_bound"]
                .iter().any(|name| o.contains_key(*name)), "ambiguous linear attention config")?;
            let a = v.object()?;
            (field(a, "num_heads")?, field(a, "head_dim")?, field(a, "short_conv_kernel_size")?, field(a, "gate_lower_bound")?)
        } else {
            (field(o, "linear_num_heads")?, field(o, "linear_head_dim")?, field(o, "linear_conv_kernel_dim")?, field(o, "linear_lower_bound")?)
        };
        let d = Self { hidden_size: dim(field(o, "hidden_size")?)?, num_heads: dim(heads)?, head_dim: dim(head)?,
            conv_kernel_size: dim(conv)?, gate_lower_bound: match lower { Value::Null => None, v => Some(v.f64()? as f32) },
            rms_norm_eps: field(o, "rms_norm_eps")?.f64()? as f32 };
        d.validate()?;
        Ok(d)
    }
    pub fn validate(&self) -> Result<()> {
        need(self.hidden_size > 0 && self.num_heads > 0 && self.head_dim > 0 && self.conv_kernel_size > 0,
            "dimensions must be positive")?;
        need(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0, "invalid RMS norm epsilon")?;
        if let Some(bound) = self.gate_lower_bound { need(bound.is_finite() && bound < 0.0, "invalid gate lower bound")?; }
        let width = mul(self.num_heads, self.head_dim)?;
        for count in [mul(width, self.hidden_size)?, mul(width, self.conv_kernel_size)?,
            mul(self.num_heads, self.hidden_size)?, mul(self.head_dim, self.hidden_size)?,
            mul(width, self.head_dim)?] { buffer_size(count)?; }
        Ok(())
    }
    fn width(self) -> usize { self.num_heads * self.head_dim }
}

/// Decoded f32 weights. Matrices use row-major `[output][input]` order.
/// Convolution weights use `[head * head_dim][tap]`, oldest tap first.
/// Q/K/V and output projection shapes are `[width, hidden_size]` and
/// `[hidden_size, width]`. The gate ranks are `head_dim`, where
/// `width = num_heads * head_dim`. There are no projection or conv biases.
#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub q_conv1d: &'a [f32],
    pub k_conv1d: &'a [f32],
    pub v_conv1d: &'a [f32],
    /// `[num_heads, hidden_size]`, followed by sigmoid.
    pub b_proj: &'a [f32],
    pub f_a_proj: &'a [f32],
    pub f_b_proj: &'a [f32],
    pub g_a_proj: &'a [f32],
    pub g_b_proj: &'a [f32],
    /// One bias per key channel, `[width]`.
    pub dt_bias: &'a [f32],
    /// One log rate per head, `[num_heads]`.
    pub a_log: &'a [f32],
    /// Shared per-head RMS norm weight, `[head_dim]`.
    pub o_norm: &'a [f32],
    pub o_proj: &'a [f32],
}

/// One sequence and one layer. Keep the same weights for this state's lifetime.
/// A clone preserves the recurrence and all separate convolution histories.
#[derive(Debug, Clone)]
pub struct State {
    config: Config,
    s: Vec<f32>,
    q_history: Vec<f32>,
    k_history: Vec<f32>,
    v_history: Vec<f32>,
}
impl State {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let history = (config.conv_kernel_size - 1) * config.width();
        Ok(Self { config, s: zeros(config.width() * config.head_dim)?,
            q_history: zeros(history)?, k_history: zeros(history)?, v_history: zeros(history)? })
    }
    /// State in `[head][key_channel][value_channel]` order.
    pub fn recurrent_state(&self) -> &[f32] { &self.s }
}

/// Scalar host reference. Math library results are local to the host target.
/// Host bit equality does not establish GPU or cross-target bit equality.
pub struct Reference<'a> { config: Config, weights: Weights<'a> }
impl<'a> Reference<'a> {
    pub fn new(config: Config, weights: Weights<'a>) -> Result<Self> {
        config.validate()?;
        let (d, w) = (config, weights);
        let width = d.width();
        for (data, count) in [(w.q_proj, width * d.hidden_size), (w.k_proj, width * d.hidden_size),
            (w.v_proj, width * d.hidden_size), (w.q_conv1d, width * d.conv_kernel_size),
            (w.k_conv1d, width * d.conv_kernel_size), (w.v_conv1d, width * d.conv_kernel_size),
            (w.b_proj, d.num_heads * d.hidden_size), (w.f_a_proj, d.head_dim * d.hidden_size),
            (w.f_b_proj, width * d.head_dim), (w.g_a_proj, d.head_dim * d.hidden_size),
            (w.g_b_proj, width * d.head_dim), (w.dt_bias, width), (w.a_log, d.num_heads),
            (w.o_norm, d.head_dim), (w.o_proj, d.hidden_size * width)] {
            need(data.len() == count, "weight shape differs from config")?;
            finite(data)?;
        }
        Ok(Self { config, weights })
    }
    /// Advance one token. An input or arithmetic error leaves state unchanged.
    pub fn step(&self, state: &mut State, x: &[f32]) -> Result<Vec<f32>> {
        need(state.config == self.config, "state config differs from reference")?;
        need(x.len() == self.config.hidden_size, "input width differs from config")?;
        finite(x)?;
        let mut next = state.clone();
        let output = self.step_inner(&mut next, x)?;
        *state = next;
        Ok(output)
    }
    /// Scan a token-major `[tokens][hidden_size]` chunk from the supplied state.
    /// An empty chunk is valid. Every chunk uses the exact step arithmetic.
    /// An input or arithmetic error leaves the full input state unchanged.
    pub fn scan(&self, state: &mut State, x: &[f32]) -> Result<Vec<f32>> {
        need(state.config == self.config, "state config differs from reference")?;
        need(x.len() % self.config.hidden_size == 0, "chunk width differs from config")?;
        finite(x)?;
        let mut output = Vec::new();
        output.try_reserve_exact(x.len()).map_err(|_| invalid("KDA: cannot reserve chunk output"))?;
        let mut next = state.clone();
        for token in x.chunks_exact(self.config.hidden_size) { output.extend(self.step_inner(&mut next, token)?); }
        *state = next;
        Ok(output)
    }
    fn step_inner(&self, state: &mut State, x: &[f32]) -> Result<Vec<f32>> {
        let d = self.config;
        let p = self.prepare(state, x)?;
        let mut core = recurrent(d, &mut state.s, &p);
        finite(&state.s)?; finite(&core)?;
        for h in 0..d.num_heads {
            let row = &mut core[h * d.head_dim..(h + 1) * d.head_dim];
            let mut squares = 0.0;
            for &v in row.iter() { squares += v * v; }
            need(squares.is_finite(), "RMS norm sum overflow")?;
            let inverse = 1.0 / (squares / d.head_dim as f32 + d.rms_norm_eps).sqrt();
            for j in 0..d.head_dim {
                let norm = row[j] * inverse;
                let weighted = self.weights.o_norm[j] * norm;
                row[j] = weighted * sigmoid(p.output_gate[h * d.head_dim + j]);
            }
        }
        matvec(self.weights.o_proj, &core)
    }
    fn prepare(&self, state: &mut State, x: &[f32]) -> Result<Prepared> {
        let (d, w) = (self.config, self.weights);
        let mut query = convolution(d, w.q_conv1d, &mut state.q_history, &matvec(w.q_proj, x)?)?;
        let mut key = convolution(d, w.k_conv1d, &mut state.k_history, &matvec(w.k_proj, x)?)?;
        let value = convolution(d, w.v_conv1d, &mut state.v_history, &matvec(w.v_proj, x)?)?;
        let scale = 1.0 / (d.head_dim as f32).sqrt();
        for h in 0..d.num_heads {
            let q = &mut query[h * d.head_dim..(h + 1) * d.head_dim];
            let k = &mut key[h * d.head_dim..(h + 1) * d.head_dim];
            l2norm(q)?; l2norm(k)?;
            for v in q { *v *= scale; }
        }
        let beta = matvec(w.b_proj, x)?.into_iter().map(sigmoid).collect();
        let mut log_decay = matvec(w.f_b_proj, &matvec(w.f_a_proj, x)?)?;
        for h in 0..d.num_heads {
            let rate = w.a_log[h].exp();
            need(rate.is_finite(), "decay rate overflow")?;
            for j in 0..d.head_dim {
                let at = h * d.head_dim + j;
                let activation = log_decay[at] + w.dt_bias[at];
                need(activation.is_finite(), "forget gate sum overflow")?;
                log_decay[at] = match d.gate_lower_bound {
                    Some(bound) => bound * sigmoid(rate * activation),
                    None => -rate * softplus(activation),
                };
            }
        }
        finite(&log_decay)?;
        let output_gate = matvec(w.g_b_proj, &matvec(w.g_a_proj, x)?)?;
        Ok(Prepared { query, key, value, beta, log_decay, output_gate })
    }
}

struct Prepared {
    query: Vec<f32>, key: Vec<f32>, value: Vec<f32>, beta: Vec<f32>,
    log_decay: Vec<f32>, output_gate: Vec<f32>,
}
fn matvec(weights: &[f32], x: &[f32]) -> Result<Vec<f32>> {
    let mut output = Vec::with_capacity(weights.len() / x.len());
    for row in weights.chunks_exact(x.len()) {
        let mut sum = 0.0;
        for i in 0..x.len() { sum += row[i] * x[i]; }
        output.push(sum);
    }
    finite(&output)?;
    Ok(output)
}
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn silu(x: f32) -> f32 { x * sigmoid(x) }
fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
fn l2norm(x: &mut [f32]) -> Result<()> {
    let mut squares = 0.0;
    for &v in x.iter() { squares += v * v; }
    need(squares.is_finite(), "L2 norm sum overflow")?;
    let norm = (squares + 1e-6).sqrt();
    for v in x { *v /= norm; }
    Ok(())
}
fn convolution(d: Config, weights: &[f32], history: &mut [f32], raw: &[f32]) -> Result<Vec<f32>> {
    let (width, taps) = (d.width(), d.conv_kernel_size);
    let mut output = vec![0.0; width];
    for ch in 0..width {
        let mut sum = 0.0;
        for tap in 0..taps {
            let sample = if tap + 1 == taps { raw[ch] } else { history[tap * width + ch] };
            sum += sample * weights[ch * taps + tap];
        }
        need(sum.is_finite(), "convolution sum overflow")?;
        output[ch] = silu(sum);
    }
    if taps > 1 {
        history.copy_within(width.., 0);
        history[(taps - 2) * width..].copy_from_slice(raw);
    }
    Ok(output)
}
fn recurrent(d: Config, state: &mut [f32], p: &Prepared) -> Vec<f32> {
    let mut output = vec![0.0; d.width()];
    for h in 0..d.num_heads {
        let offset = h * d.head_dim;
        let s = &mut state[offset * d.head_dim..(offset + d.head_dim) * d.head_dim];
        for i in 0..d.head_dim {
            let decay = p.log_decay[offset + i].exp();
            for j in 0..d.head_dim { s[i * d.head_dim + j] *= decay; }
        }
        let mut delta = vec![0.0; d.head_dim];
        for j in 0..d.head_dim {
            let mut memory = 0.0;
            for i in 0..d.head_dim { memory += s[i * d.head_dim + j] * p.key[offset + i]; }
            delta[j] = (p.value[offset + j] - memory) * p.beta[h];
        }
        for i in 0..d.head_dim {
            for j in 0..d.head_dim { s[i * d.head_dim + j] += p.key[offset + i] * delta[j]; }
        }
        for j in 0..d.head_dim {
            let mut sum = 0.0;
            for i in 0..d.head_dim { sum += s[i * d.head_dim + j] * p.query[offset + i]; }
            output[offset + j] = sum;
        }
    }
    output
}

#[cfg(test)]
mod tests;
