//! Fixed-order host reference for softmax attention with a sink, a relative
//! position term, length clips, shared key and value heads and an output gate.
//! A layer attends to a sliding window or to the full prefix.
//! See `sink_attention/README.md` for the contract and the fixed sum orders.
//!
//! This module does not admit a checkpoint or change serving. All sizes come
//! from config data and tensor shapes. The source is the public Transformers
//! release v5.24.0, commit `eeee11099135592f1d03c2871fbd3d8132e5a4d7`, file
//! `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`:
//! lines 67-76 (plain norm), 99-142 (shared heads, scores, sink, softmax),
//! 176-236 (clips and position scale), 239-270 (relative term), 458-543
//! (projections and output gate) and 546-727 (the two layer kinds).
//! No source code is copied here.
//!
//! Number types follow that source. A projection is a BF16 product: one f32
//! chain in ascending input index from +0.0, then one rounding to BF16. Clips,
//! norms, the softmax and the gate use f32 and round their result to BF16.
//! Products and sums are separate f32 operations.
//!
//! The source leaves the order of its sums to a library. This reference uses
//! the fixed orders of the engine's device operations, so that a device run can
//! give the same words: a chain for a product, the 32-lane tree for the squares
//! of a clip, the norm tree for the squares of a norm, and the distance order
//! for the softmax sum and the value sum. No order depends on the place of a
//! token in a buffer. Every exponential, logarithm and BF16 round goes through
//! the engine's shared `fixed_math`. Equal bits here do not establish equal bits on a GPU.
use crate::{Result, fixed_math::{exp, ln, rn as round_bf16, sigmoid}, invalid, json::{self, Value, field}, shapes::CONFIG_BYTES,
    short_conv::{Slot, Taps, VisitHistories, reciprocal}};

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("sink attention: {message}"))) }
}
fn mul(a: usize, b: usize) -> Result<usize> {
    let n = a.checked_mul(b).ok_or_else(|| invalid("sink attention: dimension product overflow"))?;
    need(n.checked_mul(std::mem::size_of::<f32>()).is_some_and(|bytes| bytes <= isize::MAX as usize), "buffer exceeds the address range")?;
    Ok(n)
}
/// Round to the nearest BF16 value, ties to even. An overflow to infinity is an error.
fn round(x: f32) -> Result<f32> {
    need(x.is_finite(), "result is not finite")?;
    let out = round_bf16(x);
    need(out.is_finite(), "result is not finite in BF16")?;
    Ok(out)
}
fn exact(x: &[f32]) -> Result<()> {
    need(x.iter().all(|&v| v.is_finite() && round_bf16(v).to_bits() == v.to_bits()), "values must be exact BF16")
}
/// One BF16 product: an f32 chain in ascending index from +0.0, rounded one time.
fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    let mut sum = 0.0f32;
    for i in 0..a.len() { sum += a[i] * b[i]; }
    round(sum)
}
/// A BF16 matrix product. `matrix` is row-major `[output][input]`.
fn rows(matrix: &[f32], x: &[f32]) -> Result<Vec<f32>> { matrix.chunks_exact(x.len()).map(|row| dot(row, x)).collect() }
/// One head of one token in a `[token][kv_head][head_dim]` buffer.
fn head_row(data: &[f32], kv: usize, head_dim: usize, token: usize, head: usize) -> &[f32] {
    &data[token * kv + head * head_dim..token * kv + (head + 1) * head_dim]
}

/// Lanes of the fixed sum trees.
pub const SUM_LANES: usize = 32;
/// Values in one chunk of the norm tree.
pub const NORM_CHUNK: usize = 128;
/// Distances in one block of the distance order.
pub const DISTANCE_BLOCK: usize = 1024;
/// Join 32 lanes at the distances 16, 8, 4, 2 and 1: a lane adds its upper partner.
fn join(mut lanes: [f32; SUM_LANES]) -> f32 {
    for distance in [16, 8, 4, 2, 1] {
        let old = lanes;
        for lane in 0..SUM_LANES - distance { lanes[lane] += old[lane + distance]; }
    }
    lanes[0]
}
/// The 32-lane tree of the engine's device operations for a sum of squares:
/// lane `i mod 32` is a chain in ascending index from +0.0, then the join.
fn square_tree(x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; SUM_LANES];
    for (i, &v) in x.iter().enumerate() { lanes[i % SUM_LANES] += v * v; }
    join(lanes)
}
/// The tree of the engine's norm operation: the 32-lane tree for each chunk of
/// 128 values, then the chunk sums in 32 lanes and the same join.
fn norm_squares(x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; SUM_LANES];
    for (chunk, values) in x.chunks(NORM_CHUNK).enumerate() { lanes[chunk % SUM_LANES] += square_tree(values); }
    join(lanes)
}
/// The distance order. `term(d)` belongs to the key `d` tokens back, and
/// `d = 0` is the token itself. Distances are cut into blocks of 1,024. In a
/// block, lane `d mod 32` is a chain in ascending `d` from +0.0, then the join.
/// The block sums are one chain in ascending block number from +0.0. The order
/// is a function of the distance and of the count, and of nothing else.
fn distance_sum(count: usize, term: impl Fn(usize) -> f32) -> f32 {
    let (mut total, mut start) = (0.0f32, 0);
    while start < count {
        let end = (start + DISTANCE_BLOCK).min(count);
        let mut lanes = [0.0f32; SUM_LANES];
        for distance in start..end { lanes[distance % SUM_LANES] += term(distance); }
        total += join(lanes);
        start = end;
    }
    total
}

/// Added to the length before the clip takes its reciprocal (source lines 187 and 202).
pub const CLIP_EPSILON: f32 = 1e-6;
/// The form of a number divided by a 32-bit tensor value: the number in f32
/// times the f32 reciprocal of the value. These are two rounded steps, not one
/// division. The source writes `threshold / (norm + 1e-6)` (lines 187 and 202).
/// MEASURED on the source's GPU and CPU runs: 4,000,000 of 4,000,000 values
/// equal this form for each of 4 numbers, and a true division equals about 72
/// percent for 2 of them. Lab receipt, not in this repository:
/// `ink/ref/contract-A/divform2.json`.
pub fn number_over(number: f32, value: f32) -> f32 { number * (1.0 / value) }
// TEMP: local copies of two shared small functions. The shared module is on
// another branch now. When both are on one branch, replace the two functions
// and the constant above with it, after its sums use the same trees.
/// `x * min(1, threshold / (length(x) + 1e-6))` in f32. The caller rounds.
/// The division has the form of `number_over`. The squares use the 32-lane tree.
fn length_clip(x: &[f32], threshold: f32) -> Result<Vec<f32>> {
    let ratio = number_over(threshold, square_tree(x).sqrt() + CLIP_EPSILON);
    need(!ratio.is_nan(), "clip input is not finite")?;
    let factor = if ratio > 1.0 { 1.0 } else { ratio };
    Ok(x.iter().map(|&v| v * factor).collect())
}
/// `x * 1 / sqrt(mean(x * x) + eps)` in f32 with no learned weight. The caller
/// rounds. The squares use the norm tree.
fn plain_norm(x: &[f32], eps: f32) -> Result<Vec<f32>> {
    let inverse = 1.0 / (norm_squares(x) / x.len() as f32 + eps).sqrt();
    need(inverse.is_finite(), "norm input is not finite")?;
    Ok(x.iter().map(|&v| v * inverse).collect())
}
/// The factor on a score: the f32 value of `head_dim` to the power -0.5
/// (source lines 558 and 126). The factor itself is not rounded to BF16.
pub fn score_scale(head_dim: usize) -> f32 { (1.0 / (head_dim as f64).sqrt()) as f32 }
/// The key and value head that query head `head` reads (source lines 99-108).
pub const fn shared_head(head: usize, num_heads: usize, num_kv_heads: usize) -> usize { head / (num_heads / num_kv_heads) }

/// The tokens that a layer can see.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reach {
    /// This token and the `window - 1` tokens before it. The relative table has
    /// one column for each distance in the window.
    Window { window: usize },
    /// Every token up to this one. The relative table has `rel_extent` columns.
    /// Query and relative states are scaled by
    /// `1 + log_alpha * ln(max((position + 1) / log_floor, 1))`.
    /// `log_floor` is held in f64: it is the divisor of a tensor (`reciprocal`).
    Full { rel_extent: usize, log_alpha: f32, log_floor: f64 },
}
impl Reach {
    pub fn rel_extent(self) -> usize { match self { Self::Window { window } => window, Self::Full { rel_extent, .. } => rel_extent } }
    /// The position scale in f32, or `None` for a window layer (source lines
    /// 215-220). The host makes it with the logarithm of `fixed_math`. The
    /// division by `log_floor` is a product with its f32 reciprocal. The scale
    /// is exactly 1 while that product is not above 1.
    pub fn position_scale(self, position: usize) -> Option<f32> {
        let Self::Full { log_alpha, log_floor, .. } = self else { return None };
        let ratio = ((position as f64 + 1.0) as f32 * reciprocal(log_floor)).max(1.0);
        Some(1.0 + log_alpha * ln(ratio))
    }
}

/// Sizes and thresholds of one attention layer. No model name is needed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    /// Query heads.
    pub num_heads: usize,
    /// Key and value heads. Each one serves `num_heads / num_kv_heads` query heads.
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Width of the relative state of one query head.
    pub rel_dim: usize,
    /// Taps of the key and value convolutions.
    pub conv_taps: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub rms_norm_eps: f32,
    pub reach: Reach,
}
impl Config {
    /// Read one layer kind from a text config or from its `text_config` object.
    /// `window` selects the fields of a sliding-window layer. No default sizes
    /// are used. This is not a model plan parser and it does not read a model
    /// name. The caller decides which layers have a window.
    pub fn from_json(input: &[u8], window: bool) -> Result<Self> {
        let doc = json::parse(input, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let o = root.get("text_config").map(Value::object).transpose()?.unwrap_or(root);
        let dim = |name| -> Result<usize> {
            usize::try_from(field(o, name)?.u64()?).map_err(|_| invalid("sink attention: dimension exceeds usize"))
        };
        let real = |name| -> Result<f32> { Ok(field(o, name)?.f64()? as f32) };
        for name in ["q_bias", "o_bias"] {
            if let Some(v) = o.get(name) { need(!v.boolean()?, "projection bias is not supported")?; }
        }
        if let Some(v) = o.get("use_value_norm_no_gamma") { need(v.boolean()?, "a weighted value norm is not supported")?; }
        let (num_heads, num_kv_heads, head_dim, reach) = if window {
            (dim("swa_num_attention_heads")?, dim("swa_num_key_value_heads")?, dim("swa_head_dim")?,
                Reach::Window { window: dim("sliding_window_size")? })
        } else {
            (dim("num_attention_heads")?, dim("num_key_value_heads")?, dim("head_dim")?,
                Reach::Full { rel_extent: dim("rel_extent")?, log_alpha: real("log_scaling_alpha")?, log_floor: field(o, "log_scaling_n_floor")?.f64()? })
        };
        let d = Self { hidden_size: dim("hidden_size")?, num_heads, num_kv_heads, head_dim, rel_dim: dim("d_rel")?,
            conv_taps: dim("sconv_kernel_size")?, qk_clip: real("qk_l2_clip_threshold")?,
            rel_clip: real("rel_l2_clip_threshold")?, rms_norm_eps: real("rms_norm_eps")?, reach };
        d.validate()?;
        Ok(d)
    }
    pub fn validate(&self) -> Result<()> {
        need(self.hidden_size > 0 && self.num_heads > 0 && self.num_kv_heads > 0 && self.head_dim > 0
            && self.rel_dim > 0 && self.conv_taps > 0 && self.reach.rel_extent() > 0, "dimensions must be positive")?;
        need(self.num_heads.is_multiple_of(self.num_kv_heads), "query heads must be a multiple of key and value heads")?;
        for clip in [self.qk_clip, self.rel_clip] { need(clip.is_finite() && clip > 0.0, "invalid clip threshold")?; }
        need(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0, "invalid RMS norm epsilon")?;
        if let Reach::Full { log_alpha, log_floor, .. } = self.reach {
            need(log_alpha.is_finite() && log_alpha >= 0.0 && log_floor > 0.0 && reciprocal(log_floor).is_normal(), "invalid position scale")?;
        }
        let (q, kv) = (mul(self.num_heads, self.head_dim)?, mul(self.num_kv_heads, self.head_dim)?);
        mul(q, self.hidden_size)?; mul(kv, self.hidden_size)?; mul(mul(self.num_heads, self.rel_dim)?, self.hidden_size)?;
        mul(self.rel_dim, self.reach.rel_extent())?;
        Ok(())
    }
    fn query_width(&self) -> usize { self.num_heads * self.head_dim }
    /// Channels of the key and value convolutions.
    pub fn kv_width(&self) -> usize { self.num_kv_heads * self.head_dim }
    /// Histories for one visit of a layer with this config.
    pub fn histories(&self) -> Result<VisitHistories> { VisitHistories::new(self.kv_width(), self.hidden_size, self.conv_taps) }
}

/// Borrowed weights as f32. Matrices use row-major `[output][input]` order.
/// Every value must be exact BF16, apart from fixed convolution taps. There is
/// no bias. The names follow the operation, not a checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    /// `[num_heads * head_dim, hidden_size]`.
    pub q_proj: &'a [f32],
    /// `[num_heads, head_dim]`, multiplied into the query before its clip.
    pub q_scale: &'a [f32],
    /// `[num_kv_heads * head_dim, hidden_size]`.
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    /// Convolutions over the projected keys and values, before the clip or norm.
    pub k_conv: Taps<'a>,
    pub v_conv: Taps<'a>,
    /// `[num_heads * rel_dim, hidden_size]`.
    pub rel_proj: &'a [f32],
    /// `[num_heads, rel_dim]`, multiplied into the relative state before its clip.
    pub rel_scale: &'a [f32],
    /// `[rel_dim, rel_extent]`. Column `d` belongs to the key `d` tokens back.
    pub rel_table: &'a [f32],
    /// `[num_heads, hidden_size]`.
    pub sink_proj: &'a [f32],
    /// `[num_heads * head_dim, hidden_size]`, followed by sigmoid.
    pub gate_proj: &'a [f32],
    /// `[hidden_size, num_heads * head_dim]`.
    pub o_proj: &'a [f32],
}

/// Finished keys and values of one visit for one sequence, oldest token first.
/// A window layer keeps only the tokens that the next token can see.
/// Use the same weights for the lifetime of a state.
#[derive(Debug, Clone, PartialEq)]
pub struct State { config: Config, keys: Vec<f32>, values: Vec<f32>, seen: usize }
impl State {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, keys: Vec::new(), values: Vec::new(), seen: 0 })
    }
    /// Tokens read so far. This is the position of the next token.
    pub fn position(&self) -> usize { self.seen }
    /// Tokens whose keys and values are kept.
    pub fn len(&self) -> usize { self.keys.len() / self.config.kv_width() }
    pub fn is_empty(&self) -> bool { self.keys.is_empty() }
    /// Kept keys in `[token][kv_head][head_dim]` order, after convolution and clip.
    pub fn keys(&self) -> &[f32] { &self.keys }
    /// Kept values in the same order, after convolution and plain norm.
    pub fn values(&self) -> &[f32] { &self.values }
}

/// BF16 results of one token, as f32. Use them to check one stage at a time.
#[derive(Debug, Clone, PartialEq)]
pub struct Parts {
    /// `[num_heads][head_dim]`, after scale, clip and position scale.
    pub query: Vec<f32>,
    /// `[num_kv_heads][head_dim]`, the finished key that the state keeps.
    pub key: Vec<f32>,
    /// `[num_kv_heads][head_dim]`, the finished value that the state keeps.
    pub value: Vec<f32>,
    /// `[num_heads][rel_dim]`, after scale, clip and position scale.
    pub relative: Vec<f32>,
    /// `[num_heads]`.
    pub sink: Vec<f32>,
    /// `[num_heads][visible keys]`, oldest key first: scores after the third rounding.
    pub scores: Vec<f32>,
    /// `[num_heads][visible keys]`: softmax weights after their rounding.
    pub weights: Vec<f32>,
    /// `[num_heads][head_dim]`: weighted values before the gate.
    pub mixed: Vec<f32>,
    /// `[num_heads][head_dim]`: after the gate.
    pub gated: Vec<f32>,
    /// `[hidden_size]`.
    pub output: Vec<f32>,
}

/// Scalar host reference for one layer.
pub struct Reference<'a> { config: Config, weights: Weights<'a> }
impl<'a> Reference<'a> {
    pub fn new(config: Config, weights: Weights<'a>) -> Result<Self> {
        config.validate()?;
        let (d, w) = (config, weights);
        let (q, kv, h) = (d.query_width(), d.kv_width(), d.hidden_size);
        for (data, count) in [(w.q_proj, q * h), (w.q_scale, q), (w.k_proj, kv * h), (w.v_proj, kv * h),
            (w.rel_proj, d.num_heads * d.rel_dim * h), (w.rel_scale, d.num_heads * d.rel_dim),
            (w.rel_table, d.rel_dim * d.reach.rel_extent()), (w.sink_proj, d.num_heads * h),
            (w.gate_proj, q * h), (w.o_proj, h * q)] {
            need(data.len() == count, "weight shape differs from config")?;
            exact(data)?;
        }
        for conv in [w.k_conv, w.v_conv] {
            need(conv.channels() == kv && conv.taps() == d.conv_taps, "convolution shape differs from config")?;
        }
        Ok(Self { config, weights })
    }
    /// Advance one token and return the attention output `[hidden_size]`.
    /// `query_input` feeds the query, the relative term, the sink and the
    /// output gate. `key_input` and `value_input` feed the key and the value.
    /// All three are exact BF16 vectors. An error leaves both states unchanged.
    pub fn step(&self, state: &mut State, conv: &mut VisitHistories, query_input: &[f32], key_input: &[f32], value_input: &[f32]) -> Result<Vec<f32>> {
        Ok(self.step_parts(state, conv, query_input, key_input, value_input)?.output)
    }
    /// Scan token-major `[tokens][hidden_size]` chunks with the exact step
    /// arithmetic. An empty chunk is valid. An error leaves both states unchanged.
    #[allow(clippy::manual_is_multiple_of)] // A zero divisor must still panic in `%`.
    pub fn scan(&self, state: &mut State, conv: &mut VisitHistories, query_input: &[f32], key_input: &[f32], value_input: &[f32]) -> Result<Vec<f32>> {
        let h = self.config.hidden_size;
        need(query_input.len() % h == 0 && key_input.len() == query_input.len() && value_input.len() == query_input.len(),
            "chunk width differs from config")?;
        let (mut next, mut next_conv) = (state.clone(), conv.clone());
        let mut output = Vec::new();
        output.try_reserve_exact(query_input.len()).map_err(|_| invalid("sink attention: cannot reserve chunk output"))?;
        for ((q, k), v) in query_input.chunks_exact(h).zip(key_input.chunks_exact(h)).zip(value_input.chunks_exact(h)) {
            output.extend(self.step_parts(&mut next, &mut next_conv, q, k, v)?.output);
        }
        *state = next; *conv = next_conv;
        Ok(output)
    }
    /// Advance one token and return every stage. See [`Reference::step`].
    #[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
    pub fn step_parts(&self, state: &mut State, conv: &mut VisitHistories, query_input: &[f32], key_input: &[f32], value_input: &[f32]) -> Result<Parts> {
        let (d, w) = (self.config, self.weights);
        need(state.config == d, "state config differs from reference")?;
        for x in [query_input, key_input, value_input] {
            need(x.len() == d.hidden_size, "input width differs from config")?;
            exact(x)?;
        }
        let tau = d.reach.position_scale(state.seen);
        let query = self.head_states(&rows(w.q_proj, query_input)?, w.q_scale, d.head_dim, d.qk_clip, tau)?;
        let relative = self.head_states(&rows(w.rel_proj, query_input)?, w.rel_scale, d.rel_dim, d.rel_clip, tau)?;
        let sink = rows(w.sink_proj, query_input)?;

        // Key and value: project, convolve, then clip the key and norm the value by head.
        let key_raw = rows(w.k_proj, key_input)?;
        let value_raw = rows(w.v_proj, value_input)?;
        let key_conv = w.k_conv.apply(conv.get(Slot::Key), &key_raw, key_input)?;
        let value_conv = w.v_conv.apply(conv.get(Slot::Value), &value_raw, value_input)?;
        let (mut key, mut value) = (Vec::with_capacity(d.kv_width()), Vec::with_capacity(d.kv_width()));
        for head in key_conv.chunks_exact(d.head_dim) { for v in length_clip(head, d.qk_clip)? { key.push(round(v)?); } }
        for head in value_conv.chunks_exact(d.head_dim) { for v in plain_norm(head, d.rms_norm_eps)? { value.push(round(v)?); } }

        // Visible keys in position order: the kept tokens, then this token.
        let (kept, kv) = (state.len(), d.kv_width());
        let visible = kept + 1;
        let (scale, extent) = (score_scale(d.head_dim), d.reach.rel_extent());
        let mut scores = Vec::with_capacity(d.num_heads * visible);
        let mut weights = Vec::with_capacity(d.num_heads * visible);
        let mut mixed = Vec::with_capacity(d.query_width());
        for head in 0..d.num_heads {
            let q = &query[head * d.head_dim..(head + 1) * d.head_dim];
            let r = &relative[head * d.rel_dim..(head + 1) * d.rel_dim];
            let shared = shared_head(head, d.num_heads, d.num_kv_heads);
            // A score is rounded to BF16 three times: after the dot product, after
            // the scale and after the relative term (source lines 126-128).
            let mut head_scores = Vec::with_capacity(visible);
            for token in 0..visible {
                let distance = visible - 1 - token;
                let bias = if distance < extent {
                    let mut sum = 0.0f32;
                    for i in 0..d.rel_dim { sum += r[i] * w.rel_table[i * extent + distance]; }
                    round(sum)?
                } else { 0.0 };
                let k = if token < kept { head_row(&state.keys, kv, d.head_dim, token, shared) } else { head_row(&key, kv, d.head_dim, 0, shared) };
                let product = dot(q, k)?;
                head_scores.push(round(round(product * scale)? + bias)?);
            }
            // Softmax in f32: take the largest value out, then the exponentials. The
            // sum runs in the distance order and the sink is added last. The sink has
            // no value. The weights are rounded before they meet the values.
            let mut max = sink[head];
            for &score in &head_scores { max = max.max(score); }
            let exps: Vec<f32> = head_scores.iter().map(|&score| exp(score - max)).collect();
            let total = distance_sum(visible, |distance| exps[visible - 1 - distance]) + exp(sink[head] - max);
            need(total.is_finite() && total > 0.0, "invalid softmax sum")?;
            let head_weights = exps.iter().map(|&e| round(e / total)).collect::<Result<Vec<f32>>>()?;
            let value_of = |distance: usize| {
                let token = visible - 1 - distance;
                if token < kept { head_row(&state.values, kv, d.head_dim, token, shared) } else { head_row(&value, kv, d.head_dim, 0, shared) }
            };
            let out: Vec<f32> = (0..d.head_dim).map(|i| distance_sum(visible, |distance| head_weights[visible - 1 - distance] * value_of(distance)[i])).collect();
            for sum in out { mixed.push(round(sum)?); }
            scores.extend(head_scores); weights.extend(head_weights);
        }

        // Output gate from the query input, then the output projection.
        let gate = rows(w.gate_proj, query_input)?;
        let gated = mixed.iter().zip(&gate).map(|(&o, &g)| round(o * sigmoid(g))).collect::<Result<Vec<f32>>>()?;
        let output = rows(w.o_proj, &gated)?;

        // Commit. Nothing above changed a state.
        let grow = |data: &mut Vec<f32>| data.try_reserve(kv).map_err(|_| invalid("sink attention: cannot reserve key and value storage"));
        grow(&mut state.keys)?; grow(&mut state.values)?;
        conv.get_mut(Slot::Key).push(&key_raw)?;
        conv.get_mut(Slot::Value).push(&value_raw)?;
        state.keys.extend_from_slice(&key);
        state.values.extend_from_slice(&value);
        state.seen += 1;
        if let Reach::Window { window } = d.reach {
            let extra = state.len().saturating_sub(window - 1);
            state.keys.drain(..extra * kv);
            state.values.drain(..extra * kv);
        }
        Ok(Parts { query, key, value, relative, sink, scores, weights, mixed, gated, output })
    }
    /// Multiply each head by its learned scale, clip its length and round to BF16.
    /// A full layer then multiplies by the position scale in f32 and rounds
    /// again (source lines 198-202 and 233-236).
    fn head_states(&self, raw: &[f32], scales: &[f32], width: usize, threshold: f32, tau: Option<f32>) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(raw.len());
        for (head, scale) in raw.chunks_exact(width).zip(scales.chunks_exact(width)) {
            let scaled: Vec<f32> = head.iter().zip(scale).map(|(&v, &s)| v * s).collect();
            need(scaled.iter().all(|v| v.is_finite()), "scaled state is not finite")?;
            for v in length_clip(&scaled, threshold)? {
                let clipped = round(v)?;
                output.push(match tau { Some(tau) => round(clipped * tau)?, None => clipped });
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
