//! Host CPU form of a DSpark / MTP speculative draft model with a recurrent
//! Markov head (`mtp.safetensors`).
//!
//! Contract source: `N/ink/MTP-SPEC.md` (read from the maker's serving engine
//! for the math only; zero code copied):
//! - `srt/models/inkling2_mtp.py`: lines 31-101 (Markov head, FP64 head fold),
//!   103-162 (layer layout and context projections), 209-259 (block embedding,
//!   `mix_rmsnorm`, `f_rnn`), 284-345 (context KV write), 411-413 (`fuse_weight`).
//! - `kernels/ops/speculative/inkling2_mtp.py`: lines 7-57 (`mix_rmsnorm`),
//!   60-160 (BF16-rounded RNN gate update and input concatenation).
//! - `srt/models/dflash.py`: lines 327-358 (non-causal block attention),
//!   369-609 (QK-norm, NeoX RoPE, GQA attention), 839-898 (SwiGLU MLP),
//!   1043-1140 and 1438-1459 (pre-norm residual layer stack and final norm).
//! - `srt/speculative/dspark_components/dspark_draft.py`: lines 83-101 (anchor
//!   row drop when `sample_from_anchor` is false), 257-302 (Markov walk).
//!
//! All dimensions, layer schedules, tap indices, window sizes, vocabulary
//! bounds, and softcap parameters come from `config.json` and tensor shapes.
//! Shared math (`exp`, `sigmoid`, `tanh`, `rn`) comes from `fixed_math`.
use crate::{
    Result, invalid,
    flash::nvfp4::bf16_to_f32,
    json::{self, field},
    norm_cap::{SoftCap, exp, round_bf16, sigmoid, tanh, weighted_norm},
    safetensors::{self, Dtype, TensorFile},
    shapes::{CONFIG_BYTES, HEADER_BYTES, MAX_HIDDEN, MAX_LAYERS, MAX_SHARDS, MAX_TENSOR_RANK, MAX_TENSORS},
};
use std::path::Path;

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("mtp draft: {message}"))) }
}

fn to_usize(v: u64, what: &str) -> Result<usize> {
    usize::try_from(v).map_err(|_| invalid(format!("mtp draft: {what} exceeds usize")))
}

fn bf16_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    let mut out = weighted_norm(x, weight, eps)?;
    for v in &mut out { *v = round_bf16(*v); }
    need(out.iter().all(|v| v.is_finite()), "norm result is not finite in BF16")?;
    Ok(out)
}

fn bf16_bits(v: f32) -> u16 { (round_bf16(v).to_bits() >> 16) as u16 }

fn f32_to_bf16_vec(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&v| bf16_bits(v)).collect()
}

fn bf16_bytes_to_u16(bytes: &[u8]) -> Result<Vec<u16>> {
    need(bytes.len().is_multiple_of(2), "BF16 byte length must be even")?;
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*pair);
        need(bf16_to_f32(bits).is_finite(), "BF16 tensor holds a non-finite value")?;
        out.push(bits);
    }
    Ok(out)
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    need(bytes.len().is_multiple_of(2), "BF16 byte length must be even")?;
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.as_chunks::<2>().0 {
        let val = bf16_to_f32(u16::from_le_bytes(*pair));
        need(val.is_finite(), "BF16 tensor holds a non-finite value")?;
        out.push(val);
    }
    Ok(out)
}

/// Configuration of the draft model, parsed from `mtp_config` and `text_config` in `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct MtpConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub local_layer_ids: Vec<usize>,
    pub sliding_window_size: usize,
    pub block_size: usize,
    pub markov_rank: usize,
    pub target_layer_ids: Vec<usize>,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub final_logit_softcapping: f64,
}

impl MtpConfig {
    /// Parse `mtp_config` and the required shared vocabulary fields from `config.json` bytes.
    pub fn from_config_json(input: &[u8]) -> Result<Self> {
        let doc = json::parse(input, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let text = field(root, "text_config")?.object()?;
        let mtp = field(root, "mtp_config")?.object()?;
        if let Some(v) = mtp.get("q_bias") { need(!v.boolean()?, "q_bias must be false")?; }
        if let Some(v) = mtp.get("o_bias") { need(!v.boolean()?, "o_bias must be false")?; }
        let head_type = field(mtp, "markov_head_type")?.string()?;
        need(head_type == "rnn_hidden", "markov_head_type must be rnn_hidden")?;
        if let Some(dt) = mtp.get("torch_dtype") {
            need(dt.string()? == "bfloat16", "mtp_config torch_dtype must be bfloat16")?;
        }
        let u = |o, k: &str| -> Result<usize> { to_usize(field(o, k)?.u64()?, k) };
        let list = |o, k: &str| -> Result<Vec<usize>> {
            field(o, k)?.array()?.iter().map(|v| to_usize(v.u64()?, k)).collect()
        };
        let hidden_size = u(mtp, "hidden_size")?;
        need(hidden_size == u(text, "hidden_size")?, "mtp_config hidden_size differs from text_config")?;
        let vocab_size = u(text, "vocab_size")?;
        let unpadded_vocab_size = match text.get("unpadded_vocab_size") {
            Some(v) => to_usize(v.u64()?, "unpadded_vocab_size")?,
            None => vocab_size,
        };
        let cfg = Self {
            hidden_size,
            num_layers: u(mtp, "num_hidden_layers")?,
            num_heads: u(mtp, "num_attention_heads")?,
            num_kv_heads: u(mtp, "num_key_value_heads")?,
            head_dim: u(mtp, "head_dim")?,
            intermediate_size: u(mtp, "intermediate_size")?,
            max_position_embeddings: u(mtp, "max_position_embeddings")?,
            rms_norm_eps: field(mtp, "rms_norm_eps")?.f64()? as f32,
            rope_theta: field(mtp, "rope_theta")?.f64()?,
            local_layer_ids: list(mtp, "local_layer_ids")?,
            sliding_window_size: u(mtp, "sliding_window_size")?,
            block_size: u(mtp, "block_size")?,
            markov_rank: u(mtp, "markov_rank")?,
            target_layer_ids: list(mtp, "target_layer_ids")?,
            vocab_size,
            unpadded_vocab_size,
            final_logit_softcapping: field(text, "final_logit_softcapping")?.f64()?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        need(self.hidden_size > 0 && self.hidden_size as u64 <= MAX_HIDDEN, "hidden_size out of bounds")?;
        need(self.num_layers > 0 && self.num_layers as u64 <= MAX_LAYERS, "num_layers out of bounds")?;
        need(self.num_heads > 0 && self.num_kv_heads > 0, "head counts must be positive")?;
        need(self.num_heads.is_multiple_of(self.num_kv_heads), "num_heads must be divisible by num_kv_heads")?;
        need(self.head_dim > 0 && self.head_dim.is_multiple_of(2), "head_dim must be positive and even")?;
        need(self.intermediate_size > 0, "intermediate_size must be positive")?;
        need(self.max_position_embeddings > 0, "max_position_embeddings must be positive")?;
        need(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0, "rms_norm_eps must be positive finite")?;
        need(self.rope_theta.is_finite() && self.rope_theta > 0.0, "rope_theta must be positive finite")?;
        need(self.block_size >= 2, "block_size must be at least 2")?;
        let latent_width = self.markov_rank;
        need(latent_width > 0, "markov_rank must be positive")?;
        need(self.target_layer_ids.len() >= 2, "target_layer_ids must have at least 2 entries")?;
        for w in self.target_layer_ids.windows(2) {
            need(w[0] < w[1], "target_layer_ids must be strictly increasing")?;
        }
        for &id in &self.local_layer_ids {
            need(id < self.num_layers, "local_layer_ids entry out of range")?;
        }
        if !self.local_layer_ids.is_empty() {
            need(self.sliding_window_size > 0, "sliding_window_size must be positive for local layers")?;
        }
        need(self.unpadded_vocab_size > 0 && self.unpadded_vocab_size <= self.vocab_size, "vocab bounds invalid")?;
        SoftCap::new(self.final_logit_softcapping, "mtp logit softcap")?;
        Ok(())
    }

    /// Number of proposed tokens per draft block (`block_size - 1`).
    pub fn gamma(&self) -> usize { self.block_size - 1 }
    pub fn q_width(&self) -> usize { self.num_heads * self.head_dim }
    pub fn kv_width(&self) -> usize { self.num_kv_heads * self.head_dim }
    pub fn is_local(&self, layer: usize) -> bool { self.local_layer_ids.contains(&layer) }
}

/// Weights of one draft transformer layer (BF16 matrices stored as `u16` bits).
#[derive(Debug, Clone)]
pub struct MtpLayerWeights {
    pub dflare_norm: Vec<f32>,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub k_norm_ctx: Vec<f32>,
    pub q_proj: Vec<u16>,
    pub k_proj: Vec<u16>,
    pub v_proj: Vec<u16>,
    pub k_proj_ctx: Vec<u16>,
    pub v_proj_ctx: Vec<u16>,
    pub o_proj: Vec<u16>,
    pub gate_proj: Vec<u16>,
    pub up_proj: Vec<u16>,
    pub down_proj: Vec<u16>,
}

/// Weights of the recurrent Markov head.
#[derive(Debug, Clone)]
pub struct MtpMarkovWeights {
    pub dflare_norm: Vec<f32>,
    pub markov_w1: Vec<u16>,
    pub rnn_proj_weight: Vec<u16>,
    pub rnn_proj_bias: Vec<f32>,
    pub hidden_out_weight: Vec<f32>,
    pub hidden_out_bias: Vec<f32>,
}

/// All raw weights required to build [`MtpModel`], decoupled from filesystem layout.
#[derive(Debug, Clone)]
pub struct MtpRawWeights {
    pub fuse_weight: Vec<f32>,
    pub mask_embed: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub layers: Vec<MtpLayerWeights>,
    pub markov: MtpMarkovWeights,
    /// First `unpadded_vocab_size` rows of the main model's unembed matrix (`[unpadded_vocab_size, hidden_size]`).
    pub lm_head: Vec<u16>,
}

/// Context KV cache of one draft layer for one sequence.
#[derive(Debug, Clone, Default)]
pub struct MtpLayerKv {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

/// Running draft state for one chat sequence: per-layer context KV and latest `f_rnn`.
#[derive(Debug, Clone)]
pub struct MtpChat {
    pub layers: Vec<MtpLayerKv>,
    pub last_f_rnn: Option<Vec<f32>>,
    pub prefix_len: usize,
}

impl MtpChat {
    pub fn new(num_layers: usize) -> Self {
        Self { layers: (0..num_layers).map(|_| MtpLayerKv::default()).collect(), last_f_rnn: None, prefix_len: 0 }
    }
    pub fn prefix_len(&self) -> usize { self.prefix_len }
}

/// Loaded host CPU draft model with precomputed `fuse_alpha`, RoPE frequencies,
/// and FP64-folded Markov output head (`lm_head @ hidden_out`).
pub struct MtpModel {
    config: MtpConfig,
    logit_cap: SoftCap,
    fuse_alpha: Vec<Vec<f32>>,
    inv_freq: Vec<f64>,
    mask_embed: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<MtpLayerWeights>,
    markov_dflare_norm: Vec<f32>,
    markov_w1: Vec<u16>,
    rnn_proj_weight: Vec<u16>,
    rnn_proj_bias: Vec<f32>,
    lm_head: Vec<u16>,
    fold_weight: Vec<u16>,
    fold_bias: Vec<f32>,
}

fn read_bf16_u16(file: &TensorFile, name: &str, expected_shape: &[usize]) -> Result<Vec<u16>> {
    let info = file.header().tensors.get(name)
        .ok_or_else(|| invalid(format!("mtp draft: missing tensor {name}")))?;
    need(info.dtype == Dtype::Bf16, &format!("tensor {name} must be BF16"))?;
    let expected_u64: Vec<u64> = expected_shape.iter().map(|&d| d as u64).collect();
    need(info.shape == expected_u64, &format!("tensor {name} shape {:?} != expected {:?}", info.shape, expected_u64))?;
    let byte_len = to_usize(info.byte_len(), "tensor byte length")?;
    let mut bytes = vec![0u8; byte_len];
    file.read_range(name, 0, &mut bytes)?;
    bf16_bytes_to_u16(&bytes)
}

fn read_bf16_f32(file: &TensorFile, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
    let info = file.header().tensors.get(name)
        .ok_or_else(|| invalid(format!("mtp draft: missing tensor {name}")))?;
    need(info.dtype == Dtype::Bf16, &format!("tensor {name} must be BF16"))?;
    let expected_u64: Vec<u64> = expected_shape.iter().map(|&d| d as u64).collect();
    need(info.shape == expected_u64, &format!("tensor {name} shape {:?} != expected {:?}", info.shape, expected_u64))?;
    let byte_len = to_usize(info.byte_len(), "tensor byte length")?;
    let mut bytes = vec![0u8; byte_len];
    file.read_range(name, 0, &mut bytes)?;
    bf16_bytes_to_f32(&bytes)
}

/// Row-wise FP32 softmax using `fixed_math::exp`, rounded to BF16.
fn compute_fuse_alpha(fuse_weight: &[f32], rows: usize, cols: usize) -> Result<Vec<Vec<f32>>> {
    need(fuse_weight.len() == rows * cols, "fuse_weight length mismatch")?;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &fuse_weight[r * cols..(r + 1) * cols];
        let mut max_v = f32::NEG_INFINITY;
        for &v in row { if v > max_v { max_v = v; } }
        let exps: Vec<f32> = row.iter().map(|&v| exp(v - max_v)).collect();
        let sum: f32 = exps.iter().copied().sum();
        need(sum.is_finite() && sum > 0.0, "fuse_weight softmax sum invalid")?;
        out.push(exps.into_iter().map(|e| round_bf16(e / sum)).collect());
    }
    Ok(out)
}

/// Pre-fold `lm_head @ hidden_out.weight` and `lm_head @ hidden_out.bias` in FP64 -> BF16,
/// matching `Inkling2MTPHead.refresh_fold`.
fn fold_markov_head(
    lm_head: &[u16],
    hidden_out_w: &[f32],
    hidden_out_b: &[f32],
    vocab: usize,
    hidden: usize,
    latent_width: usize,
    threads: usize,
) -> Result<(Vec<u16>, Vec<f32>)> {
    need(lm_head.len() == vocab * hidden, "lm_head size mismatch")?;
    need(hidden_out_w.len() == hidden * latent_width, "hidden_out.weight size mismatch")?;
    need(hidden_out_b.len() == hidden, "hidden_out.bias size mismatch")?;
    // Transpose hidden_out_w [hidden, latent_width] -> [latent_width, hidden] in f64 for cache-local dots.
    let mut proj_t = vec![0.0f64; latent_width * hidden];
    for h in 0..hidden {
        for r in 0..latent_width {
            proj_t[r * hidden + h] = hidden_out_w[h * latent_width + r] as f64;
        }
    }
    let bias_f64: Vec<f64> = hidden_out_b.iter().map(|&v| v as f64).collect();
    let workers = threads.clamp(1, vocab.max(1));
    let chunk_rows = vocab.div_ceil(workers);
    let mut fold_w = vec![0u16; vocab * latent_width];
    let mut fold_b = vec![0.0f32; vocab];

    let work = |row_start: usize, w_out: &mut [u16], b_out: &mut [f32]| {
        let mut row_f64 = vec![0.0f64; hidden];
        for (local_i, b_slot) in b_out.iter_mut().enumerate() {
            let v = row_start + local_i;
            let src = &lm_head[v * hidden..(v + 1) * hidden];
            for (dst, &bits) in row_f64.iter_mut().zip(src) {
                *dst = bf16_to_f32(bits) as f64;
            }
            let mut bsum = 0.0f64;
            for (&a, &b) in row_f64.iter().zip(&bias_f64) { bsum += a * b; }
            *b_slot = round_bf16(bsum as f32);
            let w_dst = &mut w_out[local_i * latent_width..(local_i + 1) * latent_width];
            for (r, slot) in w_dst.iter_mut().enumerate() {
                let pcol = &proj_t[r * hidden..(r + 1) * hidden];
                let mut sum = 0.0f64;
                for (&a, &b) in row_f64.iter().zip(pcol) { sum += a * b; }
                *slot = bf16_bits(sum as f32);
            }
        }
    };

    if workers == 1 {
        work(0, &mut fold_w, &mut fold_b);
    } else {
        std::thread::scope(|scope| {
            let mut w_rest = fold_w.as_mut_slice();
            let mut b_rest = fold_b.as_mut_slice();
            let mut start = 0usize;
            while !b_rest.is_empty() {
                let take = chunk_rows.min(b_rest.len());
                let (w_part, w_next) = w_rest.split_at_mut(take * latent_width);
                let (b_part, b_next) = b_rest.split_at_mut(take);
                w_rest = w_next;
                b_rest = b_next;
                let row_start = start;
                start += take;
                scope.spawn(move || work(row_start, w_part, b_part));
            }
        });
    }
    Ok((fold_w, fold_b))
}

/// Multi-row BF16 matrix multiply: `out[b][r] = round_bf16(sum_c weight[r, c] * inputs[b][c])`.
fn matmul_batch_bf16(
    weight: &[u16],
    out_dim: usize,
    in_dim: usize,
    inputs: &[Vec<f32>],
    threads: usize,
) -> Vec<Vec<f32>> {
    let batch = inputs.len();
    let workers = threads.clamp(1, out_dim.max(1));
    let mut flat = vec![0.0f32; out_dim * batch];
    let run_chunk = |row_start: usize, dst: &mut [f32]| {
        let rows = dst.len() / batch;
        for r_local in 0..rows {
            let row = &weight[(row_start + r_local) * in_dim..(row_start + r_local + 1) * in_dim];
            let out_slice = &mut dst[r_local * batch..(r_local + 1) * batch];
            if batch == 8 {
                let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let (mut s4, mut s5, mut s6, mut s7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let (i0, i1, i2, i3) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
                let (i4, i5, i6, i7) = (&inputs[4], &inputs[5], &inputs[6], &inputs[7]);
                for c in 0..in_dim {
                    let w = bf16_to_f32(row[c]);
                    s0 += w * i0[c]; s1 += w * i1[c]; s2 += w * i2[c]; s3 += w * i3[c];
                    s4 += w * i4[c]; s5 += w * i5[c]; s6 += w * i6[c]; s7 += w * i7[c];
                }
                out_slice[0] = round_bf16(s0); out_slice[1] = round_bf16(s1);
                out_slice[2] = round_bf16(s2); out_slice[3] = round_bf16(s3);
                out_slice[4] = round_bf16(s4); out_slice[5] = round_bf16(s5);
                out_slice[6] = round_bf16(s6); out_slice[7] = round_bf16(s7);
            } else if batch == 7 {
                let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let (mut s4, mut s5, mut s6) = (0.0f32, 0.0f32, 0.0f32);
                let (i0, i1, i2, i3) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
                let (i4, i5, i6) = (&inputs[4], &inputs[5], &inputs[6]);
                for c in 0..in_dim {
                    let w = bf16_to_f32(row[c]);
                    s0 += w * i0[c]; s1 += w * i1[c]; s2 += w * i2[c]; s3 += w * i3[c];
                    s4 += w * i4[c]; s5 += w * i5[c]; s6 += w * i6[c];
                }
                out_slice[0] = round_bf16(s0); out_slice[1] = round_bf16(s1);
                out_slice[2] = round_bf16(s2); out_slice[3] = round_bf16(s3);
                out_slice[4] = round_bf16(s4); out_slice[5] = round_bf16(s5);
                out_slice[6] = round_bf16(s6);
            } else {
                for (b, inp) in inputs.iter().enumerate() {
                    let mut sum = 0.0f32;
                    for c in 0..in_dim { sum += bf16_to_f32(row[c]) * inp[c]; }
                    out_slice[b] = round_bf16(sum);
                }
            }
        }
    };
    if workers == 1 {
        run_chunk(0, &mut flat);
    } else {
        let chunk_rows = out_dim.div_ceil(workers);
        std::thread::scope(|scope| {
            let mut rest = flat.as_mut_slice();
            let mut start = 0usize;
            while !rest.is_empty() {
                let take = chunk_rows.min(rest.len() / batch);
                let (part, next) = rest.split_at_mut(take * batch);
                rest = next;
                let row_start = start;
                start += take;
                scope.spawn(move || run_chunk(row_start, part));
            }
        });
    }
    let mut outputs = vec![vec![0.0f32; out_dim]; batch];
    for r in 0..out_dim {
        for b in 0..batch {
            outputs[b][r] = flat[r * batch + b];
        }
    }
    outputs
}

fn matvec_bf16(weight: &[u16], out_dim: usize, in_dim: usize, input: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(out_dim);
    for r in 0..out_dim {
        let row = &weight[r * in_dim..(r + 1) * in_dim];
        let mut sum = 0.0f32;
        for c in 0..in_dim { sum += bf16_to_f32(row[c]) * input[c]; }
        out.push(round_bf16(sum));
    }
    out
}

fn apply_neox_rope_inplace(vec: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, inv_freq: &[f64]) {
    let half = head_dim / 2;
    let mut cos_v = Vec::with_capacity(half);
    let mut sin_v = Vec::with_capacity(half);
    for &f in inv_freq {
        let angle = (pos as f64) * f;
        cos_v.push(angle.cos() as f32);
        sin_v.push(angle.sin() as f32);
    }
    for h in 0..num_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        for m in 0..half {
            let (x1, x2) = (head[m], head[half + m]);
            let (c, s) = (cos_v[m], sin_v[m]);
            head[m] = round_bf16(x1 * c - x2 * s);
            head[half + m] = round_bf16(x2 * c + x1 * s);
        }
    }
}

fn apply_head_norm_inplace(vec: &mut [f32], num_heads: usize, head_dim: usize, weight: &[f32], eps: f32) -> Result<()> {
    for h in 0..num_heads {
        let slice = &mut vec[h * head_dim..(h + 1) * head_dim];
        let normed = bf16_norm(slice, weight, eps)?;
        slice.copy_from_slice(&normed);
    }
    Ok(())
}

impl MtpModel {
    /// Construct an [`MtpModel`] from validated config and raw weights (also used by unit tests).
    pub fn from_raw(config: MtpConfig, raw: MtpRawWeights, fold_threads: usize) -> Result<Self> {
        Self::from_raw_with_fold(config,raw,|a,b,c,v,h,r|fold_markov_head(a,b,c,v,h,r,fold_threads).map_err(|e|e.to_string()))
    }
    fn from_raw_with_fold(config:MtpConfig,raw:MtpRawWeights,fold:impl FnOnce(&[u16],&[f32],&[f32],usize,usize,usize)->std::result::Result<(Vec<u16>,Vec<f32>),String>)->Result<Self> {
        config.validate()?;
        let h = config.hidden_size;
        let q_w = config.q_width();
        let kv_w = config.kv_width();
        let inter = config.intermediate_size;
        let latent_width = config.markov_rank;
        let taps = config.target_layer_ids.len();
        let logit_cap = SoftCap::new(config.final_logit_softcapping, "mtp logit softcap")?;
        need(raw.mask_embed.len() == h && raw.final_norm.len() == h, "mask_embed or final_norm length mismatch")?;
        need(raw.layers.len() == config.num_layers, "layer count mismatch")?;
        for l in &raw.layers {
            need(l.dflare_norm.len() == h && l.input_layernorm.len() == h && l.post_attention_layernorm.len() == h, "layer norm mismatch")?;
            need(l.q_norm.len() == config.head_dim && l.k_norm.len() == config.head_dim && l.k_norm_ctx.len() == config.head_dim, "qk norm mismatch")?;
            need(l.q_proj.len() == q_w * h && l.k_proj.len() == kv_w * h && l.v_proj.len() == kv_w * h, "qkv proj mismatch")?;
            need(l.k_proj_ctx.len() == kv_w * h && l.v_proj_ctx.len() == kv_w * h && l.o_proj.len() == h * q_w, "ctx/o proj mismatch")?;
            need(l.gate_proj.len() == inter * h && l.up_proj.len() == inter * h && l.down_proj.len() == h * inter, "mlp proj mismatch")?;
        }
        need(raw.markov.dflare_norm.len() == h, "markov dflare_norm mismatch")?;
        need(raw.markov.markov_w1.len() == config.vocab_size * latent_width, "markov_w1 mismatch")?;
        need(raw.markov.rnn_proj_weight.len() == (3 * latent_width) * (2 * latent_width + 2 * h), "rnn_proj.weight mismatch")?;
        need(raw.markov.rnn_proj_bias.len() == 3 * latent_width, "rnn_proj.bias mismatch")?;
        let fuse_alpha = compute_fuse_alpha(&raw.fuse_weight, config.num_layers + 1, taps)?;
        let inv_freq: Vec<f64> = (0..config.head_dim / 2)
            .map(|m| 1.0 / config.rope_theta.powf((2 * m) as f64 / config.head_dim as f64))
            .collect();
        let (fold_weight, fold_bias) = fold(
            &raw.lm_head,
            &raw.markov.hidden_out_weight,
            &raw.markov.hidden_out_bias,
            config.unpadded_vocab_size,
            h,
            latent_width,
        ).map_err(invalid)?;
        need(fold_weight.len()==config.unpadded_vocab_size*latent_width && fold_bias.len()==config.unpadded_vocab_size,"folded head shape mismatch")?;
        Ok(Self {
            config,
            logit_cap,
            fuse_alpha,
            inv_freq,
            mask_embed: raw.mask_embed,
            final_norm: raw.final_norm,
            layers: raw.layers,
            markov_dflare_norm: raw.markov.dflare_norm,
            markov_w1: raw.markov.markov_w1,
            rnn_proj_weight: raw.markov.rnn_proj_weight,
            rnn_proj_bias: raw.markov.rnn_proj_bias,
            lm_head: raw.lm_head,
            fold_weight,
            fold_bias,
        })
    }

    /// Open `config.json`, `mtp.safetensors`, and the main model's unembed head from `model_dir`.
    pub fn open(model_dir: &Path, fold_threads: usize) -> Result<Self> {
        Self::open_with_fold(model_dir,|a,b,c,v,h,r|fold_markov_head(a,b,c,v,h,r,fold_threads).map_err(|e|e.to_string()))
    }
    /// Read the shipped weights and run the supplied load-time head product once.
    pub fn open_with_fold(model_dir:&Path,fold:impl FnOnce(&[u16],&[f32],&[f32],usize,usize,usize)->std::result::Result<(Vec<u16>,Vec<f32>),String>)->Result<Self> {
        let root = model_dir.canonicalize()?;
        let config_bytes = std::fs::read(root.join("config.json"))?;
        let config = MtpConfig::from_config_json(&config_bytes)?;
        let limits = safetensors::Limits {
            header_bytes: HEADER_BYTES,
            tensors: MAX_TENSORS,
            rank: MAX_TENSOR_RANK,
            shards: MAX_SHARDS,
        };
        let mtp_path = root.join("mtp.safetensors").canonicalize()?;
        need(mtp_path.starts_with(&root), "mtp.safetensors leaves the model folder")?;
        let mtp_file = TensorFile::open(&mtp_path, limits)?;

        let h = config.hidden_size;
        let q_w = config.q_width();
        let kv_w = config.kv_width();
        let inter = config.intermediate_size;
        let rank = config.markov_rank;
        let taps = config.target_layer_ids.len();

        let fuse_weight = read_bf16_f32(&mtp_file, "fuse_weight", &[config.num_layers + 1, taps])?;
        let mask_embed = read_bf16_f32(&mtp_file, "mask_embed", &[h])?;
        let final_norm = read_bf16_f32(&mtp_file, "norm.weight", &[h])?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("layers.{i}");
            layers.push(MtpLayerWeights {
                dflare_norm: read_bf16_f32(&mtp_file, &format!("{p}.dflare_norm.weight"), &[h])?,
                input_layernorm: read_bf16_f32(&mtp_file, &format!("{p}.input_layernorm.weight"), &[h])?,
                post_attention_layernorm: read_bf16_f32(&mtp_file, &format!("{p}.post_attention_layernorm.weight"), &[h])?,
                q_norm: read_bf16_f32(&mtp_file, &format!("{p}.self_attn.q_norm.weight"), &[config.head_dim])?,
                k_norm: read_bf16_f32(&mtp_file, &format!("{p}.self_attn.k_norm.weight"), &[config.head_dim])?,
                k_norm_ctx: read_bf16_f32(&mtp_file, &format!("{p}.self_attn.k_norm_ctx.weight"), &[config.head_dim])?,
                q_proj: read_bf16_u16(&mtp_file, &format!("{p}.self_attn.q_proj.weight"), &[q_w, h])?,
                k_proj: read_bf16_u16(&mtp_file, &format!("{p}.self_attn.k_proj.weight"), &[kv_w, h])?,
                v_proj: read_bf16_u16(&mtp_file, &format!("{p}.self_attn.v_proj.weight"), &[kv_w, h])?,
                k_proj_ctx: read_bf16_u16(&mtp_file, &format!("{p}.self_attn.k_proj_ctx.weight"), &[kv_w, h])?,
                v_proj_ctx: read_bf16_u16(&mtp_file, &format!("{p}.self_attn.v_proj_ctx.weight"), &[kv_w, h])?,
                o_proj: read_bf16_u16(&mtp_file, &format!("{p}.self_attn.o_proj.weight"), &[h, q_w])?,
                gate_proj: read_bf16_u16(&mtp_file, &format!("{p}.mlp.gate_proj.weight"), &[inter, h])?,
                up_proj: read_bf16_u16(&mtp_file, &format!("{p}.mlp.up_proj.weight"), &[inter, h])?,
                down_proj: read_bf16_u16(&mtp_file, &format!("{p}.mlp.down_proj.weight"), &[h, inter])?,
            });
        }
        let markov = MtpMarkovWeights {
            dflare_norm: read_bf16_f32(&mtp_file, "markov_head.dflare_norm.weight", &[h])?,
            markov_w1: read_bf16_u16(&mtp_file, "markov_head.markov_w1.weight", &[config.vocab_size, rank])?,
            rnn_proj_weight: read_bf16_u16(&mtp_file, "markov_head.rnn_proj.weight", &[3 * rank, 2 * rank + 2 * h])?,
            rnn_proj_bias: read_bf16_f32(&mtp_file, "markov_head.rnn_proj.bias", &[3 * rank])?,
            hidden_out_weight: read_bf16_f32(&mtp_file, "markov_head.hidden_out.weight", &[h, rank])?,
            hidden_out_bias: read_bf16_f32(&mtp_file, "markov_head.hidden_out.bias", &[h])?,
        };
        // Read the shared unembed head (first unpadded_vocab_size rows) through loop_host's Folder.
        let (_, index, folder) = crate::loop_host::open_folder(&root)?;
        let expected = index.expected_main().map_err(|e| invalid(format!("mtp draft: tensor index: {e}")))?;
        let (head_name, head_info) = expected
            .iter()
            .find(|(_, e)| e.role == crate::plan::inkling_index::StoredRole::Head && e.place == crate::shapes::Place::Model)
            .ok_or_else(|| invalid("mtp draft: main model has no Head tensor"))?;
        need(head_info.dtype == Dtype::Bf16, "main model Head tensor must be BF16")?;
        need(head_info.shape == [config.vocab_size as u64, h as u64], "main model Head shape mismatch")?;
        let head_bytes_len = config.unpadded_vocab_size * h * 2;
        let mut head_bytes = vec![0u8; head_bytes_len];
        crate::loop_host::Source::read(&folder, head_name, 0, &mut head_bytes)?;
        let lm_head = bf16_bytes_to_u16(&head_bytes)?;

        Self::from_raw_with_fold(
            config,
            MtpRawWeights { fuse_weight, mask_embed, final_norm, layers, markov, lm_head },
            fold,
        )
    }

    pub fn config(&self) -> &MtpConfig { &self.config }
    pub fn chat(&self) -> MtpChat { MtpChat::new(self.config.num_layers) }
    /// The parts of the loaded model, for a backend that runs the large products itself
    /// (`spark_tp::loop_decoder::draft`). The small steps stay in the functions below.
    pub fn head(&self)->&[u16] {&self.lm_head}
    pub fn layers(&self) -> &[MtpLayerWeights] { &self.layers }
    pub fn mask_embed(&self) -> &[f32] { &self.mask_embed }
    pub fn final_norm(&self) -> &[f32] { &self.final_norm }
    pub fn markov_norm(&self) -> &[f32] { &self.markov_dflare_norm }
    pub fn markov_w1(&self) -> &[u16] { &self.markov_w1 }
    pub fn fuse_alpha(&self)->&[Vec<f32>] {&self.fuse_alpha}
    pub fn rope_phases(&self,start:usize,rows:usize)->Vec<f32> {
        let mut phases=Vec::with_capacity(rows*self.config.head_dim);
        for p in start..start+rows {
            phases.extend(self.inv_freq.iter().map(|f|(p as f64*f).cos() as f32));
            phases.extend(self.inv_freq.iter().map(|f|(p as f64*f).sin() as f32));
        }
        phases
    }
    /// `rnn_proj`: the weight `[3 * rank, 2 * rank + 2 * hidden]` and its bias.
    pub fn rnn_proj(&self) -> (&[u16], &[f32]) { (&self.rnn_proj_weight, &self.rnn_proj_bias) }
    /// The folded head of the markov step, `lm_head @ hidden_out`: the weight `[unpadded vocab, rank]` and the bias.
    pub fn fold(&self) -> (&[u16], &[f32]) { (&self.fold_weight, &self.fold_bias) }
    /// The weighted norm of the draft model, rounded to BF16.
    pub fn norm(&self, x: &[f32], weight: &[f32]) -> Result<Vec<f32>> { bf16_norm(x, weight, self.config.rms_norm_eps) }
    /// The norm of each head of `heads` heads of `vec`, in place.
    pub fn head_norm(&self, vec: &mut [f32], heads: usize, weight: &[f32]) -> Result<()> { apply_head_norm_inplace(vec, heads, self.config.head_dim, weight, self.config.rms_norm_eps) }
    /// The rotation of each head of `heads` heads of `vec` at position `pos`, in place.
    pub fn rope(&self, vec: &mut [f32], heads: usize, pos: usize) { apply_neox_rope_inplace(vec, heads, self.config.head_dim, pos, &self.inv_freq) }
    /// One step of the recurrent markov state from the `rnn_proj` output with its bias: the new state and the latent.
    pub fn rnn_step(&self, proj: &[f32], state: &mut [f32], latent: &mut [f32]) {
        let rank = self.config.markov_rank;
        for j in 0..rank {
            let (gate_raw, cand_raw, out_raw) = (proj[j], proj[rank + j], proj[2 * rank + j]);
            let g = round_bf16(sigmoid(gate_raw));
            let kept = round_bf16(g * state[j]);
            let blend = round_bf16(round_bf16(1.0f32 - g) * round_bf16(tanh(cand_raw)));
            state[j] = round_bf16(kept + blend);
            latent[j] = round_bf16(tanh(out_raw));
        }
    }

    /// Mix the captured target taps for one token using `fuse_alpha[row]` and `norm_weight`.
    pub fn mix_rmsnorm(&self, taps: &[&[f32]], row: usize, norm_weight: &[f32]) -> Result<Vec<f32>> {
        let h = self.config.hidden_size;
        let alpha = self.fuse_alpha.get(row).ok_or_else(|| invalid("mtp draft: fuse row out of range"))?;
        need(taps.len() == alpha.len(), "tap count differs from fuse_weight columns")?;
        for &tap in taps { need(tap.len() == h, "tap vector length differs from hidden_size")?; }
        let mut mixed = vec![0.0f32; h];
        for (k, &tap) in taps.iter().enumerate() {
            let a = alpha[k];
            // The taps are captured as BF16 (`inkling2.py` lines 285 and 289) and widened for the mix.
            for i in 0..h { mixed[i] += round_bf16(tap[i]) * a; }
        }
        bf16_norm(&mixed, norm_weight, self.config.rms_norm_eps)
    }

    /// Commit one main-model token's captured tap vectors into the draft KV cache and update `f_rnn`.
    pub fn push_context(&self, chat: &mut MtpChat, taps: &[&[f32]]) -> Result<()> {
        need(chat.layers.len() == self.config.num_layers, "chat layer count mismatch")?;
        let pos = chat.prefix_len;
        need(pos < self.config.max_position_embeddings, "context position exceeds max_position_embeddings")?;
        let h = self.config.hidden_size;
        let kv_w = self.config.kv_width();
        for (i, layer) in self.layers.iter().enumerate() {
            let ctx = self.mix_rmsnorm(taps, i, &layer.dflare_norm)?;
            let mut k = matvec_bf16(&layer.k_proj_ctx, kv_w, h, &ctx);
            let v = matvec_bf16(&layer.v_proj_ctx, kv_w, h, &ctx);
            apply_head_norm_inplace(&mut k, self.config.num_kv_heads, self.config.head_dim, &layer.k_norm_ctx, self.config.rms_norm_eps)?;
            apply_neox_rope_inplace(&mut k, self.config.num_kv_heads, self.config.head_dim, pos, &self.inv_freq);
            chat.layers[i].k.push(k);
            chat.layers[i].v.push(v);
        }
        let f_rnn = self.mix_rmsnorm(taps, self.config.num_layers, &self.markov_dflare_norm)?;
        chat.last_f_rnn = Some(f_rnn);
        chat.prefix_len += 1;
        Ok(())
    }

    /// Run the 3-layer draft backbone on one block of `block_size` positions starting at `chat.prefix_len`.
    /// Returns the final normalized hidden rows for the `gamma = block_size - 1` masked positions.
    pub fn forward_block(&self, chat: &MtpChat, anchor_embed: &[f32], threads: usize) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let bsize = cfg.block_size;
        let p = chat.prefix_len;
        need(p > 0, "forward_block requires at least one committed context token")?;
        need(p + bsize <= cfg.max_position_embeddings, "block exceeds max_position_embeddings")?;
        need(anchor_embed.len() == h, "anchor_embed length differs from hidden_size")?;
        let threads = threads.max(1);

        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(bsize);
        hidden.push(anchor_embed.to_vec());
        for _ in 1..bsize { hidden.push(self.mask_embed.clone()); }
        let mut residual: Option<Vec<Vec<f32>>> = None;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-norm with fused residual add (Qwen3/DFlash order).
            let x = match residual.as_mut() {
                None => {
                    let norm_in: Vec<Vec<f32>> = hidden.iter()
                        .map(|row| bf16_norm(row, &layer.input_layernorm, cfg.rms_norm_eps))
                        .collect::<Result<_>>()?;
                    residual = Some(hidden);
                    norm_in
                }
                Some(res) => {
                    for (r_row, h_row) in res.iter_mut().zip(&hidden) {
                        for (rv, &hv) in r_row.iter_mut().zip(h_row) { *rv = round_bf16(*rv + hv); }
                    }
                    res.iter()
                        .map(|row| bf16_norm(row, &layer.input_layernorm, cfg.rms_norm_eps))
                        .collect::<Result<_>>()?
                }
            };

            let mut q_blk = matmul_batch_bf16(&layer.q_proj, cfg.q_width(), h, &x, threads);
            let mut k_blk = matmul_batch_bf16(&layer.k_proj, cfg.kv_width(), h, &x, threads);
            let v_blk = matmul_batch_bf16(&layer.v_proj, cfg.kv_width(), h, &x, threads);
            for r in 0..bsize {
                apply_head_norm_inplace(&mut q_blk[r], cfg.num_heads, cfg.head_dim, &layer.q_norm, cfg.rms_norm_eps)?;
                apply_head_norm_inplace(&mut k_blk[r], cfg.num_kv_heads, cfg.head_dim, &layer.k_norm, cfg.rms_norm_eps)?;
                apply_neox_rope_inplace(&mut q_blk[r], cfg.num_heads, cfg.head_dim, p + r, &self.inv_freq);
                apply_neox_rope_inplace(&mut k_blk[r], cfg.num_kv_heads, cfg.head_dim, p + r, &self.inv_freq);
            }

            // Non-causal (bidirectional) block attention over [win_start..p) context KV + all block KV.
            let win_start = if cfg.is_local(layer_idx) {
                p.saturating_sub(cfg.sliding_window_size.saturating_sub(1))
            } else {
                0
            };
            let ctx_k = &chat.layers[layer_idx].k[win_start..p];
            let ctx_v = &chat.layers[layer_idx].v[win_start..p];
            let n_keys = ctx_k.len() + bsize;
            let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
            let heads_per_kv = cfg.num_heads / cfg.num_kv_heads;
            let mut attn_concat = vec![vec![0.0f32; cfg.q_width()]; bsize];
            let mut scores = vec![0.0f32; n_keys];
            let mut weights = vec![0.0f32; n_keys];

            for r in 0..bsize {
                for qh in 0..cfg.num_heads {
                    let kvh = qh / heads_per_kv;
                    let q_slice = &q_blk[r][qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
                    let kv_off = kvh * cfg.head_dim;
                    let mut max_s = f32::NEG_INFINITY;
                    for (ci, ck) in ctx_k.iter().enumerate() {
                        let k_slice = &ck[kv_off..kv_off + cfg.head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..cfg.head_dim { dot += q_slice[d] * k_slice[d]; }
                        let s = dot * scale;
                        scores[ci] = s;
                        if s > max_s { max_s = s; }
                    }
                    for (bi, bk) in k_blk.iter().enumerate() {
                        let k_slice = &bk[kv_off..kv_off + cfg.head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..cfg.head_dim { dot += q_slice[d] * k_slice[d]; }
                        let s = dot * scale;
                        scores[ctx_k.len() + bi] = s;
                        if s > max_s { max_s = s; }
                    }
                    let mut denom = 0.0f32;
                    for i in 0..n_keys {
                        let e = exp(scores[i] - max_s);
                        weights[i] = e;
                        denom += e;
                    }
                    let inv_d = 1.0f32 / denom;
                    let out_slice = &mut attn_concat[r][qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
                    for (ci, cv) in ctx_v.iter().enumerate() {
                        let w = weights[ci] * inv_d;
                        let v_slice = &cv[kv_off..kv_off + cfg.head_dim];
                        for d in 0..cfg.head_dim { out_slice[d] += w * v_slice[d]; }
                    }
                    for (bi, bv) in v_blk.iter().enumerate() {
                        let w = weights[ctx_k.len() + bi] * inv_d;
                        let v_slice = &bv[kv_off..kv_off + cfg.head_dim];
                        for d in 0..cfg.head_dim { out_slice[d] += w * v_slice[d]; }
                    }
                    for v in out_slice.iter_mut() { *v = round_bf16(*v); }
                }
            }

            let attn_out = matmul_batch_bf16(&layer.o_proj, h, cfg.q_width(), &attn_concat, threads);
            let res = residual.as_mut().unwrap();
            for (r_row, a_row) in res.iter_mut().zip(&attn_out) {
                for (rv, &av) in r_row.iter_mut().zip(a_row) { *rv = round_bf16(*rv + av); }
            }
            let y: Vec<Vec<f32>> = res.iter()
                .map(|row| bf16_norm(row, &layer.post_attention_layernorm, cfg.rms_norm_eps))
                .collect::<Result<_>>()?;

            let gate = matmul_batch_bf16(&layer.gate_proj, cfg.intermediate_size, h, &y, threads);
            let up = matmul_batch_bf16(&layer.up_proj, cfg.intermediate_size, h, &y, threads);
            let mut act = vec![vec![0.0f32; cfg.intermediate_size]; bsize];
            for r in 0..bsize {
                for j in 0..cfg.intermediate_size {
                    let g = gate[r][j];
                    let silu_g = round_bf16(g * sigmoid(g));
                    act[r][j] = round_bf16(silu_g * up[r][j]);
                }
            }
            hidden = matmul_batch_bf16(&layer.down_proj, h, cfg.intermediate_size, &act, threads);
        }

        // Final residual add + norm, then drop anchor row 0 (sample_from_anchor = false).
        let res = residual.as_mut().unwrap();
        for (r_row, h_row) in res.iter_mut().zip(&hidden) {
            for (rv, &hv) in r_row.iter_mut().zip(h_row) { *rv = round_bf16(*rv + hv); }
        }
        let mut draft_hidden = Vec::with_capacity(cfg.gamma());
        for row in res.iter().skip(1) {
            draft_hidden.push(bf16_norm(row, &self.final_norm, cfg.rms_norm_eps)?);
        }
        Ok(draft_hidden)
    }

    /// Run the recurrent Markov head walk on `draft_hidden` (`gamma` rows) and return `gamma` greedy tokens.
    pub fn markov_walk_greedy(
        &self,
        chat: &MtpChat,
        anchor_token: u32,
        draft_hidden: &[Vec<f32>],
        threads: usize,
    ) -> Result<Vec<u32>> {
        let cfg = &self.config;
        let gamma = cfg.gamma();
        let h = cfg.hidden_size;
        let rank = cfg.markov_rank;
        let vocab = cfg.unpadded_vocab_size;
        need(draft_hidden.len() == gamma, "draft_hidden row count differs from gamma")?;
        need((anchor_token as usize) < cfg.vocab_size, "anchor_token outside vocab_size")?;
        let f_rnn = chat.last_f_rnn.as_ref().ok_or_else(|| invalid("mtp draft: chat has no f_rnn"))?;
        let threads = threads.max(1);

        // Compute base logits for all gamma draft rows in one pass over lm_head.
        let base_logits = matmul_batch_bf16(&self.lm_head, vocab, h, draft_hidden, threads);

        let mut state = vec![0.0f32; rank];
        let mut prev_token = anchor_token as usize;
        let mut proposed = Vec::with_capacity(gamma);
        let in_width = 2 * rank + 2 * h;
        let mut rnn_in = vec![0.0f32; in_width];
        let mut latent = vec![0.0f32; rank];

        for step in 0..gamma {
            need(prev_token < cfg.vocab_size, "prev_token outside vocab_size")?;
            rnn_in[..rank].copy_from_slice(&state);
            let w1_row = &self.markov_w1[prev_token * rank..(prev_token + 1) * rank];
            for j in 0..rank { rnn_in[rank + j] = bf16_to_f32(w1_row[j]); }
            rnn_in[2 * rank..2 * rank + h].copy_from_slice(&draft_hidden[step]);
            rnn_in[2 * rank + h..].copy_from_slice(f_rnn);

            // rnn_proj: [3 * rank, in_width] + bias
            // One linear step with its bias, rounded to BF16 once.
            let proj: Vec<f32> = (0..3 * rank).map(|r| {
                let row = &self.rnn_proj_weight[r * in_width..(r + 1) * in_width];
                let mut sum = 0.0f32;
                for c in 0..in_width { sum += bf16_to_f32(row[c]) * rnn_in[c]; }
                round_bf16(sum + self.rnn_proj_bias[r])
            }).collect();

            for j in 0..rank {
                let gate_raw = proj[j];
                let cand_raw = proj[rank + j];
                let out_raw = proj[2 * rank + j];
                let g = round_bf16(sigmoid(gate_raw));
                let kept = round_bf16(g * state[j]);
                let blend = round_bf16(round_bf16(1.0f32 - g) * round_bf16(tanh(cand_raw)));
                state[j] = round_bf16(kept + blend);
                latent[j] = round_bf16(tanh(out_raw));
            }

            // Find argmax of softcap(base_logits[step][v] + bias[v]) over v in 0..vocab.
            let base_row = &base_logits[step];
            let best_token = self.step_argmax(base_row, &latent, threads);
            proposed.push(best_token as u32);
            prev_token = best_token;
        }
        Ok(proposed)
    }

    /// Parallel argmax over `0..unpadded_vocab_size` with deterministic lowest-id tie breaking.
    fn step_argmax(&self, base_row: &[f32], latent: &[f32], threads: usize) -> usize {
        let vocab = self.config.unpadded_vocab_size;
        let rank = self.config.markov_rank;
        let cap = self.logit_cap;
        let workers = threads.clamp(1, vocab.max(1));
        let eval_range = |start: usize, end: usize| -> (usize, f32) {
            let mut best_id = start;
            let mut best_val = f32::NEG_INFINITY;
            for (v, &base) in base_row.iter().enumerate().take(end).skip(start) {
                let fw = &self.fold_weight[v * rank..(v + 1) * rank];
                let mut dot = 0.0f32;
                for j in 0..rank { dot += bf16_to_f32(fw[j]) * latent[j]; }
                let bias = round_bf16(self.fold_bias[v] + dot);
                let combined = round_bf16(base + bias);
                let val = cap.apply(combined);
                if val > best_val {
                    best_val = val;
                    best_id = v;
                }
            }
            (best_id, best_val)
        };
        if workers == 1 {
            return eval_range(0, vocab).0;
        }
        let chunk = vocab.div_ceil(workers);
        let mut results: Vec<(usize, f32)> = Vec::new();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            let mut start = 0usize;
            while start < vocab {
                let end = (start + chunk).min(vocab);
                handles.push(scope.spawn(move || eval_range(start, end)));
                start = end;
            }
            for handle in handles { results.push(handle.join().unwrap()); }
        });
        // Merge in rising index order so lower token ID wins ties.
        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (id, val) in results {
            if val > best_val || (val == best_val && id < best_id) {
                best_val = val;
                best_id = id;
            }
        }
        best_id
    }

    /// Full draft pass: run the 3-layer backbone and Markov head walk, returning `gamma` proposed token IDs.
    pub fn propose_block(
        &self,
        chat: &MtpChat,
        anchor_token: u32,
        anchor_embed: &[f32],
        threads: usize,
    ) -> Result<Vec<u32>> {
        let draft_hidden = self.forward_block(chat, anchor_embed, threads)?;
        self.markov_walk_greedy(chat, anchor_token, &draft_hidden, threads)
    }

    /// Helper for converting an FP32 slice to BF16 `u16` bits (for synthetic fixtures in tests).
    pub fn encode_bf16(values: &[f32]) -> Vec<u16> { f32_to_bf16_vec(values) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> MtpConfig {
        MtpConfig {
            hidden_size: 4,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            intermediate_size: 6,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            local_layer_ids: vec![0],
            sliding_window_size: 4,
            block_size: 4,
            markov_rank: 2,
            target_layer_ids: vec![1, 3],
            vocab_size: 8,
            unpadded_vocab_size: 6,
            final_logit_softcapping: 30.0,
        }
    }

    fn det_vec(len: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut state = seed.wrapping_add(0x9e37_79b9);
        (0..len).map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            round_bf16(unit * scale)
        }).collect()
    }

    fn synthetic_raw(cfg: &MtpConfig) -> MtpRawWeights {
        let h = cfg.hidden_size;
        let q_w = cfg.q_width();
        let kv_w = cfg.kv_width();
        let inter = cfg.intermediate_size;
        let rank = cfg.markov_rank;
        let taps = cfg.target_layer_ids.len();
        let mut s = 1u32;
        let mut next = |n: usize, scale: f32| { s += 1; det_vec(n, s, scale) };
        let layers = (0..cfg.num_layers).map(|_| MtpLayerWeights {
            dflare_norm: vec![1.0; h],
            input_layernorm: vec![1.0; h],
            post_attention_layernorm: vec![1.0; h],
            q_norm: vec![1.0; cfg.head_dim],
            k_norm: vec![1.0; cfg.head_dim],
            k_norm_ctx: vec![1.0; cfg.head_dim],
            q_proj: f32_to_bf16_vec(&next(q_w * h, 0.4)),
            k_proj: f32_to_bf16_vec(&next(kv_w * h, 0.4)),
            v_proj: f32_to_bf16_vec(&next(kv_w * h, 0.4)),
            k_proj_ctx: f32_to_bf16_vec(&next(kv_w * h, 0.4)),
            v_proj_ctx: f32_to_bf16_vec(&next(kv_w * h, 0.4)),
            o_proj: f32_to_bf16_vec(&next(h * q_w, 0.4)),
            gate_proj: f32_to_bf16_vec(&next(inter * h, 0.4)),
            up_proj: f32_to_bf16_vec(&next(inter * h, 0.4)),
            down_proj: f32_to_bf16_vec(&next(h * inter, 0.4)),
        }).collect();
        MtpRawWeights {
            fuse_weight: next((cfg.num_layers + 1) * taps, 0.8),
            mask_embed: next(h, 0.5),
            final_norm: vec![1.0; h],
            layers,
            markov: MtpMarkovWeights {
                dflare_norm: vec![1.0; h],
                markov_w1: f32_to_bf16_vec(&next(cfg.vocab_size * rank, 0.5)),
                rnn_proj_weight: f32_to_bf16_vec(&next((3 * rank) * (2 * rank + 2 * h), 0.3)),
                rnn_proj_bias: next(3 * rank, 0.2),
                hidden_out_weight: next(h * rank, 0.5),
                hidden_out_bias: next(h, 0.2),
            },
            lm_head: f32_to_bf16_vec(&next(cfg.unpadded_vocab_size * h, 0.6)),
        }
    }

    #[test]
    fn config_json_parses_and_refuses_bad_fields() {
        let json = br#"{
            "text_config": {
                "hidden_size": 4, "vocab_size": 8, "unpadded_vocab_size": 6,
                "final_logit_softcapping": 30.0
            },
            "mtp_config": {
                "hidden_size": 4, "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 2, "intermediate_size": 6,
                "max_position_embeddings": 64, "torch_dtype": "bfloat16",
                "q_bias": false, "o_bias": false, "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0, "local_layer_ids": [0], "sliding_window_size": 4,
                "block_size": 4, "markov_rank": 2, "markov_head_type": "rnn_hidden",
                "target_layer_ids": [1, 3]
            }
        }"#;
        let cfg = MtpConfig::from_config_json(json).unwrap();
        assert_eq!(cfg.gamma(), 3);
        assert!(cfg.is_local(0) && !cfg.is_local(1));
        let bad_bias = String::from_utf8(json.to_vec()).unwrap().replace("\"q_bias\": false", "\"q_bias\": true");
        assert!(MtpConfig::from_config_json(bad_bias.as_bytes()).is_err());
        let bad_type = String::from_utf8(json.to_vec()).unwrap().replace("rnn_hidden", "vanilla");
        assert!(MtpConfig::from_config_json(bad_type.as_bytes()).is_err());
    }

    #[test]
    fn folded_head_matches_unfolded_and_is_thread_invariant() {
        let cfg = sample_config();
        let raw = synthetic_raw(&cfg);
        let model1 = MtpModel::from_raw(cfg.clone(), raw.clone(), 1).unwrap();
        let model4 = MtpModel::from_raw(cfg.clone(), raw.clone(), 4).unwrap();
        assert_eq!(model1.fold_weight, model4.fold_weight);
        assert_eq!(model1.fold_bias, model4.fold_bias);

        let mut chat = model1.chat();
        let t0 = det_vec(cfg.hidden_size, 100, 0.5);
        let t1 = det_vec(cfg.hidden_size, 101, 0.5);
        model1.push_context(&mut chat, &[&t0, &t1]).unwrap();
        let t2 = det_vec(cfg.hidden_size, 102, 0.5);
        let t3 = det_vec(cfg.hidden_size, 103, 0.5);
        model1.push_context(&mut chat, &[&t2, &t3]).unwrap();

        let anchor = det_vec(cfg.hidden_size, 200, 0.5);
        let p1 = model1.propose_block(&chat, 2, &anchor, 1).unwrap();
        let p4 = model4.propose_block(&chat, 2, &anchor, 4).unwrap();
        assert_eq!(p1.len(), cfg.gamma());
        assert_eq!(p1, p4);
        for &tok in &p1 { assert!((tok as usize) < cfg.unpadded_vocab_size); }

        // Check folded bias matches unfolded lm_head @ (hidden_out_w @ latent + hidden_out_b) closely.
        let latent = det_vec(cfg.markov_rank, 300, 0.5);
        let mut delta_h = vec![0.0f64; cfg.hidden_size];
        for (h, slot) in delta_h.iter_mut().enumerate() {
            let mut s = raw.markov.hidden_out_bias[h] as f64;
            for (r, &l) in latent.iter().enumerate() {
                s += (raw.markov.hidden_out_weight[h * cfg.markov_rank + r] as f64) * (l as f64);
            }
            *slot = s;
        }
        for v in 0..cfg.unpadded_vocab_size {
            let mut unfolded = 0.0f64;
            for (h, &d) in delta_h.iter().enumerate() {
                unfolded += (bf16_to_f32(raw.lm_head[v * cfg.hidden_size + h]) as f64) * d;
            }
            let mut folded = model1.fold_bias[v] as f64;
            for (r, &l) in latent.iter().enumerate() {
                folded += (bf16_to_f32(model1.fold_weight[v * cfg.markov_rank + r]) as f64) * (l as f64);
            }
            assert!((unfolded - folded).abs() < 2e-2, "v={v}: unfolded={unfolded} folded={folded}");
        }
    }

    #[test]
    fn neox_rope_preserves_norm_and_rotates_half_split() {
        let mut v = vec![3.0f32, 4.0, 0.0, 0.0];
        let inv = [1.0f64, 0.25];
        apply_neox_rope_inplace(&mut v, 1, 4, 0, &inv);
        assert_eq!(v, [3.0, 4.0, 0.0, 0.0]);
        let mut w = vec![1.0f32, 0.0, 0.0, 1.0];
        apply_neox_rope_inplace(&mut w, 1, 4, 1, &inv);
        let c0 = 1.0f64.cos() as f32;
        let s0 = 1.0f64.sin() as f32;
        assert!((w[0] - c0).abs() < 2e-2 && (w[2] - s0).abs() < 2e-2);
    }

    #[test]
    fn empty_context_is_refused() {
        let cfg = sample_config();
        let model = MtpModel::from_raw(cfg.clone(), synthetic_raw(&cfg), 1).unwrap();
        let chat = model.chat();
        let anchor = det_vec(cfg.hidden_size, 50, 0.5);
        assert!(model.propose_block(&chat, 0, &anchor, 1).is_err());
    }
}
