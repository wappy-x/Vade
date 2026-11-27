//! Fixed-order f32 host reference for the block mix of a block residual.
//! Inside a block the stream holds only the sum that the layers of that block
//! added. A layer reads the past through a softmax blend over the finished
//! block vectors and the stream, with one learned query for each result.
//! The end mix before the last norm is the same operation with one query.
//! Source: Transformers v5.24.0, commit `eeee11099135592f1d03c2871fbd3d8132e5a4d7`,
//! `models/inkling2_loop/modeling_inkling2_loop.py`. Plain norm: lines 67-76.
//! Score, softmax and blend: lines 126 and 137-139. Mix: lines 828-853.
//! End mix: line 1442. This file holds its own code and none of that source.
//! All arithmetic is f32. A product and a sum are separate f32 operations,
//! never a fused multiply-add. The sum orders are the engine's: a chain in
//! rising index from zero, and for a sum of squares the fixed 32-lane tree.
//! The exponential is the one of the shared `fixed_math` module, not the
//! platform's. This module does not enable a model plan or change serving.
//! See `block_mix/README.md`.
use crate::{Result, fixed_math, invalid};

/// Epsilon of the plain norm inside a mix. The source builds that norm with
/// its own default (line 68), not with the config field `rms_norm_eps`.
pub const NORM_EPSILON: f32 = 1.0e-6;

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("block mix: {message}"))) }
}
fn finite(x: &[f32]) -> Result<()> { need(x.iter().all(|v| v.is_finite()), "values must be finite") }

/// Join 32 lane sums at distances 16, 8, 4, 2, 1: at each distance a lane
/// below the distance adds its partner above it. This is the join of the
/// engine's norm code (`square_sum` in `gpu/src/causal_ops.rs`).
pub(crate) fn join_lanes(mut lanes: [f32; 32]) -> f32 {
    for distance in [16, 8, 4, 2, 1] {
        for lane in 0..distance { lanes[lane] += lanes[lane + distance]; }
    }
    lanes[0]
}
/// The sum of squares in the engine's fixed 32-lane tree: the shared `norm_cap::tree_sum_squares`.
#[cfg(test)]
pub(crate) fn square_sum(x: &[f32]) -> f32 { crate::norm_cap::tree_sum_squares(x) }
/// Plain norm with no learned weight: `x * (1 / sqrt(mean(x^2) + eps))`.
/// The sum of squares uses the 32-lane tree. The result is not rounded.
/// The arithmetic is the one shared function `norm_cap::tree_plain_norm`.
pub fn plain_norm(x: &[f32], eps: f32) -> Result<Vec<f32>> {
    need(!x.is_empty(), "norm input is empty")?;
    need(eps.is_finite() && eps > 0.0, "invalid norm epsilon")?;
    finite(x)?;
    // The shared function gives zeros when the squares overflow. A mix refuses that input.
    need((crate::norm_cap::tree_sum_squares(x) / x.len() as f32 + eps).is_finite(), "norm exceeds f32 range")?;
    crate::norm_cap::tree_plain_norm(x, eps)
}

/// Result of one mix. `weights` is `[query][entry]`, with the stream as the
/// last entry. `values` is `[query][hidden]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mix {
    pub weights: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
}

/// Blend the finished block vectors and the stream, once for each query row.
/// `queries` is row-major `[rows][hidden]`. `blocks` is oldest first. The
/// stream goes last. Each entry is normed first, and the result is a blend of
/// the normed entries, not of the raw entries.
/// Order: norm every entry. For each query row, each score is the dot product
/// as a chain in rising index from zero, then one product with the f32 value
/// of `hidden^-0.5`. The softmax takes the largest score out first, adds the
/// exponentials as a chain in entry order and divides each one by the sum.
/// The blend adds `weight * normed` as a chain in entry order from zero.
pub fn mix(hidden: usize, queries: &[f32], blocks: &[&[f32]], stream: &[f32]) -> Result<Mix> {
    need(hidden > 0 && !queries.is_empty() && queries.len().is_multiple_of(hidden), "query shape differs from the hidden size")?;
    need(stream.len() == hidden && blocks.iter().all(|b| b.len() == hidden), "entry shape differs from the hidden size")?;
    finite(queries)?;
    let mut normed = Vec::with_capacity(blocks.len() + 1);
    for entry in blocks.iter().copied().chain([stream]) { normed.push(plain_norm(entry, NORM_EPSILON)?); }
    // The source multiplies the f32 scores by the scalar `hidden ** -0.5`.
    let scale = (1.0 / (hidden as f64).sqrt()) as f32;
    let (mut weights, mut values) = (Vec::new(), Vec::new());
    for query in queries.chunks_exact(hidden) {
        let mut scores = Vec::with_capacity(normed.len());
        for entry in &normed {
            let mut sum = 0.0f32;
            for d in 0..hidden { sum += query[d] * entry[d]; }
            scores.push(sum * scale);
        }
        softmax(&mut scores)?;
        let mut out = vec![0.0f32; hidden];
        for (weight, entry) in scores.iter().zip(&normed) {
            for d in 0..hidden { out[d] += weight * entry[d]; }
        }
        finite(&out)?;
        weights.push(scores);
        values.push(out);
    }
    Ok(Mix { weights, values })
}

/// The mix before the last norm: every finished block and the last stream,
/// with the one output query. The caller applies the last norm next.
pub fn end_mix(hidden: usize, output_query: &[f32], blocks: &[&[f32]], stream: &[f32]) -> Result<Vec<f32>> {
    need(output_query.len() == hidden, "the end mix takes one query")?;
    let mut mixed = mix(hidden, output_query, blocks, stream)?;
    Ok(mixed.values.swap_remove(0))
}

// No sink and no position term here.
fn softmax(scores: &mut [f32]) -> Result<()> {
    finite(scores)?;
    let mut max = f32::NEG_INFINITY;
    for &score in scores.iter() { max = max.max(score); }
    let mut sum = 0.0f32;
    for score in scores.iter_mut() { *score = fixed_math::exp(*score - max); sum += *score; }
    need(sum.is_finite() && sum > 0.0, "invalid softmax sum")?;
    for score in scores { *score /= sum; }
    Ok(())
}

#[cfg(test)]
mod tests;
