//! Fixed-order host reference for short causal convolutions over the token axis.
//! Two kinds exist. One kind has fixed taps for each channel. The other kind adds
//! taps made from the current token to static taps. See `short_conv/README.md`.
//!
//! This module does not admit a checkpoint or change serving. All sizes come
//! from the caller. The source is the public Transformers release v5.24.0,
//! commit `eeee11099135592f1d03c2871fbd3d8132e5a4d7`, file
//! `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`:
//! lines 285-325 (the causal convolution calls), 328-393 (fixed taps) and
//! 396-455 (taps made from the token). No source code is copied here.
//!
//! Arithmetic is f32. Each channel adds its tap products in one chain from
//! +0.0, the oldest token first. Then it adds the input of the current token and
//! multiplies one time by the f32 reciprocal of the square root of the term
//! count (`reciprocal`). The result is rounded to BF16 one time. Products and
//! sums are separate f32 operations.
//! The source leaves the order of the taps to its library. This order is the
//! order of the engine's device operation for the same convolution.
use crate::{Result, fixed_math::rn as round_bf16, invalid};

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("short convolution: {message}"))) }
}
fn mul(a: usize, b: usize) -> Result<usize> {
    let n = a.checked_mul(b).ok_or_else(|| invalid("short convolution: dimension product overflow"))?;
    need(n.checked_mul(std::mem::size_of::<f32>()).is_some_and(|bytes| bytes <= isize::MAX as usize), "buffer exceeds the address range")?;
    Ok(n)
}
fn finite(x: &[f32]) -> Result<()> { need(x.iter().all(|v| v.is_finite()), "values must be finite") }
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

/// The token that a stored tap weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapOrder {
    /// Stored tap 0 weights the oldest token of the window. The last tap weights the current token.
    OldestFirst,
    /// Stored tap 0 weights the current token. Stored tap `n` weights the token `n` steps back.
    NewestFirst,
}
impl TapOrder {
    /// Steps back from the current token for stored tap `tap` of `taps`.
    pub const fn lag(self, tap: usize, taps: usize) -> usize {
        match self { Self::OldestFirst => taps - 1 - tap, Self::NewestFirst => tap }
    }
    /// The stored tap that weights the token `lag` steps back.
    pub const fn tap(self, lag: usize, taps: usize) -> usize { self.lag(lag, taps) }
}
/// Layout of a fixed-tap weight row `[channel][tap]`. Source lines 314-322 pad
/// the sequence on the left, and lines 293-299 put the state before the new token.
/// This layout is inferred from those lines. A reference run must confirm it.
pub const FIXED_TAP_ORDER: TapOrder = TapOrder::OldestFirst;
/// Layout of the static taps `[tap][channel]` and of each head's projected taps.
/// Source line 410 states that tap 0 weights the current token. Line 446 agrees.
pub const TOKEN_TAP_ORDER: TapOrder = TapOrder::NewestFirst;
/// Index of one projected tap. The projection gives `[head][tap]` (source line 431).
pub const fn projected_tap(head: usize, tap: usize, taps: usize) -> usize { head * taps + tap }
/// The form of a 32-bit tensor divided by a number: a product with the
/// reciprocal. The reciprocal is `1 / divisor` in f64, rounded to f32 one time.
/// The source writes a division (lines 392 and 454). Its GPU run multiplies:
/// MEASURED on that run, 4,000,000 of 4,000,000 values equal this form for
/// each of the divisors sqrt(5), 3, 30 and sqrt(7), and a true f32 division
/// equals fewer. Lab receipt, not in this repository:
/// `ink/ref/contract-A/divform.json`.
pub fn reciprocal(divisor: f64) -> f32 { (1.0 / divisor) as f32 }
/// The factor of a convolution: `reciprocal(sqrt(terms))`. A convolution
/// multiplies by it one time. `terms` counts the taps and the added input.
fn factor(terms: usize) -> f32 { reciprocal((terms as f64).sqrt()) }

/// The last `taps - 1` inputs of one convolution, oldest first. A new history
/// holds zeros: tokens before the start of the text count as zero.
#[derive(Debug, Clone, PartialEq)]
pub struct History { channels: usize, taps: usize, data: Vec<f32> }
impl History {
    pub fn new(channels: usize, taps: usize) -> Result<Self> {
        need(channels > 0 && taps > 0, "dimensions must be positive")?;
        let count = mul(channels, taps - 1)?;
        let mut data = Vec::new();
        data.try_reserve_exact(count).map_err(|_| invalid("short convolution: cannot reserve history storage"))?;
        data.resize(count, 0.0);
        Ok(Self { channels, taps, data })
    }
    pub fn channels(&self) -> usize { self.channels }
    pub fn taps(&self) -> usize { self.taps }
    /// Kept inputs in `[step][channel]` order, oldest step first.
    pub fn inputs(&self) -> &[f32] { &self.data }
    /// The input `lag` steps back from the next token, with `1 <= lag < taps`.
    fn past(&self, lag: usize) -> &[f32] {
        let row = self.taps - 1 - lag;
        &self.data[row * self.channels..(row + 1) * self.channels]
    }
    /// Record one input after its output is computed.
    pub fn push(&mut self, input: &[f32]) -> Result<()> {
        need(input.len() == self.channels, "input width differs from history")?;
        finite(input)?;
        if self.taps > 1 {
            self.data.copy_within(self.channels.., 0);
            let start = (self.taps - 2) * self.channels;
            self.data[start..].copy_from_slice(input);
        }
        Ok(())
    }
}
fn sample<'a>(history: &'a History, input: &'a [f32], lag: usize) -> &'a [f32] {
    if lag == 0 { input } else { history.past(lag) }
}
fn check_step(history: &History, channels: usize, taps: usize, input: &[f32]) -> Result<()> {
    need(history.channels == channels && history.taps == taps, "history shape differs from the convolution")?;
    need(input.len() == channels, "input width differs from the convolution")?;
    finite(input)
}

/// Fixed taps: `y = (sum over taps of w[channel][tap] * u + u_now) / sqrt(taps + 1)`.
/// `weights` is row-major `[channel][tap]` in [`FIXED_TAP_ORDER`]. Values are f32.
#[derive(Debug, Clone, Copy)]
pub struct FixedTaps<'a> { channels: usize, taps: usize, weights: &'a [f32] }
impl<'a> FixedTaps<'a> {
    pub fn new(channels: usize, taps: usize, weights: &'a [f32]) -> Result<Self> {
        need(channels > 0 && taps > 0, "dimensions must be positive")?;
        need(weights.len() == mul(channels, taps)?, "fixed tap shape differs from the dimensions")?;
        finite(weights)?;
        Ok(Self { channels, taps, weights })
    }
    pub fn channels(&self) -> usize { self.channels }
    pub fn taps(&self) -> usize { self.taps }
    /// Compute one token. The history does not change.
    pub fn apply(&self, history: &History, input: &[f32]) -> Result<Vec<f32>> {
        check_step(history, self.channels, self.taps, input)?;
        let factor = factor(self.taps + 1);
        let mut output = Vec::with_capacity(self.channels);
        for channel in 0..self.channels {
            let mut sum = 0.0f32;
            for lag in (0..self.taps).rev() {
                let tap = FIXED_TAP_ORDER.tap(lag, self.taps);
                sum += self.weights[channel * self.taps + tap] * sample(history, input, lag)[channel];
            }
            output.push(round((sum + input[channel]) * factor)?);
        }
        Ok(output)
    }
    /// Compute one token, then record its input. An error leaves the history unchanged.
    pub fn step(&self, history: &mut History, input: &[f32]) -> Result<Vec<f32>> {
        let output = self.apply(history, input)?;
        history.push(input)?;
        Ok(output)
    }
}

/// Taps made from the token. For channel `c` in head `h` and stored tap `n`:
/// `tap = static_taps[n][c] + projected[h][n]`, with `projected = tap_proj * source`.
/// `y = (sum over taps of tap * u + u_now) / sqrt(2 * taps + 1)`.
/// `static_taps` is `[tap][channel]` and `tap_proj` is `[heads * taps][source]`,
/// both in [`TOKEN_TAP_ORDER`] and both exact BF16 values. The taps of a token
/// come from that token only. They weight the window of inputs that ends at it.
#[derive(Debug, Clone, Copy)]
pub struct TokenTaps<'a> { channels: usize, heads: usize, taps: usize, source: usize, static_taps: &'a [f32], tap_proj: &'a [f32] }
impl<'a> TokenTaps<'a> {
    pub fn new(channels: usize, heads: usize, taps: usize, source: usize, static_taps: &'a [f32], tap_proj: &'a [f32]) -> Result<Self> {
        need(channels > 0 && heads > 0 && taps > 0 && source > 0, "dimensions must be positive")?;
        need(channels.is_multiple_of(heads), "channels must be a multiple of heads")?;
        need(static_taps.len() == mul(taps, channels)?, "static tap shape differs from the dimensions")?;
        need(tap_proj.len() == mul(mul(heads, taps)?, source)?, "tap projection shape differs from the dimensions")?;
        exact(static_taps)?; exact(tap_proj)?;
        Ok(Self { channels, heads, taps, source, static_taps, tap_proj })
    }
    pub fn channels(&self) -> usize { self.channels }
    pub fn taps(&self) -> usize { self.taps }
    /// Projected taps `[head][tap]` for one token: a BF16 product, widened to f32.
    pub fn projected(&self, source: &[f32]) -> Result<Vec<f32>> {
        need(source.len() == self.source, "source width differs from the tap projection")?;
        exact(source)?;
        let mut output = Vec::with_capacity(self.heads * self.taps);
        for row in self.tap_proj.chunks_exact(self.source) {
            let mut sum = 0.0f32;
            for i in 0..self.source { sum += row[i] * source[i]; }
            output.push(round(sum)?);
        }
        Ok(output)
    }
    /// Compute one token. `source` is the vector that the taps are made from.
    /// The history does not change.
    pub fn apply(&self, history: &History, input: &[f32], source: &[f32]) -> Result<Vec<f32>> {
        check_step(history, self.channels, self.taps, input)?;
        let projected = self.projected(source)?;
        let (factor, head_width) = (factor(2 * self.taps + 1), self.channels / self.heads);
        let mut output = Vec::with_capacity(self.channels);
        for channel in 0..self.channels {
            let head = channel / head_width;
            let mut sum = 0.0f32;
            for lag in (0..self.taps).rev() {
                let tap = TOKEN_TAP_ORDER.tap(lag, self.taps);
                let weight = self.static_taps[tap * self.channels + channel] + projected[projected_tap(head, tap, self.taps)];
                sum += weight * sample(history, input, lag)[channel];
            }
            output.push(round((sum + input[channel]) * factor)?);
        }
        Ok(output)
    }
    /// Compute one token, then record its input. An error leaves the history unchanged.
    pub fn step(&self, history: &mut History, input: &[f32], source: &[f32]) -> Result<Vec<f32>> {
        let output = self.apply(history, input, source)?;
        history.push(input)?;
        Ok(output)
    }
}

/// One of the two kinds, for a caller that takes the kind from the tensors.
#[derive(Debug, Clone, Copy)]
pub enum Taps<'a> { Fixed(FixedTaps<'a>), TokenMade(TokenTaps<'a>) }
impl Taps<'_> {
    pub fn channels(&self) -> usize { match self { Self::Fixed(c) => c.channels, Self::TokenMade(c) => c.channels } }
    pub fn taps(&self) -> usize { match self { Self::Fixed(c) => c.taps, Self::TokenMade(c) => c.taps } }
    /// Compute one token. Fixed taps do not read `source`.
    pub fn apply(&self, history: &History, input: &[f32], source: &[f32]) -> Result<Vec<f32>> {
        match self { Self::Fixed(c) => c.apply(history, input), Self::TokenMade(c) => c.apply(history, input, source) }
    }
}

/// The four convolutions of one layer visit, in the order of the source's state
/// index: key 0, value 1, after attention 2, after the experts 3 (source lines
/// 562-567 and 868-873).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot { Key, Value, AfterAttention, AfterExperts }
impl Slot {
    pub const ALL: [Self; 4] = [Self::Key, Self::Value, Self::AfterAttention, Self::AfterExperts];
    pub const fn index(self) -> usize { self as usize }
}
/// The four histories of one layer visit for one sequence. Each visit of a
/// layer has its own set. A clone keeps all four histories.
#[derive(Debug, Clone, PartialEq)]
pub struct VisitHistories { slots: [History; 4] }
impl VisitHistories {
    /// `kv_width` is the channel count of the key and value convolutions.
    /// `hidden_size` is the channel count of the other two.
    pub fn new(kv_width: usize, hidden_size: usize, taps: usize) -> Result<Self> {
        Ok(Self { slots: [History::new(kv_width, taps)?, History::new(kv_width, taps)?,
            History::new(hidden_size, taps)?, History::new(hidden_size, taps)?] })
    }
    pub fn get(&self, slot: Slot) -> &History { &self.slots[slot.index()] }
    pub fn get_mut(&mut self, slot: Slot) -> &mut History { &mut self.slots[slot.index()] }
    /// Kept f32 values in all four histories.
    pub fn elements(&self) -> usize { self.slots.iter().map(|h| h.data.len()).sum() }
}

#[cfg(test)]
mod tests;
