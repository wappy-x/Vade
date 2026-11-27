//! Host reference of the two ends of a text model that has a weighted norm
//! after the embedding and a soft cap on its logits. One token.
//!
//! Source (read for the math, no code copied): the public `transformers`
//! release v5.24.0, `models/inkling2_loop/modeling_inkling2_loop.py`.
//! Embedding and its norm: lines 1240-1246 and 1322. Last norm: lines
//! 1442-1443. Head and logit cap: lines 1814-1816. Head rows that are used:
//! `core_model_loading.py` lines 856-878.
//!
//! Number contract of each step (FP32 = 32-bit float):
//! - Embedding row: BF16. Its weighted norm: FP32, rounded to BF16.
//! - Last norm: FP32 input, FP32 arithmetic, rounded to BF16.
//! - Head: BF16 weights and BF16 input. Each logit is one rising FP32 chain
//!   from 0.0 over the columns, each product rounded before its add, then
//!   rounded to BF16.
//! - Logit cap: FP32, not rounded.
//!
//! The head can store more rows than the model can write. Only the first
//! `used_rows` rows are read. The other rows are padding and are not zeros,
//! so an engine that reads them can pick a token that does not exist.
//! The sum order inside the head product is this reference's contract. The
//! source leaves it to its library, so last-bit parity is not claimed.
use crate::{Result, invalid, norm_cap::{SoftCap, round_bf16, weighted_norm}};

/// The input vector of the first layer: the embedding row of `token`, then the
/// weighted norm, then one BF16 rounding. `table` is `[rows, hidden]` BF16 values.
pub fn embed(table: &[f32], hidden: usize, token: usize, norm_weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    if hidden == 0 || table.is_empty() || !table.len().is_multiple_of(hidden) { return Err(invalid("embedding table must be rows by hidden")); }
    if token >= table.len() / hidden { return Err(invalid("token is outside the embedding table")); }
    embed_row(&table[token * hidden..(token + 1) * hidden], norm_weight, epsilon)
}
/// The same step for one embedding row that the caller already holds.
pub fn embed_row(row: &[f32], norm_weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    exact(row, "embedding row")?; exact(norm_weight, "embedding norm weight")?;
    rounded(weighted_norm(row, norm_weight, epsilon)?, "embedding norm")
}

/// The last norm: weighted norm of the FP32 output of the final mix, then one BF16 rounding.
pub fn last_norm(mixed: &[f32], norm_weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    exact(norm_weight, "last norm weight")?;
    rounded(weighted_norm(mixed, norm_weight, epsilon)?, "last norm")
}

/// BF16 logits of the first `used_rows` head rows, before the cap: `round(head[i] . y)`.
/// `head` is `[stored rows, hidden]` BF16 values. `y` is the BF16 output of
/// [`last_norm`]. Rows at `used_rows` and above are never read.
pub fn head_rows(head: &[f32], hidden: usize, used_rows: usize, y: &[f32]) -> Result<Vec<f32>> {
    if hidden == 0 || y.len() != hidden || !head.len().is_multiple_of(hidden) || used_rows == 0 || used_rows > head.len() / hidden {
        return Err(invalid("head must be stored rows by hidden, with 1 or more used rows inside it"));
    }
    let head = &head[..used_rows * hidden];
    exact(y, "head input")?; exact(head, "head weight")?;
    let mut logits = Vec::with_capacity(used_rows);
    for row in head.chunks_exact(hidden) {
        let mut sum = 0.0f32;
        for (&w, &v) in row.iter().zip(y) { sum += w * v; }
        let logit = round_bf16(sum);
        if !logit.is_finite() { return Err(invalid("head logit is not finite in BF16")); }
        logits.push(logit);
    }
    Ok(logits)
}

/// Capped logits of the first `used_rows` head rows: the soft cap of `round(head[i] . y)`.
/// The cap is FP32 and is not rounded. See [`head_rows`] for the inputs.
pub fn head_logits(head: &[f32], hidden: usize, used_rows: usize, y: &[f32], cap: SoftCap) -> Result<Vec<f32>> {
    Ok(head_rows(head, hidden, used_rows, y)?.into_iter().map(|logit| cap.apply(logit)).collect())
}

/// The token with the largest logit. The lower token number wins a tie.
pub fn top_token(logits: &[f32]) -> Result<usize> {
    if logits.is_empty() || logits.iter().any(|v| v.is_nan()) { return Err(invalid("top token needs logits that are numbers")); }
    let mut best = 0usize;
    for (token, &value) in logits.iter().enumerate() { if value > logits[best] { best = token; } }
    Ok(best)
}

fn exact(values: &[f32], what: &str) -> Result<()> {
    if values.iter().any(|&v| !v.is_finite() || round_bf16(v).to_bits() != v.to_bits()) { return Err(invalid(format!("{what} must hold finite BF16 values"))); }
    Ok(())
}
fn rounded(mut values: Vec<f32>, what: &str) -> Result<Vec<f32>> {
    for value in &mut values { *value = round_bf16(*value); }
    if values.iter().any(|v| !v.is_finite()) { return Err(invalid(format!("{what} is not finite in BF16"))); }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash::nvfp4::bf16_to_f32;
    fn cap() -> SoftCap { SoftCap::new(30.0, "logit cap").unwrap() }
    #[test]
    fn embedding_norm_hand_values() {
        // Row [3, 4], weights [2, 0.5], no epsilon: the scale is 1 / sqrt(12.5).
        // FP32 results 1.6970563 and 0.5656854. BF16 steps there are 1/128 and 1/256.
        // 1.6970563 * 128 = 217.2, so 217 / 128 = 1.6953125. 0.5656854 * 256 = 144.8, so 145 / 256 = 0.56640625.
        assert_eq!(embed_row(&[3.0, 4.0], &[2.0, 0.5], 0.0).unwrap(), [1.695_312_5, 0.566_406_25]);
        // The table lookup takes the row of the token. Row 1 of 3 here.
        let table = [9.0, 9.0, 3.0, 4.0, -1.0, 0.5];
        assert_eq!(embed(&table, 2, 1, &[2.0, 0.5], 0.0).unwrap(), [1.695_312_5, 0.566_406_25]);
        // Row [2, 2] has mean square 4: scale 1/2. Weights [1, 4] give [1, 4].
        assert_eq!(embed(&[2.0, 2.0], 2, 0, &[1.0, 4.0], 0.0).unwrap(), [1.0, 4.0]);
        assert!(embed(&table, 2, 3, &[2.0, 0.5], 0.0).is_err());
        assert!(embed(&table, 4, 0, &[1.0; 4], 0.0).is_err());
        // A row or a weight that is not a BF16 value is refused.
        assert!(embed_row(&[1.0 + 1.0 / 1024.0, 4.0], &[2.0, 0.5], 0.0).is_err());
        assert!(embed_row(&[3.0, 4.0], &[0.3, 0.5], 0.0).is_err());
    }
    #[test]
    fn last_norm_takes_fp32_and_gives_bf16() {
        // The input may hold any FP32 value. [1 + 1/1024, 1 + 1/1024] has a scale of 1 / (1 + 1/1024).
        let x = 1.0 + 1.0 / 1024.0f32;
        let y = last_norm(&[x, x], &[1.0, -2.0], 0.0).unwrap();
        assert_eq!(y, [1.0, -2.0]);
        // [3, 4] with weights [2, 0.5]: the same hand values as the embedding norm.
        assert_eq!(last_norm(&[3.0, 4.0], &[2.0, 0.5], 0.0).unwrap(), [1.695_312_5, 0.566_406_25]);
        assert!(last_norm(&[3.0, 4.0], &[2.0], 0.0).is_err());
    }
    #[test]
    #[allow(clippy::excessive_precision)] // The hand values stay as written: 5 decimal places each.
    fn logit_cap_hand_values() {
        // Cap 30. 30 tanh(24/30) = 19.92110. 30 tanh(60/30) = 28.92083. 30 tanh(1) = 22.84782.
        // Head rows [1, 1], [5, 0], [0, 2.5], [-1, -1] with y = [12, 12] give 24, 60, 30, -24 before the cap.
        let head = [1.0, 1.0, 5.0, 0.0, 0.0, 2.5, -1.0, -1.0];
        let logits = head_logits(&head, 2, 4, &[12.0, 12.0], cap()).unwrap();
        for (got, hand) in logits.iter().zip([19.921_10, 28.920_83, 22.847_82, -19.921_10f32]) { assert!((got - hand).abs() < 2e-5, "{got} != {hand}"); }
        assert_eq!(logits, [cap().apply(24.0), cap().apply(60.0), cap().apply(30.0), cap().apply(-24.0)]);
        assert_eq!(head_rows(&head, 2, 4, &[12.0, 12.0]).unwrap(), [24.0, 60.0, 30.0, -24.0]);
        // The cap is odd, and every capped logit is inside the cap.
        assert_eq!(logits[3], -logits[0]);
        assert_eq!(head_logits(&[1.0, -1.0], 1, 2, &[32768.0], cap()).unwrap(), [30.0, -30.0]);
        // The cap keeps the top token: 60 before the cap, 28.92 after it.
        assert_eq!(top_token(&logits).unwrap(), 1);
    }
    #[test]
    fn head_product_rounds_to_bf16_before_the_cap() {
        // (1 + 1/128)^2 = 1 + 2/128 + 1/16384. BF16 keeps 1 + 2/128.
        let w = 1.0 + 1.0 / 128.0f32;
        assert_eq!(head_logits(&[w], 1, 1, &[w], cap()).unwrap(), [cap().apply(1.0 + 2.0 / 128.0)]);
        assert_ne!(head_logits(&[w], 1, 1, &[w], cap()).unwrap(), [cap().apply(w * w)]);
        // Rising FP32 chain: (2^24 + 1) - 2^24 = 0, not 1.
        assert_eq!(head_logits(&[16_777_216.0, 1.0, -16_777_216.0], 3, 1, &[1.0; 3], cap()).unwrap(), [0.0]);
    }
    #[test]
    fn padding_rows_are_never_read() {
        // 4 stored rows, 2 used. Rows 2 and 3 hold large numbers and a value that is not BF16.
        let head = [1.0, 0.0, 0.0, 1.0, 100.0, 100.0, 1.0 + 1.0 / 1024.0, f32::NAN];
        let logits = head_logits(&head, 2, 2, &[2.0, 3.0], cap()).unwrap();
        assert_eq!(logits, [cap().apply(2.0), cap().apply(3.0)]);
        assert_eq!(top_token(&logits).unwrap(), 1);
        // With all rows read, the padding row 2 would win, and row 3 is refused.
        assert_eq!(top_token(&head_logits(&head, 2, 3, &[2.0, 3.0], cap()).unwrap()).unwrap(), 2);
        assert!(head_logits(&head, 2, 4, &[2.0, 3.0], cap()).is_err());
        assert!(head_logits(&head, 2, 5, &[2.0, 3.0], cap()).is_err());
        assert!(head_logits(&head, 2, 0, &[2.0, 3.0], cap()).is_err());
        assert!(head_logits(&head, 2, 2, &[2.0], cap()).is_err());
        assert!(head_logits(&head, 2, 2, &[2.0, 1.0 + 1.0 / 1024.0], cap()).is_err());
    }
    #[test]
    fn cap_keeps_the_order_of_neighbor_bf16_logits() {
        // Every pair of neighbor BF16 values from 2^-10 up to 128 stays in strict order after the cap of 30.
        let (mut bits, mut pairs) = (0x3a80u16, 0u32);
        while bf16_to_f32(bits) < 128.0 {
            let (low, high) = (bf16_to_f32(bits), bf16_to_f32(bits + 1));
            assert!(cap().apply(low) < cap().apply(high), "order lost between {low} and {high}");
            assert!(cap().apply(-low) > cap().apply(-high));
            bits += 1; pairs += 1;
        }
        assert_eq!(pairs, 17 * 128);
        assert!(cap().apply(0.0) < cap().apply(bf16_to_f32(0x3a80)));
    }
    #[test]
    fn top_token_takes_the_lower_number_on_a_tie() {
        assert_eq!(top_token(&[1.0, 5.0, 5.0, -2.0]).unwrap(), 1);
        assert_eq!(top_token(&[-0.0, 0.0]).unwrap(), 0);
        assert_eq!(top_token(&[f32::NEG_INFINITY, -3.0]).unwrap(), 1);
        assert!(top_token(&[]).is_err() && top_token(&[1.0, f32::NAN]).is_err());
    }
}
